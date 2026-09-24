# SPDX-License-Identifier: Apache-2.0
"""Explicit worker-only OFF archives and ON numerical comparisons.

This is a correctness experiment, not a performance benchmark. Snapshots copy
on the current compute stream at the real consumer; they neither reload KV nor
wait on a connector/bank event. The readback perturbs timing and cannot prove
that an asynchronous execution without probes is free of races.
"""

import functools
import inspect
import json
import os
import re
from pathlib import Path

import torch
from layerwise_prefill_correctness_baseline import resolve_off_directory
from layerwise_prefill_correctness_layout import (
    install_local_merged_layout,
    validate_local_merged_engine,
)

# This extension is selected explicitly by the correctness launcher. Spawned
# workers need the same layout selection before constructing their engines.
install_local_merged_layout()


def record_identity(record):
    return tuple(record[name] for name in ("rank", "step", "layer", "kind", "name"))


def single_request_span(meta, prompt_len):
    if meta is None:
        raise RuntimeError("Correctness probe encountered a profiling/dummy forward")
    seq = meta.seq_lens_cpu
    query = meta.query_start_loc_cpu
    if len(seq) != 1 or len(query) != 2:
        raise RuntimeError("Correctness probe requires exactly one unpadded request")
    length = int(query[1]) - int(query[0])
    end = int(seq[0])
    if length != int(meta.num_actual_tokens) or not 0 < length <= end:
        raise RuntimeError("Invalid single-request logical token span")
    if end > prompt_len:
        return None
    if int(getattr(meta, "num_decode_tokens", 0)):
        raise RuntimeError("Decode-classified prefill tails are unsupported by the dense KV probe")
    return (end - length, end)


def valid_tensor_rows(meta, tensor, *, global_only=False, prefer_local=None):
    """Use actual scheduler/CP host metadata to exclude only padding rows."""
    if tensor.ndim == 0:
        return None
    rows = int(tensor.shape[0])
    actual = int(meta.num_actual_tokens)
    cp = getattr(meta, "dsa_cp_context", None)
    global_capacities = {actual, int(getattr(meta, "num_input_tokens", actual) or actual)}
    if cp is not None:
        global_capacities.add(int(cp.num_tokens_pad))
    global_match = rows in global_capacities
    local_match = cp is not None and rows == int(cp.local_end_with_pad - cp.local_start)
    local_valid = max(0, int(cp.local_end - cp.local_start)) if cp is not None else 0
    if global_only:
        if not global_match:
            raise RuntimeError("Global positions tensor shape disagrees with query metadata")
        return actual
    if local_match and global_match and local_valid != actual:
        if prefer_local is None:
            raise RuntimeError("Ambiguous global/local tensor rows without sequence-parallel context")
        return local_valid if prefer_local else actual
    if local_match:
        return local_valid
    if global_match:
        return actual
    raise RuntimeError("Tensor rows match neither global query nor CP-local capacity")


def slots_for_positions(cache, block_table, start, end):
    """Resolve logical tokens using the actual consumer's bank block table."""
    if cache.ndim < 3 or block_table is None or block_table.ndim != 2:
        raise RuntimeError("Unsupported paged KV layout or absent block table")
    if block_table.shape[0] != 1:
        raise RuntimeError("KV normalization requires one request block table")
    positions = torch.arange(start, end, dtype=torch.long)
    blocks = block_table[0].detach().to(device="cpu", dtype=torch.long)
    block_size = int(cache.shape[1])
    if positions.numel() and int(positions[-1]) // block_size >= blocks.numel():
        raise RuntimeError("Consumer block table does not cover the logical prefix")
    return blocks[positions // block_size] * block_size + positions % block_size


def rows_from_slots(cache, slots):
    """Snapshot on the current stream, without a new cross-stream dependency."""
    slots_cpu = slots.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    flat = cache.reshape(-1, *cache.shape[2:])
    if slots_cpu.numel() and (int(slots_cpu.min()) < 0 or int(slots_cpu.max()) >= flat.shape[0]):
        raise RuntimeError("Correctness KV snapshot contains padding/invalid slots")
    return flat.index_select(0, slots_cpu.to(cache.device))


class TensorArchive:
    """One tensor at a time: OFF stores; ON loads OFF and emits numerical stats."""

    def __init__(self, case_dir, rank, save_on_tensors=False, compare=None):
        self.root = Path(case_dir)
        if self.root.name not in ("off", "on"):
            raise ValueError("Correctness case directory must be named off or on")
        self.rank = int(rank)
        self.save = self.root.name == "off" or save_on_tensors
        self.directory = self.root / "tensors" / f"rank{self.rank}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "index.jsonl"
        self.index = self.index_path.open("x", encoding="utf-8")
        self.baseline = {}
        self.compare = compare
        self.records = {}
        self.errors = []
        self.archived_bytes = 0
        self.file_count = 0
        if self.root.name == "on":
            if self.compare is None:
                from layerwise_prefill_correctness_compare import compare_tensor_values

                self.compare = compare_tensor_values
            try:
                self.off_root = resolve_off_directory(self.root.parent)
                baseline_path = self.off_root / "tensors" / f"rank{self.rank}" / "index.jsonl"
                with baseline_path.open(encoding="utf-8") as source:
                    for line in source:
                        record = json.loads(line)
                        identity = record_identity(record)
                        if identity in self.baseline:
                            raise RuntimeError(f"Duplicate OFF tensor identity: {identity}")
                        self.baseline[identity] = record
            except BaseException:
                self.index.close()
                raise

    def record(self, tensor, *, step, layer, kind, name, span, positions=None, valid_rows=None):
        identity = (self.rank, step, layer, kind, name)
        if identity in self.records:
            error = f"Duplicate tensor observation: {identity}"
            self.errors.append(error)
            raise RuntimeError(error)
        if not isinstance(tensor, torch.Tensor):
            error = f"Required observation is not a tensor: {identity}"
            self.errors.append(error)
            raise TypeError(error)
        try:
            # clone orders a snapshot before subsequent operations on this
            # compute stream. No synchronize()/wait_event()/wait_stream().
            cpu = tensor.detach().contiguous().clone().to("cpu")
            floating = cpu.is_floating_point() or cpu.is_complex()
            finite_input = cpu.float() if str(cpu.dtype).startswith("torch.float8") else cpu
            record = dict(
                rank=self.rank,
                step=step,
                layer=layer,
                kind=kind,
                name=name,
                span=list(span),
                shape=list(cpu.shape),
                dtype=str(cpu.dtype),
                numel=cpu.numel(),
                path=None,
                nonfinite=int((~torch.isfinite(finite_input)).sum()) if floating else 0,
            )
            if valid_rows is not None and (cpu.ndim == 0 or not 0 <= valid_rows <= cpu.shape[0]):
                raise RuntimeError("Invalid tensor comparison row range")
            comparison_slice = (
                dict(axis=0, start=0, end=valid_rows) if valid_rows is not None and valid_rows < cpu.shape[0] else None
            )
            compared = cpu[:valid_rows] if comparison_slice is not None else cpu
            finite_compared = finite_input[:valid_rows] if comparison_slice is not None else finite_input
            record.update(
                comparison_slice=comparison_slice,
                comparison_shape=list(compared.shape),
                comparison_numel=compared.numel(),
                comparison_nonfinite=int((~torch.isfinite(finite_compared)).sum()) if floating else 0,
            )
            if positions is not None:
                record.update(positions=list(positions), token_axis=0)
            if self.root.name == "on":
                reference = self.baseline.get(identity)
                if reference is None or reference.get("span") != list(span):
                    record["comparison"] = dict(comparable=False, reason="OFF identity/span missing")
                elif reference.get("comparison_slice") != comparison_slice:
                    record["comparison"] = dict(comparable=False, reason="OFF valid comparison rows differ")
                else:
                    try:
                        source_path = (self.off_root / reference["path"]).resolve()
                        source_path.relative_to(self.off_root)
                        baseline = torch.load(source_path, map_location="cpu", weights_only=True)
                        reference_values = baseline[:valid_rows] if comparison_slice is not None else baseline
                        record["comparison"] = self.compare(reference_values, compared)
                        del reference_values
                        del baseline
                    except Exception as error:
                        record["comparison"] = dict(comparable=False, reason=str(error))
                if not record["comparison"].get("comparable", False):
                    self.errors.append(f"Cannot compare {identity}: {record['comparison'].get('reason')}")
            if self.save:
                path = self.directory / f"{len(self.records):07d}.pt"
                temporary = path.with_suffix(".pt.tmp")
                torch.save(cpu, temporary)
                os.replace(temporary, path)
                record["path"] = path.relative_to(self.root).as_posix()
                self.archived_bytes += path.stat().st_size
                self.file_count += 1
            self.index.write(json.dumps(record, allow_nan=False) + "\n")
            self.index.flush()
            self.records[identity] = record
            del cpu
        except BaseException as error:
            self.errors.append(f"Tensor archive failed at {identity}: {error}")
            raise

    def close(self):
        self.index.close()


def required_roles(decoder_layers, sfa_layers, indexer_layers, index_cache_layers, scale_layers):
    result = []

    def add(kind, names, layers, steps="all"):
        for name in names:
            result.append(dict(kind=kind, name=name, layers=list(layers), steps=steps))

    add("decoder", ("input", "output", "positions"), decoder_layers)
    add("sfa", ("input", "output"), sfa_layers)
    add("attention", ("query_nope", "query_rope", "topk", "output"), sfa_layers)
    add("indexer", ("query", "weights", "topk"), indexer_layers)
    add("indexer_input", ("x", "q_c"), indexer_layers)
    for kind, steps in (("kv_current", "all"), ("kv_loaded", "history")):
        add(kind, ("nope", "rope"), sfa_layers, steps)
        add(kind, ("index",), index_cache_layers, steps)
        add(kind, ("scale",), scale_layers, steps)
    return result


class CorrectnessProbe:
    def __init__(self, archive, prompt_len, layers, implementations, get_context, model_config):
        self.archive = archive
        self.prompt_len = int(prompt_len)
        self.layers = layers
        self.impls = implementations  # id(impl) -> (decoder index, attention module)
        self.get_context = get_context
        self.model_config = model_config
        self.steps = []
        self.active_decoder = []
        self.active_sfa = []
        self.handles = []
        self.patches = []
        self.excluded_calls = 0
        self.merged_sources = 0
        self.legacy_sources = 0
        self.indexer_layers = sorted(
            index for index, module in implementations.values() if module.impl.has_indexer and not module.impl.skip_topk
        )
        self.index_cache_layers = sorted(index for index, module in implementations.values() if module.impl.has_indexer)
        self.scale_layers = sorted(
            index
            for index, module in implementations.values()
            if module.impl.has_indexer and module.impl.use_sparse_c8_indexer
        )
        self.roles = required_roles(
            sorted(layers),
            sorted(layers),
            self.indexer_layers,
            self.index_cache_layers,
            self.scale_layers,
        )

    def patch(self, owner, name, factory):
        original = getattr(owner, name)
        self.patches.append((owner, name, original))
        setattr(owner, name, factory(original))

    def restore(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches.clear()

    def emit(self, tensor, kind, name, *, positions=None):
        if not self.active_sfa:
            raise RuntimeError("Tensor consumer executed outside its observed SFA")
        _, layer, meta, span = self.active_sfa[-1]
        self.archive.record(
            tensor,
            step=self.steps[-1]["step"],
            layer=layer,
            kind=kind,
            name=name,
            span=span,
            positions=positions,
            valid_rows=None if positions is not None else valid_tensor_rows(meta, tensor, prefer_local=True),
        )

    def kv(self, name, cache, table, slots):
        _, layer, _, span = self.active_sfa[-1]
        step = self.steps[-1]["step"]
        if (self.archive.rank, step, layer, "kv_current", name) in self.archive.records:
            return  # Same cache may be observed by indexer and attention.
        start, end = span
        slots = slots.reshape(-1)[: end - start]
        if slots.numel() != end - start:
            raise RuntimeError("Current KV slots do not cover every actual query token")
        self.emit(rows_from_slots(cache, slots), "kv_current", name, positions=span)
        if start:
            historical = slots_for_positions(cache, table, 0, start)
            self.emit(rows_from_slots(cache, historical), "kv_loaded", name, positions=(0, start))

    def decoder_pre(self, layer, module, args, kwargs):
        bound = inspect.signature(module.forward).bind(*args, **kwargs).arguments
        if "hidden_states" not in bound or "positions" not in bound:
            raise RuntimeError("Decoder forward does not expose positions/hidden_states")
        attention = next(item for index, item in self.impls.values() if index == layer)
        context = self.get_context()
        metadata = context.attn_metadata
        if not isinstance(metadata, dict) or attention.layer_name not in metadata:
            raise RuntimeError("Missing per-layer real forward metadata")
        span = single_request_span(metadata[attention.layer_name], self.prompt_len)
        if span is not None and layer == min(self.layers):
            previous_end = self.steps[-1]["span"][1] if self.steps else 0
            if span[0] != previous_end:
                raise RuntimeError("Main prefill steps overlap or leave a logical-token gap")
            if self.archive.rank == 0 and self.steps:
                print(
                    f"[PREFILL_CORRECTNESS] step={self.steps[-1]['step']} "
                    f"files={self.archive.file_count} bytes={self.archive.archived_bytes}",
                    flush=True,
                )
            self.steps.append(dict(step=len(self.steps), span=list(span), phase="prefill"))
        self.active_decoder.append((layer, span))
        if span is None:
            self.excluded_calls += 1
            return
        if not self.steps or self.steps[-1]["span"] != list(span):
            raise RuntimeError("Decoder layers disagree on the prefill query span")
        for name, tensor in (
            ("input", bound["hidden_states"]),
            ("positions", bound["positions"]),
            ("input_residual", bound.get("residual")),
        ):
            if tensor is not None:
                self.archive.record(
                    tensor,
                    step=self.steps[-1]["step"],
                    layer=layer,
                    kind="decoder",
                    name=name,
                    span=span,
                    valid_rows=valid_tensor_rows(
                        metadata[attention.layer_name],
                        tensor,
                        global_only=name == "positions",
                        prefer_local=getattr(context, "flash_comm_v1_enabled", None),
                    ),
                )

    def decoder_post(self, layer, module, args, kwargs, result):
        actual_layer, span = self.active_decoder.pop()
        if actual_layer != layer:
            raise RuntimeError("Decoder probe nesting mismatch")
        if span is None:
            return
        outputs = result if isinstance(result, (tuple, list)) else (result,)
        if not outputs or not isinstance(outputs[0], torch.Tensor):
            raise RuntimeError("Decoder did not return its hidden-state tensor")
        for index, tensor in enumerate(outputs):
            if tensor is not None:
                attention = next(item for idx, item in self.impls.values() if idx == layer)
                context = self.get_context()
                self.archive.record(
                    tensor,
                    step=self.steps[-1]["step"],
                    layer=layer,
                    kind="decoder",
                    name="output" if index == 0 else f"output_{index}",
                    span=span,
                    valid_rows=valid_tensor_rows(
                        context.attn_metadata[attention.layer_name],
                        tensor,
                        prefer_local=getattr(context, "flash_comm_v1_enabled", None),
                    ),
                )

    def install(self, sfa, torch_npu, connector_module, page_type):
        for layer, module in self.layers.items():
            self.handles.append(
                module.register_forward_pre_hook(functools.partial(self.decoder_pre, layer), with_kwargs=True)
            )
            self.handles.append(
                module.register_forward_hook(functools.partial(self.decoder_post, layer), with_kwargs=True)
            )

        def forward_factory(original):
            @functools.wraps(original)
            def forward(impl, layer_name, hidden_states, kv_cache, attn_metadata, *args, **kwargs):
                entry = self.impls.get(id(impl))
                if entry is None:
                    self.excluded_calls += 1
                    return original(impl, layer_name, hidden_states, kv_cache, attn_metadata, *args, **kwargs)
                if not self.active_decoder or self.active_decoder[-1][0] != entry[0]:
                    raise RuntimeError("Main SFA bypassed its decoder observation hook")
                span = single_request_span(attn_metadata, self.prompt_len)
                if span is None:
                    return original(impl, layer_name, hidden_states, kv_cache, attn_metadata, *args, **kwargs)
                if span != self.active_decoder[-1][1]:
                    raise RuntimeError("SFA and decoder spans disagree")
                self.active_sfa.append((impl, entry[0], attn_metadata, span))
                try:
                    self.emit(hidden_states, "sfa", "input")
                    result = original(impl, layer_name, hidden_states, kv_cache, attn_metadata, *args, **kwargs)
                    self.emit(result, "sfa", "output")
                    return result
                finally:
                    self.active_sfa.pop()

            return forward

        self.patch(sfa.AscendSFAImpl, "forward", forward_factory)

        def index_input_factory(original):
            signature = inspect.signature(original)

            @functools.wraps(original)
            def index_input(*args, **kwargs):
                if self.active_sfa:
                    values = signature.bind(*args, **kwargs).arguments
                    self.emit(values["x"], "indexer_input", "x")
                    self.emit(values["q_c"], "indexer_input", "q_c")
                return original(*args, **kwargs)

            return index_input

        self.patch(sfa.AscendSFAImpl, "indexer_select_post_process", index_input_factory)

        def kernel_factory(kind, tuple_output=False):
            def factory(original):
                @functools.wraps(original)
                def kernel(*args, **kwargs):
                    if not self.active_sfa:
                        return original(*args, **kwargs)
                    if args:
                        raise RuntimeError("Observed SFA kernel uses an unsupported positional schema")
                    impl, _, meta, _ = self.active_sfa[-1]
                    if kind == "indexer":
                        self.emit(kwargs["query"], "indexer", "query")
                        self.emit(kwargs["weights"], "indexer", "weights")
                        slots = meta.indexer_slot_mapping
                        slots = meta.slot_mapping if slots is None else slots
                        self.kv("index", kwargs["key"], kwargs["block_table"], slots)
                        if "key_dequant_scale" in kwargs:
                            self.kv("scale", kwargs["key_dequant_scale"], kwargs["block_table"], slots)
                            self.emit(kwargs["query_dequant_scale"], "indexer", "query_scale")
                    else:
                        if kwargs["block_table"] is not meta.block_table:
                            raise RuntimeError("Scratch/override attention is outside main prefill coverage")
                        self.emit(kwargs["query"], "attention", "query_nope")
                        self.emit(kwargs["query_rope"], "attention", "query_rope")
                        self.emit(kwargs["sparse_indices"], "attention", "topk")
                        self.kv("nope", kwargs["key"], kwargs["block_table"], meta.slot_mapping)
                        self.kv("rope", kwargs["key_rope"], kwargs["block_table"], meta.slot_mapping)
                        if impl.has_indexer and impl.skip_topk:
                            raise RuntimeError("Producer IndexCache reuse needs an explicit KV consumer probe")
                    result = original(*args, **kwargs)
                    self.emit(result[0] if tuple_output else result, kind, "topk" if kind == "indexer" else "output")
                    return result

                return kernel

            return factory

        operations = torch.ops._C_ascend
        self.patch(operations, "npu_sparse_flash_attention", kernel_factory("attention"))
        for owner, name, tuple_output in (
            (operations, "npu_lightning_indexer", False),
            (operations, "npu_lightning_indexer_quant", False),
            (torch_npu, "npu_lightning_indexer", True),
        ):
            if hasattr(owner, name):
                self.patch(owner, name, kernel_factory("indexer", tuple_output))

        def source_factory(original):
            @functools.wraps(original)
            def sources(*args, **kwargs):
                result = original(*args, **kwargs)
                caller = inspect.currentframe().f_back.f_code.co_name
                if caller == "batched_to_gpu":
                    for item in result:
                        if isinstance(item, page_type):
                            self.merged_sources += 1
                        else:
                            self.legacy_sources += 1
                return result

            return sources

        self.patch(connector_module, "_layer_source_memory_objs", source_factory)

    def finish(self, engine):
        self.restore()
        errors = list(self.archive.errors)
        if not self.steps or self.steps[-1]["span"][1] != self.prompt_len:
            errors.append("Main prefill observations do not reach the complete prompt")
        for role in self.roles:
            for step in self.steps:
                if role["steps"] == "history" and step["span"][0] == 0:
                    continue
                for layer in role["layers"]:
                    identity = (self.archive.rank, step["step"], layer, role["kind"], role["name"])
                    if identity not in self.archive.records:
                        errors.append(f"Missing required tensor {identity}")
        if self.active_decoder or self.active_sfa:
            errors.append("A probed forward did not finish")
        try:
            layout = validate_local_merged_engine(engine)
        except Exception as error:
            layout = {"error": str(error)}
            errors.append(f"Local merged engine validation failed: {error}")
        if self.archive.root.name == "on":
            missing = self.archive.baseline.keys() - self.archive.records.keys()
            if missing:
                errors.append(f"ON did not observe {len(missing)} OFF tensor identities")
        if any(step["span"][0] > 0 for step in self.steps) and self.merged_sources == 0:
            errors.append("No actual merged-page H2D source was observed")
        if self.legacy_sources:
            errors.append("Legacy nonmerged H2D source objects were observed")
        self.archive.close()
        return dict(
            rank=self.archive.rank,
            complete=not errors,
            errors=errors,
            records=len(self.archive.records),
            decoder_layers=sorted(self.layers),
            sfa_layers=sorted(self.layers),
            indexer_layers=self.indexer_layers,
            indexer_cache_layers=self.index_cache_layers,
            scale_layers=self.scale_layers,
            c8_layers=self.scale_layers,
            steps=self.steps,
            required_roles=self.roles,
            prompt_length=self.prompt_len,
            model_num_hidden_layers=int(self.model_config.num_hidden_layers),
            indexer_types=getattr(self.model_config, "indexer_types", None),
            archived_bytes=self.archive.archived_bytes,
            file_count=self.archive.file_count,
            merged_load_sources=self.merged_sources,
            legacy_load_sources=self.legacy_sources,
            layout=layout,
            excluded={"scope": "MTP/drafter and non-prefill execution", "calls": self.excluded_calls},
            observation="actual consumer, current compute stream; no added bank waits",
        )


def inventory(model, model_config, sfa_class):
    count = int(model_config.num_hidden_layers)
    layers, implementations = {}, {}
    for name, module in model.named_modules():
        match = re.search(r"(?:^|\.)layers\.(\d+)$", name)
        if match and "DecoderLayer" in type(module).__name__:
            index = int(match[1])
            if index in layers:
                raise RuntimeError(f"Ambiguous main decoder layer {index}")
            layers[index] = module
        impl = getattr(module, "impl", None)
        if isinstance(impl, sfa_class):
            layer_name = getattr(module, "layer_name", None)
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer_name or "")
            if not match:
                raise RuntimeError("SFA layer has no stable model-layer identity")
            implementations[id(impl)] = (int(match[1]), module)
    expected = list(range(count))
    if sorted(layers) != expected or sorted(index for index, _ in implementations.values()) != expected:
        raise RuntimeError("Static decoder/SFA inventory differs from model.num_hidden_layers")
    indexer_types = getattr(model_config, "indexer_types", None)
    if indexer_types is not None:
        if len(indexer_types) != count or any(value not in ("full", "shared") for value in indexer_types):
            raise RuntimeError("Unsupported model indexer_types schema")
        for index, module in implementations.values():
            expected_indexer = indexer_types[index] == "full"
            if bool(module.impl.has_indexer) != expected_indexer:
                raise RuntimeError(f"Layer {index} indexer presence disagrees with config.indexer_types")
            if not expected_indexer and not module.impl.skip_topk:
                raise RuntimeError(f"Shared consumer {index} does not consume shared top-k")
    return layers, implementations


class PrefillCorrectnessWorker:
    def install_correctness_probe(self, case_dir, prompt_len, save_on_tensors=False):
        import torch_npu
        from lmcache.v1.memory_management import LayerPageMemoryObj
        from lmcache_ascend.v1.npu_connector import npu_connectors
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        from vllm.forward_context import get_forward_context

        from vllm_ascend.attention import sfa_v1

        if hasattr(self, "_correctness_probe"):
            raise RuntimeError("Correctness probe was already installed")
        if not self.vllm_config.model_config.enforce_eager:
            raise RuntimeError("Full correctness observations require eager model execution")
        adapter = get_kv_transfer_group()._lmcache_engine
        engine = adapter.lmcache_engine
        facts = validate_local_merged_engine(engine)
        config = self.vllm_config.model_config
        model_config = getattr(config, "hf_text_config", None) or config.hf_config
        model = self.model_runner.model
        layers, implementations = inventory(model, model_config, sfa_v1.AscendSFAImpl)
        archive = TensorArchive(case_dir, self.rank, save_on_tensors)
        probe = CorrectnessProbe(archive, prompt_len, layers, implementations, get_forward_context, model_config)
        self._correctness_probe = probe
        self._correctness_engine = engine
        try:
            probe.install(sfa_v1, torch_npu, npu_connectors, LayerPageMemoryObj)
        except BaseException:
            probe.restore()
            archive.close()
            raise
        return dict(
            rank=int(self.rank),
            pid=os.getpid(),
            layout=facts,
            decoder_layers=sorted(layers),
            indexer_layers=probe.indexer_layers,
        )

    def finish_correctness_probe(self):
        if not hasattr(self, "_correctness_probe"):
            raise RuntimeError("Correctness probe was not installed")
        return self._correctness_probe.finish(self._correctness_engine)
