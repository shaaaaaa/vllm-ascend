# SPDX-License-Identifier: Apache-2.0
"""Opt-in, raw-tensor observations of real PD serving (never a transport shim).

This diagnostic deliberately reads device tensors and requires eager execution.
Each worker owns its recorder; importing this module installs no hooks. Files
contain request data, and incomplete calls remain explicitly incomplete.
"""

import functools
import inspect
import json
import os
import re
import socket
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import torch


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def cpu_tensor(value):
    return value.detach().to(device="cpu", copy=True).contiguous()


def safe_component(value):
    value = str(value)
    if not value or value in (".", "..") or len(value.encode("utf-8")) > 180:
        raise ValueError("Diagnostic request/run identity must be nonempty and at most 180 UTF-8 bytes")
    component = quote(value, safe="")
    if len(component) > 220:
        raise ValueError("Encoded diagnostic identity exceeds filesystem component limit")
    return component


def row_window(rows, actual, cp=None, prefer_local=True):
    if cp is not None:
        start, end = int(cp.local_start), int(cp.local_end)
        capacity = int(cp.local_end_with_pad) - start
        if rows == capacity and (prefer_local or rows != actual):
            return start, max(start, end)
    if rows >= actual:
        return 0, actual
    raise ValueError(f"Cannot align {rows} tensor rows with {actual} scheduled query rows")


class RequestArchive:
    def __init__(self, root, metadata):
        self.metadata = dict(metadata)
        worker = (
            f"{safe_component(metadata['host'])}-dp{metadata['dp_rank']}-tp{metadata['tp_rank']}-pid{metadata['pid']}"
        )
        self.root = Path(root) / metadata["role"] / safe_component(metadata["request_id"]) / worker
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "tensors").mkdir()
        self.metadata.update(
            schema_version=1,
            tool="pd_tensor_dump",
            complete=False,
            request_finished=False,
            errors=[],
            records=0,
            calls=[],
        )
        self.calls = {}
        self.flush()

    def flush(self):
        write_json(self.root / "manifest.json", self.metadata)

    def begin(self, info, expected):
        number = len(self.calls)
        call = dict(info, schema=1, call=number, complete=False, expected=list(expected))
        self.calls[number] = call
        self.metadata["calls"].append(number)
        self.metadata["complete"] = False
        write_json(self.root / "calls" / f"{number}.json", call)
        self.flush()
        return call

    def record(self, tensor, call, layer, kind, name, positions=None, mapping_only=False, tensor_layout="rank_local"):
        value = cpu_tensor(tensor)
        context = call["context_token_ids"]
        if positions is not None:
            if value.ndim == 0 or value.shape[0] != len(positions):
                raise ValueError(f"Invalid tensor row mapping: {kind}/{name}")
            if any(p < 0 or p >= len(context) for p in positions):
                raise ValueError("Tensor logical position is outside its recorded input context")
        relative = f"tensors/{self.metadata['records']:08d}.pt"
        temporary = (self.root / relative).with_suffix(".pt.tmp")
        torch.save(value, temporary)
        os.replace(temporary, self.root / relative)
        record = dict(
            schema=1,
            request_id=self.metadata["request_id"],
            model="main",
            call=call["call"],
            layer=layer,
            kind=kind,
            name=name,
            path=relative,
            shape=list(value.shape),
            dtype=str(value.dtype),
            positions=positions,
            token_ids=[context[p] for p in positions] if positions is not None else None,
            row_axis=0 if positions is not None else None,
            mapping_only=mapping_only,
            tensor_layout="mapping" if mapping_only else tensor_layout,
        )
        with (self.root / "index.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.metadata["records"] += 1
        return record

    def end(self, call, observed):
        missing = {(e["layer"], e["kind"], e["name"]) for e in call["expected"]} - observed
        if missing:
            self.fail(f"call {call['call']} missing tensors: {sorted(missing)[:8]}")
        if not call["context_complete"]:
            self.fail(f"call {call['call']} has unknown input context")
        call["complete"] = not missing and call["context_complete"]
        write_json(self.root / "calls" / f"{call['call']}.json", call)
        self.metadata["complete"] = not self.metadata["errors"] and all(c["complete"] for c in self.calls.values())
        self.flush()

    def fail(self, error):
        self.metadata["errors"].append(str(error))
        self.metadata["complete"] = False
        self.flush()

    def finish(self):
        self.metadata["request_finished"] = True
        self.flush()


class PDTensorDump:
    """Capture main-backbone calls, including target verification with MTP enabled.

    MTP draft internals are deliberately distinguished from the main backbone:
    they are not observed by these hooks or counted as covered by the report.
    """

    def __init__(self, runner, root, role, rank_info, get_context):
        self.runner, self.root, self.role = runner, Path(root), role
        self.rank_info, self.get_context = dict(rank_info), get_context
        self.external_ids = {}
        self.archives = {}
        self.active = None
        self.last = None
        self.sfa_stack = []
        self.sfa_caches = []
        self.handles = []
        self.patches = []
        self.layers = {}
        self.attentions = {}
        self.logical_topk = None
        self.scheduled = {}

    def observe_scheduler(self, output):
        for req in output.scheduled_new_reqs:
            self.external_ids[req.req_id] = getattr(req, "external_req_id", None)
        for internal in output.finished_req_ids:
            archive = self.archives.pop(internal, None)
            if archive is not None:
                archive.finish()
            self.external_ids.pop(internal, None)
        self.scheduled = dict(output.num_scheduled_tokens)

    def metadata(self, layer=None):
        context = self.get_context()
        values = context.attn_metadata
        if not isinstance(values, dict):
            return None
        for index, name in self.attentions.values():
            if layer is None or index == layer:
                return values.get(name)
        return None

    def patch(self, owner, name, factory):
        original = getattr(owner, name)
        self.patches.append((owner, name, original))
        setattr(owner, name, factory(original))

    def install(self, sfa_class, torch_npu):
        for name, module in self.runner.model.named_modules():
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
            if match and "DecoderLayer" in type(module).__name__:
                layer = int(match[1])
                if layer in self.layers:
                    raise ValueError("Ambiguous main decoder layer inventory")
                self.layers[layer] = module
            impl = getattr(module, "impl", None)
            if isinstance(impl, sfa_class):
                layer_name = module.layer_name
                match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
                if match:
                    self.attentions[id(impl)] = (int(match[1]), layer_name)
        if not self.layers or sorted(self.layers) != sorted(v[0] for v in self.attentions.values()):
            raise ValueError("PD tensor dump requires a complete main decoder/SFA layer inventory")
        config = self.runner.model_config.hf_text_config
        if sorted(self.layers) != list(range(int(config.num_hidden_layers))):
            raise ValueError("PD tensor dump layer inventory differs from model configuration")
        for layer, module in self.layers.items():
            self.handles.append(
                module.register_forward_pre_hook(functools.partial(self.decoder_pre, layer), with_kwargs=True)
            )
            self.handles.append(
                module.register_forward_hook(functools.partial(self.decoder_post, layer), with_kwargs=True)
            )
        self.patch(sfa_class, "forward", self.sfa_factory)
        self.patch(sfa_class, "_get_indexcache_topk_indices", self.shared_topk_factory)
        self.patch(torch.ops._C_ascend, "npu_sparse_flash_attention", self.kernel_factory("attention"))
        for owner, name, tuple_result in (
            (torch.ops._C_ascend, "npu_lightning_indexer", False),
            (torch.ops._C_ascend, "npu_lightning_indexer_quant", False),
            (torch_npu, "npu_lightning_indexer", True),
        ):
            if hasattr(owner, name):
                self.patch(owner, name, self.kernel_factory("indexer", tuple_result))
        self.patch(self.runner.model, "compute_logits", self.logits_factory)

    def restore(self):
        for handle in self.handles:
            handle.remove()
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)

    def expected(self):
        result = [
            dict(layer=-1, kind=kind, name=name)
            for kind, name in (
                ("model_input", "input_ids"),
                ("model_input", "positions"),
                ("model_output", "hidden_states"),
                ("logits", "output"),
            )
        ]
        for layer in self.layers:
            for kind, names in (
                ("decoder", ("input", "output")),
                ("sfa", ("input", "output")),
                ("attention", ("query_nope", "query_rope", "logical_topk", "output")),
                ("kv_current", ("nope", "rope")),
                ("kv_consumed", ("nope", "rope")),
            ):
                result.extend(dict(layer=layer, kind=kind, name=name) for name in names)
        return result

    def start(self, input_ids, positions):
        if self.last is not None:
            raise ValueError("Previous PD tensor call did not finish sampling")
        meta = self.metadata()
        if meta is None or not self.scheduled:
            return False  # Warmup/EP dummy forward: never assign it to a user request.
        req_ids = list(self.runner.input_batch.req_ids)
        actual = int(meta.num_actual_tokens)
        bounds = [int(x) for x in meta.query_start_loc_cpu]
        if len(bounds) < len(req_ids) + 1 or bounds[len(req_ids)] != actual:
            raise ValueError("PD dump cannot align scheduled requests with attention query rows")
        ids = cpu_tensor(input_ids).reshape(-1).tolist()[:actual]
        pos = cpu_tensor(positions).reshape(-1).tolist()[:actual]
        if len(ids) != actual or len(pos) != actual:
            raise ValueError("PD dump needs unsharded, one-dimensional main model token/position inputs")
        entries = []
        for row, internal in enumerate(req_ids):
            start, end = bounds[row : row + 2]
            if start == end:
                continue
            external = self.external_ids.get(internal)
            if external is None:
                raise ValueError("Missing external_req_id; update vLLM and vLLM-Ascend together before PD dump")
            request = self.runner.requests[internal]
            if getattr(request.sampling_params, "n", 1) != 1:
                raise ValueError("PD tensor dump currently requires sampling n=1")
            if request.prompt_token_ids is None:
                raise ValueError("PD tensor dump currently requires text token IDs, not prompt embeddings")
            context = list(request.prompt_token_ids) + list(request.output_token_ids)
            query_positions, query_ids = pos[start:end], ids[start:end]
            if any(p < 0 for p in query_positions) or len(set(query_positions)) != len(query_positions):
                raise ValueError("Ambiguous main model positions for request")
            needed = max(query_positions) + 1
            context.extend([None] * max(0, needed - len(context)))
            for position, token in zip(query_positions, query_ids):
                context[position] = token
            context = context[:needed]
            archive = self.archives.get(internal)
            if archive is None:
                archive = RequestArchive(
                    self.root,
                    dict(
                        self.rank_info,
                        request_id=external,
                        internal_request_id=internal,
                        role=self.role,
                        model_id=str(self.runner.model_config.model),
                        num_layers=len(self.layers),
                        layerwise_prefill=bool(getattr(self.runner, "layerwise_prefill_p_node", False)),
                        scope="eager main backbone and target verification; MTP draft internals excluded",
                        sampling_params=str(request.sampling_params),
                        prompt_token_ids=list(request.prompt_token_ids),
                    ),
                )
                self.archives[internal] = archive
            call = archive.begin(
                dict(
                    phase="prefill" if min(query_positions) < len(request.prompt_token_ids) else "decode",
                    positions=query_positions,
                    token_ids=query_ids,
                    context_token_ids=context,
                    context_complete=all(x is not None for x in context),
                ),
                self.expected(),
            )
            entries.append(dict(archive=archive, call=call, start=start, end=end, row=row, observed=set()))
        self.active = dict(entries=entries, positions=pos, token_ids=ids, actual=actual)
        self.last = self.active
        self.logical_topk = None
        self.emit(input_ids, -1, "model_input", "input_ids", meta, prefer_local=False)
        self.emit(positions, -1, "model_input", "positions", meta, prefer_local=False)
        return True

    @contextmanager
    def forward(self, input_ids, positions):
        try:
            self.start(input_ids, positions)
            yield
        except BaseException as error:
            if self.active:
                for entry in self.active["entries"]:
                    entry["archive"].fail(f"{type(error).__name__}: {error}")
            raise
        finally:
            self.active = None

    def record(self, entry, tensor, layer, kind, name, positions=None, mapping_only=False, layout="rank_local"):
        key = (layer, kind, name)
        if key in entry["observed"]:
            raise ValueError(f"Duplicate PD tensor observation {key}")
        entry["archive"].record(tensor, entry["call"], layer, kind, name, positions, mapping_only, layout)
        entry["observed"].add(key)

    def emit(self, tensor, layer, kind, name, meta, *, prefer_local=True, mapping_only=False):
        if self.active is None or tensor is None:
            return
        if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
            raise ValueError(f"Unsupported required tensor {kind}/{name}")
        cp = getattr(meta, "dsa_cp_context", None)
        lo, hi = row_window(tensor.shape[0], self.active["actual"], cp, prefer_local)
        for entry in self.active["entries"]:
            start, end = max(lo, entry["start"]), min(hi, entry["end"])
            start, end = max(lo, start), max(start, end)
            positions = self.active["positions"][start:end]
            self.record(
                entry,
                tensor[start - lo : end - lo],
                layer,
                kind,
                name,
                positions,
                mapping_only,
                "sequence_sharded" if cp is not None and prefer_local else "rank_local",
            )

    def decoder_pre(self, layer, module, args, kwargs):
        if self.active is None:
            return
        values = inspect.signature(module.forward).bind(*args, **kwargs).arguments
        for name, value in (("input", values.get("hidden_states")), ("input_residual", values.get("residual"))):
            if value is not None:
                self.emit(value, layer, "decoder", name, self.metadata(layer))

    def decoder_post(self, layer, module, args, kwargs, result):
        if self.active is None:
            return
        for index, value in enumerate(result if isinstance(result, (list, tuple)) else (result,)):
            if value is not None:
                self.emit(value, layer, "decoder", "output" if index == 0 else f"output_{index}", self.metadata(layer))

    def sfa_factory(self, original):
        @functools.wraps(original)
        def forward(impl, layer_name, hidden_states, kv_cache, attn_metadata, *args, **kwargs):
            if self.active is None:
                return original(impl, layer_name, hidden_states, kv_cache, attn_metadata, *args, **kwargs)
            layer, _ = self.attentions[id(impl)]
            self.sfa_stack.append((layer, attn_metadata))
            self.sfa_caches.append(kv_cache)
            try:
                self.emit(hidden_states, layer, "sfa", "input", attn_metadata)
                result = original(impl, layer_name, hidden_states, kv_cache, attn_metadata, *args, **kwargs)
                self.emit(result, layer, "sfa", "output", attn_metadata)
                return result
            finally:
                self.sfa_caches.pop()
                self.sfa_stack.pop()

        return forward

    def shared_topk_factory(self, original):
        @functools.wraps(original)
        def shared_topk(*args, **kwargs):
            result = original(*args, **kwargs)
            if self.active and self.sfa_stack:
                self.logical_topk = cpu_tensor(result)
            return result

        return shared_topk

    def current_kv(self, kv_cache, layer, meta):
        # Observe after CP all-gather/cache write, immediately before attention.
        # exec_kv alone runs too early and is bypassed by the MLAPO path.
        slots = cpu_tensor(meta.slot_mapping).long().reshape(-1)
        if len(slots) < self.active["actual"]:
            raise ValueError("Current KV slot mapping misses scheduled query rows")
        for entry in self.active["entries"]:
            start, end = entry["start"], entry["end"]
            selected = slots[start:end]
            positions = self.active["positions"][start:end]
            for cache, name in zip(kv_cache[:2], ("nope", "rope")):
                flat = cache.reshape(-1, *cache.shape[2:])
                if selected.numel() and (selected.min() < 0 or selected.max() >= flat.shape[0]):
                    raise ValueError("Current KV slot out of bounds")
                self.record(
                    entry,
                    flat.index_select(0, selected.to(cache.device)),
                    layer,
                    "kv_current",
                    name,
                    positions,
                    layout="replicated",
                )

    def indexer_kv(self, tensor, table, lengths, layer, name):
        table, lengths = cpu_tensor(table).long(), cpu_tensor(lengths).long().reshape(-1)
        block_size = int(tensor.shape[1])
        flat = tensor.reshape(-1, *tensor.shape[2:])
        for entry in self.active["entries"]:
            row = entry["row"]
            if row >= len(lengths):
                raise ValueError("Indexer request/table row missing")
            # P's indexer cache is replicated after CP all-gather/cache write.
            # Its kernel length only describes this TP rank's query window (it
            # can be zero), so archive the computed global prefix through this
            # chunk's end. D records the prefix its kernel actually consumes.
            prefix = len(entry["call"]["context_token_ids"]) if self.role == "P" else int(lengths[row])
            positions = torch.arange(prefix, dtype=torch.long)
            if positions.numel() and int(positions[-1]) // block_size >= table.shape[1]:
                raise ValueError("Indexer block table does not cover prefix")
            slots = table[row, positions // block_size] * block_size + positions % block_size
            if slots.numel() and (slots.min() < 0 or slots.max() >= flat.shape[0]):
                raise ValueError("Indexer KV physical slot out of bounds")
            self.record(
                entry,
                flat.index_select(0, slots.to(tensor.device)),
                layer,
                "kv_indexer",
                name,
                positions.tolist(),
                layout="replicated",
            )

    def attention_kv(self, values, layer, meta):
        selected = cpu_tensor(values["sparse_indices"]).long().reshape(values["query"].shape[0], -1)
        if self.logical_topk is None:
            raise ValueError("Attention has no observed logical indexer output")
        logical = self.logical_topk.long().reshape(self.logical_topk.shape[0], -1)
        table = cpu_tensor(values["block_table"]).long()
        lengths = cpu_tensor(values["actual_seq_lengths_kv"]).long().reshape(-1)
        cumulative = cpu_tensor(values["actual_seq_lengths_query"]).long().reshape(-1)
        cp = getattr(meta, "dsa_cp_context", None)
        lo, hi = row_window(selected.shape[0], self.active["actual"], cp)
        if logical.shape != selected.shape:
            raise ValueError("Attention selection and indexer output shape differ")
        for entry in self.active["entries"]:
            start, end = max(lo, entry["start"]), min(hi, entry["end"])
            end = max(start, end)
            a, b = start - lo, end - lo
            physical, original = selected[a:b], logical[a:b]
            owners = torch.bucketize(torch.arange(a, b), cumulative, right=True)
            positions = torch.tensor(self.active["positions"][start:end], dtype=torch.long)
            valid = (physical >= 0) & (original >= 0) & (physical < lengths[owners, None])
            valid &= original <= positions[:, None]
            block_size = int(values["key"].shape[1])
            blocks = physical.clamp_min(0) // block_size
            if valid.any() and int(blocks[valid].max()) >= table.shape[1]:
                raise ValueError("Attention selection exceeds block table")
            slots = table[owners[:, None], blocks.clamp_max(max(0, table.shape[1] - 1))]
            slots = slots * block_size + physical.clamp_min(0) % block_size
            pairs = torch.unique(torch.stack((original[valid], slots[valid]), dim=1), dim=0, sorted=True)
            for key, name in (("key", "nope"), ("key_rope", "rope")):
                cache = values[key]
                flat = cache.reshape(-1, *cache.shape[2:])
                if pairs.numel() and (pairs[:, 1].min() < 0 or pairs[:, 1].max() >= flat.shape[0]):
                    raise ValueError("Attention KV physical slot out of bounds")
                rows = cpu_tensor(flat.index_select(0, pairs[:, 1].to(cache.device)))
                keep = torch.ones(len(pairs), dtype=torch.bool)
                if len(pairs) > 1:
                    duplicate = pairs[1:, 0] == pairs[:-1, 0]
                    if duplicate.any() and not torch.equal(rows[1:][duplicate], rows[:-1][duplicate]):
                        raise ValueError("Logical KV has different contents in aliased physical slots")
                    keep[1:] = ~duplicate
                self.record(entry, rows[keep], layer, "kv_consumed", name, pairs[keep, 0].tolist(), layout="replicated")
        self.emit(self.logical_topk, layer, "attention", "logical_topk", meta)

    def kernel_factory(self, kind, tuple_result=False):
        def factory(original):
            @functools.wraps(original)
            def kernel(*args, **kwargs):
                if not self.active or not self.sfa_stack:
                    return original(*args, **kwargs)
                if args:
                    raise ValueError("PD dump sparse kernel requires the known keyword argument schema")
                layer, meta = self.sfa_stack[-1]
                if kind == "indexer":
                    for key in ("query", "weights", "query_dequant_scale"):
                        if key in kwargs:
                            self.emit(kwargs[key], layer, kind, key, meta)
                    self.indexer_kv(
                        kwargs["key"], kwargs["block_table"], kwargs["actual_seq_lengths_key"], layer, "key"
                    )
                    if "key_dequant_scale" in kwargs:
                        self.indexer_kv(
                            kwargs["key_dequant_scale"],
                            kwargs["block_table"],
                            kwargs["actual_seq_lengths_key"],
                            layer,
                            "scale",
                        )
                else:
                    self.current_kv(self.sfa_caches[-1], layer, meta)
                    self.emit(kwargs["query"], layer, kind, "query_nope", meta)
                    self.emit(kwargs["query_rope"], layer, kind, "query_rope", meta)
                    self.emit(kwargs["sparse_indices"], layer, "mapping", "sparse_indices", meta, mapping_only=True)
                    self.attention_kv(kwargs, layer, meta)
                result = original(*args, **kwargs)
                value = result[0] if tuple_result else result
                self.emit(value, layer, kind, "topk" if kind == "indexer" else "output", meta)
                if kind == "indexer":
                    self.logical_topk = cpu_tensor(value)
                return result

            return kernel

        return factory

    def logits_factory(self, original):
        @functools.wraps(original)
        def logits(*args, **kwargs):
            try:
                result = original(*args, **kwargs)
                if self.last is None:
                    return result
                if result is None:
                    raise ValueError("PD tensor dump requires observable main model logits on every TP rank")
                indices = cpu_tensor(self.runner.logits_indices).long().reshape(-1).tolist()
                for entry in self.last["entries"]:
                    rows = [i for i, index in enumerate(indices) if entry["start"] <= index < entry["end"]]
                    positions = [self.last["positions"][indices[i]] for i in rows]
                    selected = torch.tensor(rows, dtype=torch.long, device=result.device)
                    self.record(entry, result.index_select(0, selected), -1, "logits", "output", positions)
                    entry["archive"].flush()
                return result
            except BaseException as error:
                if self.last is not None:
                    for entry in self.last["entries"]:
                        entry["archive"].fail(f"logits {type(error).__name__}: {error}")
                raise

        return logits

    def sampled(self, request_ids, token_ids):
        if self.last is None:
            return
        try:
            # The runner supplies accepted CPU tokens AFTER discarding unfinished
            # prefill rows and rejection-sampler padding, never raw sampler output.
            if len(token_ids) != len(request_ids):
                raise ValueError("Sampled request rows differ from the captured batch")
            captured = {entry["archive"].metadata["internal_request_id"] for entry in self.last["entries"]}
            if len(set(request_ids)) != len(request_ids) or not captured.issubset(request_ids):
                raise ValueError("Sampled request identities differ from the captured batch")
            for internal, row in zip(request_ids, token_ids):
                if internal not in captured:
                    continue
                archive = self.archives.get(internal)
                if archive is None:
                    continue
                valid = [int(token) for token in row if 0 <= token < self.runner.input_batch.vocab_size]
                with (archive.root / "sampled.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"after_call": len(archive.calls) - 1, "token_ids": valid}) + "\n")
            for entry in self.last["entries"]:
                entry["archive"].end(entry["call"], entry["observed"])
        except BaseException as error:
            for entry in self.last["entries"]:
                entry["archive"].fail(f"sampling {type(error).__name__}: {error}")
            raise
        finally:
            self.last = None


def install_pd_tensor_dump(runner, root):
    # Imports stay here: a disabled diagnostic must not initialize LMCache,
    # torch_npu or any recording state as an import side effect.
    import torch_npu
    from lmcache.integration.vllm.utils import lmcache_get_or_create_config
    from vllm.distributed import get_dp_group, get_tp_group
    from vllm.forward_context import get_forward_context

    from vllm_ascend import envs
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

    if not Path(root).is_absolute():
        raise ValueError("VLLM_ASCEND_PD_TENSOR_DUMP_DIR must be an absolute run directory")
    if envs.VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD and envs.VLLM_ASCEND_DSA_OFFLOAD_FREE_PAGED:
        raise ValueError("PD tensor dump does not support the separate FREE_PAGED latent-pool path")
    if not runner.model_config.enforce_eager or int(runner.compilation_config.mode) != 0:
        raise ValueError(
            'PD tensor dump requires --enforce-eager --compilation-config \'{"mode":0,"cudagraph_mode":"NONE"}\''
        )
    if runner.use_async_scheduling:
        raise ValueError("PD tensor dump requires --no-async-scheduling for unambiguous sampled token records")
    parallel = runner.vllm_config.parallel_config
    if any(
        int(getattr(parallel, name, 1)) != 1
        for name in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
    ):
        raise ValueError("PD tensor dump supports TP/DP/EP with PP=PCP=DCP=1")
    role = {"sender": "P", "receiver": "D"}.get(str(lmcache_get_or_create_config().pd_role))
    if role is None:
        raise ValueError("PD tensor dump needs LMCache pd_role sender or receiver")
    tp, dp = get_tp_group(), get_dp_group()
    probe = PDTensorDump(
        runner,
        root,
        role,
        dict(
            host=socket.gethostname(),
            pid=os.getpid(),
            tp_rank=tp.rank_in_group,
            tp_size=tp.world_size,
            dp_rank=dp.rank_in_group,
            dp_size=dp.world_size,
        ),
        get_forward_context,
    )
    try:
        probe.install(AscendSFAImpl, torch_npu)
    except BaseException:
        probe.restore()
        raise
    return probe
