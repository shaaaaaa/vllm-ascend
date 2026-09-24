# SPDX-License-Identifier: Apache-2.0
"""Full-tensor observations for the explicitly selected file-PD diagnostic.

This module never imports the older local-only correctness worker. Tensor
readback deliberately changes timing: these observations are not a benchmark
or evidence that execution without observations is race-free.
"""

import functools
import inspect
import json
import os
import re
from pathlib import Path


def install_file_sdk():
    # Spawned workers import their extension before constructing connectors.
    from layerwise_prefill_file_store import install

    install()


def cpu_tensor(value):
    return value.detach().contiguous().clone().to("cpu")


def integer_list(value):
    import torch

    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise RuntimeError("Expected a one-dimensional token/position tensor")
    return cpu_tensor(value).to(torch.int64).tolist()


def row_window(rows, actual, cp=None, *, prefer_local=False):
    """Return the global query slice represented by valid rows of a tensor."""
    if cp is not None:
        start = int(cp.local_start)
        end = max(start, int(cp.local_end))
        capacity = int(cp.local_end_with_pad) - start
        if rows == capacity and (prefer_local or rows != actual):
            return start, end
    if rows >= actual:
        return 0, actual
    raise RuntimeError(f"Cannot map {rows} tensor rows to {actual} query tokens")


def query_context(meta, positions, token_ids, *, positions_local=False):
    """Normalize actual model inputs, never assign global tokens to TP padding."""
    actual = int(meta.num_actual_tokens)
    seq = [int(x) for x in meta.seq_lens_cpu]
    query = [int(x) for x in meta.query_start_loc_cpu]
    if len(seq) != 1 or len(query) != 2 or query[1] - query[0] != actual:
        raise RuntimeError("File correctness requires exactly one request")
    if actual <= 0:
        raise RuntimeError("Dummy/empty forward occurred after probe installation")
    cp = getattr(meta, "dsa_cp_context", None)
    if len(positions) >= actual and not positions_local:
        logical = positions[:actual]
    else:
        if cp is None:
            raise RuntimeError("Model positions do not cover its query")
        start, end = row_window(len(positions), actual, cp, prefer_local=True)
        logical = list(range(seq[0] - actual, seq[0]))
        if positions[: end - start] != logical[start:end]:
            raise RuntimeError("Sharded model positions disagree with logical query metadata")
    if len(token_ids) < actual:
        raise RuntimeError("Model input IDs do not expose all logical query tokens")
    if len(set(logical)) != len(logical) or any(p < 0 for p in logical):
        raise RuntimeError("Query contains duplicate or negative logical positions")
    return logical, token_ids[:actual]


def sparse_slots(indices, block_table, query_lengths, key_lengths, logical_indices, query_positions, block_size):
    """CPU mapping of actual sparse-FA selections to physical cache slots.

    The logical indexer output is retained before compact-scratch remapping.
    Kernel indices and block tables resolve the actual bytes consumed. Invalid
    and causally masked selections are retained in the mask, never read as KV.
    """
    import torch

    selected = indices.to(torch.int64).reshape(indices.shape[0], -1)
    logical = logical_indices.to(torch.int64).reshape(logical_indices.shape[0], -1)
    if selected.shape != logical.shape or selected.shape[0] != len(query_positions):
        raise RuntimeError("Sparse kernel and logical indexer rows cannot be aligned")
    cumulative = query_lengths.to(torch.int64).reshape(-1)
    lengths = key_lengths.to(torch.int64).reshape(-1)
    table = block_table.to(torch.int64)
    if table.ndim != 2 or cumulative.numel() != lengths.numel() or cumulative.numel() != table.shape[0]:
        raise RuntimeError("Sparse kernel request/table dimensions disagree")
    if not cumulative.numel() or int(cumulative[-1]) != selected.shape[0]:
        raise RuntimeError("Sparse kernel query lengths do not cover every query row")
    owners = torch.bucketize(torch.arange(selected.shape[0]), cumulative, right=True)
    valid = (selected >= 0) & (selected < lengths[owners, None]) & (logical >= 0)
    valid &= logical <= torch.tensor(query_positions, dtype=torch.int64)[:, None]
    block_indices = torch.div(selected.clamp_min(0), block_size, rounding_mode="floor")
    if valid.any() and int(block_indices[valid].max()) >= table.shape[1]:
        raise RuntimeError("Sparse selection exceeds the actual consumer block table")
    safe = block_indices.clamp(0, max(0, table.shape[1] - 1))
    slots = table[owners[:, None], safe] * block_size + selected.clamp_min(0) % block_size
    slots[~valid] = -1
    return logical, slots, valid


def unique_kv_rows(cache, logical, slots, valid):
    """Save each consumed logical token once, checking scratch aliases exactly."""
    import torch

    pairs = torch.stack((logical[valid], slots[valid]), dim=1)
    pairs = torch.unique(pairs, dim=0, sorted=True)
    flat = cache.reshape(-1, *cache.shape[2:])
    if pairs.numel() and (int(pairs[:, 1].min()) < 0 or int(pairs[:, 1].max()) >= flat.shape[0]):
        raise RuntimeError("Sparse selection references an invalid physical KV slot")
    values = cpu_tensor(flat.index_select(0, pairs[:, 1].to(cache.device)))
    keep = torch.ones(len(pairs), dtype=torch.bool)
    if len(pairs) > 1:
        duplicate = pairs[1:, 0] == pairs[:-1, 0]
        if duplicate.any():
            equal = (values[1:] == values[:-1]).reshape(len(pairs) - 1, -1).all(dim=1)
            if not bool(equal[duplicate].all()):
                raise RuntimeError("One logical KV position has unequal consumed physical copies")
            keep[1:] = ~duplicate
    return values[keep], pairs[keep, 0].tolist(), pairs


class FileTensorArchive:
    def __init__(self, stage_dir, rank):
        self.root = Path(stage_dir)
        self.rank = int(rank)
        self.directory = self.root / "tensors" / f"rank{self.rank}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index = (self.directory / "index.jsonl").open("x", encoding="utf-8")
        self.count = 0
        self.bytes = 0
        self.identities = set()

    def record(
        self,
        tensor,
        call,
        layer,
        kind,
        name,
        *,
        positions=None,
        token_ids=None,
        mapping_only=False,
        tensor_layout="rank_local",
        cross_rank=False,
    ):
        import torch

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Required tensor missing: {kind}/{name}")
        identity = (call["model"], call["call"], layer, kind, name)
        if identity in self.identities:
            raise RuntimeError(f"Duplicate tensor observation: {identity}")
        if positions is not None:
            if tensor.ndim == 0 or len(positions) > tensor.shape[0] or len(token_ids or []) != len(positions):
                raise RuntimeError("Tensor token mapping does not match its valid rows")
        value = cpu_tensor(tensor)
        relative = Path("tensors") / f"rank{self.rank}" / f"{self.count:08d}.pt"
        destination = self.root / relative
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(value, temporary)
        os.replace(temporary, destination)
        record = dict(
            schema=1,
            rank=self.rank,
            model=call["model"],
            call=call["call"],
            layer=layer,
            kind=kind,
            name=name,
            path=relative.as_posix(),
            shape=list(value.shape),
            dtype=str(value.dtype),
            positions=positions,
            token_ids=token_ids,
            row_axis=0 if positions is not None else None,
            valid_rows=len(positions) if positions is not None else None,
            phase=call["phase"],
            call_positions=call["positions"],
            call_token_ids=call["token_ids"],
            mapping_only=mapping_only,
            tensor_layout=tensor_layout,
            cross_rank=cross_rank,
        )
        self.index.write(json.dumps(record, allow_nan=False) + "\n")
        self.index.flush()
        self.count += 1
        self.bytes += destination.stat().st_size
        self.identities.add(identity)
        call["expected"].append(
            {
                key: record[key]
                for key in (
                    "layer",
                    "kind",
                    "name",
                    "row_axis",
                    "positions",
                    "token_ids",
                    "mapping_only",
                    "tensor_layout",
                    "cross_rank",
                )
            }
        )
        return record

    def close(self):
        self.index.close()


def model_inventory(model, sfa_class):
    layers, implementations = {}, {}
    for name, module in model.named_modules():
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
        if match and "DecoderLayer" in type(module).__name__:
            index = int(match[1])
            if index in layers:
                raise RuntimeError("Ambiguous decoder layer inventory")
            layers[index] = module
        impl = getattr(module, "impl", None)
        if isinstance(impl, sfa_class):
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", getattr(module, "layer_name", ""))
            if not match:
                raise RuntimeError("SFA has no stable logical layer identity")
            implementations[id(impl)] = (int(match[1]), module)
    if not layers or sorted(layers) != sorted(index for index, _ in implementations.values()):
        raise RuntimeError("Decoder/SFA inventories do not match")
    indices = sorted(
        index for index, module in implementations.values() if module.impl.has_indexer and not module.impl.skip_topk
    )
    roles = []
    for kind, names, role_layers in (
        ("decoder", ("input", "output", "positions"), sorted(layers)),
        ("sfa", ("input", "output"), sorted(layers)),
        ("attention", ("query_nope", "query_rope", "topk", "logical_topk", "output"), sorted(layers)),
        ("kv_consumed", ("nope", "rope"), sorted(layers)),
        ("kv_current", ("nope", "rope"), sorted(layers)),
        ("indexer", ("query", "weights", "topk"), indices),
        ("indexer_input", ("x", "q_c"), indices),
        ("kv_indexer", ("key",), indices),
    ):
        roles.extend(dict(kind=kind, name=name, layers=role_layers) for name in names)
    return (
        layers,
        implementations,
        dict(layers=sorted(layers), sfa_layers=sorted(layers), indexer_layers=indices, required_roles=roles),
    )


class FileProbe:
    def __init__(self, runner, archive, prompt_ids, get_context):
        self.runner = runner
        self.archive = archive
        self.prompt_ids = list(prompt_ids)
        self.history = dict(enumerate(prompt_ids))
        self.mtp_history = {}
        self.get_context = get_context
        self.models = {}
        self.inventories = {}
        self.calls = []
        self.active = []
        self.sfa_stack = []
        self.last = {}
        self.handles = []
        self.patches = []
        self.errors = []
        self.logical_topk = {}
        self.counts = {}
        self.mtp_sample_indices = None

    def local_model_positions(self, model_name):
        return model_name == "mtp" and bool(getattr(self.get_context(), "flash_comm_v1_enabled", False))

    def patch(self, owner, name, factory):
        original = getattr(owner, name)
        self.patches.append((owner, name, original))
        setattr(owner, name, factory(original))

    def metadata(self, model, layer=None):
        values = self.get_context().attn_metadata
        if not isinstance(values, dict):
            raise RuntimeError("No real per-layer metadata during file probe")
        _, impls, _ = self.inventories[model]
        attention = next(module for index, module in impls.values() if layer is None or index == layer)
        if attention.layer_name not in values:
            raise RuntimeError("Observed model layer has no attention metadata")
        return values[attention.layer_name]

    def positions_for(self, tensor, meta, call, *, prefer_local=True):
        if tensor.ndim == 0:
            return None, None
        start, end = row_window(
            tensor.shape[0], len(call["positions"]), getattr(meta, "dsa_cp_context", None), prefer_local=prefer_local
        )
        return call["positions"][start:end], call["token_ids"][start:end]

    def emit(
        self,
        tensor,
        layer,
        kind,
        name,
        *,
        meta=None,
        call=None,
        prefer_local=True,
        positions=None,
        mapping_only=False,
        raw=False,
    ):
        call = call or self.active[-1]
        if raw:
            ids = None
        elif positions is None:
            positions, ids = self.positions_for(tensor, meta, call, prefer_local=prefer_local)
        else:
            context = call["context_token_ids"]
            if any(p < 0 or p >= len(context) or context[p] is None for p in positions):
                raise RuntimeError("KV/logit position has no known input token context")
            ids = [context[p] for p in positions]
        layout, cross_rank = "rank_local", False
        if mapping_only:
            layout = "mapping"
        elif positions is not None:
            impls = self.inventories.get(call["model"], ({}, {}, {}))[1]
            cp_enabled = bool(impls) and all(
                bool(getattr(module.impl, "enable_dsa_cp", False)) for _, module in impls.values()
            )
            if cp_enabled and kind in ("kv_consumed", "kv_current", "kv_indexer"):
                # MLA latent and indexer keys are all-gathered before the real
                # consumer; these rows contain complete, replicated features.
                layout, cross_rank = "replicated", True
            elif (
                cp_enabled
                and meta is not None
                and getattr(meta, "dsa_cp_context", None) is not None
                and (
                    kind in ("decoder", "sfa", "attention", "indexer", "indexer_input", "model_input", "model_output")
                    or (kind == "mapping" and name == "attention_valid")
                )
            ):
                # DSA-CP fixes local_num_heads=num_heads*tp_size at model init.
                # Only the token axis is partitioned, including small decode
                # forwards whose valid row changes owner relative to prefill.
                layout, cross_rank = "sequence_sharded", True
        return self.archive.record(
            tensor,
            call,
            layer,
            kind,
            name,
            positions=positions,
            token_ids=ids,
            mapping_only=mapping_only,
            tensor_layout=layout,
            cross_rank=cross_rank,
        )

    def model_pre(self, model_name, module, args, kwargs):
        bound = inspect.signature(module.forward).bind(*args, **kwargs).arguments
        meta = self.metadata(model_name)
        local_positions = self.local_model_positions(model_name)
        positions, ids = query_context(
            meta, integer_list(bound["positions"]), integer_list(bound["input_ids"]), positions_local=local_positions
        )
        # MTP consumes the next token at each target-model position. Its
        # historical input context is shifted too, including a fresh D process
        # whose first MTP forward contains only the last prompt position.
        if model_name == "mtp":
            context = dict(self.mtp_history)
            context.update((position - 1, token) for position, token in self.history.items() if position > 0)
        else:
            context = dict(self.history)
        context.update(zip(positions, ids))
        if model_name == "main":
            self.history = context
        else:
            self.mtp_history = context
        end = max(positions) + 1
        full = [context.get(index) for index in range(end)]
        if any(value is None for value in full):
            raise RuntimeError("Cannot reconstruct the model's complete input token context")
        number = self.counts.get(model_name, 0)
        self.counts[model_name] = number + 1
        call = dict(
            model=model_name,
            call=number,
            phase="prefill" if min(positions) < len(self.prompt_ids) else "decode",
            positions=positions,
            token_ids=ids,
            context_token_ids=full,
            expected=[],
        )
        self.calls.append(call)
        self.active.append(call)
        self.last[model_name] = call
        if model_name == "mtp":
            if self.mtp_sample_indices is None:
                raise RuntimeError("MTP forward has no observed logits row selection")
            call["logits_indices"] = self.mtp_sample_indices
        self.logical_topk.pop(model_name, None)
        for name, tensor in bound.items():
            import torch

            if isinstance(tensor, torch.Tensor):
                prefer_local = local_positions if name == "positions" else name != "input_ids"
                self.emit(tensor, -1, "model_input", name, meta=meta, prefer_local=prefer_local)

    def model_post(self, model_name, module, args, kwargs, result):
        call = self.active[-1]
        if call["model"] != model_name:
            raise RuntimeError("Model probe nesting mismatch")
        meta = self.metadata(model_name)
        outputs = result if isinstance(result, (tuple, list)) else (result,)
        for index, tensor in enumerate(outputs):
            if tensor is not None:
                self.emit(
                    tensor,
                    -1,
                    "model_output",
                    str(index),
                    meta=meta,
                    prefer_local=self.local_model_positions(model_name),
                )
        self.active.pop()

    def decoder_pre(self, model_name, layer, module, args, kwargs):
        if not self.active or self.active[-1]["model"] != model_name:
            raise RuntimeError("Decoder ran without an observed model forward")
        bound = inspect.signature(module.forward).bind(*args, **kwargs).arguments
        meta = self.metadata(model_name, layer)
        for name, value in (
            ("input", bound.get("hidden_states")),
            ("positions", bound.get("positions")),
            ("input_residual", bound.get("residual")),
        ):
            if value is not None:
                prefer_local = self.local_model_positions(model_name) if name == "positions" else True
                self.emit(value, layer, "decoder", name, meta=meta, prefer_local=prefer_local)

    def decoder_post(self, model_name, layer, module, args, kwargs, result):
        meta = self.metadata(model_name, layer)
        outputs = result if isinstance(result, (tuple, list)) else (result,)
        for index, tensor in enumerate(outputs):
            if tensor is not None:
                self.emit(tensor, layer, "decoder", "output" if index == 0 else f"output_{index}", meta=meta)

    def logits_factory(self, model_name):
        def factory(original):
            @functools.wraps(original)
            def logits(*args, **kwargs):
                result = original(*args, **kwargs)
                if result is None:
                    raise RuntimeError("Logits are unavailable on this worker rank")
                call = self.last[model_name]
                if model_name == "main":
                    selected = integer_list(self.runner.logits_indices)
                    if len(selected) > result.shape[0] or any(i < 0 or i >= len(call["positions"]) for i in selected):
                        raise RuntimeError("Main logits indices do not map to observed model inputs")
                    positions = [call["positions"][i] for i in selected]
                else:
                    selected = call.get("logits_indices", [])
                    if (
                        not selected
                        or len(selected) > result.shape[0]
                        or any(i < 0 or i >= len(call["positions"]) for i in selected)
                    ):
                        raise RuntimeError("MTP logits indices do not map to observed model inputs")
                    positions = [call["positions"][i] for i in selected]
                self.emit(result, -1, "logits", "output", call=call, positions=positions)
                return result

            return logits

        return factory

    def dense_indexer_kv(self, tensor, table, lengths, layer, name):
        import torch

        lengths = cpu_tensor(lengths).to(torch.int64).reshape(-1)
        table = cpu_tensor(table).to(torch.int64)
        if table.shape[0] != lengths.numel():
            raise RuntimeError("Indexer KV table/length dimensions disagree")
        logical = torch.arange(int(lengths.max()), dtype=torch.int64).expand(table.shape[0], -1)
        valid = logical < lengths[:, None]
        blocks = logical // int(tensor.shape[1])
        if valid.any() and int(blocks[valid].max()) >= table.shape[1]:
            raise RuntimeError("Indexer block table does not cover the consumed prefix")
        slots = (
            torch.gather(table, 1, blocks.clamp_max(table.shape[1] - 1)) * tensor.shape[1] + logical % tensor.shape[1]
        )
        values, positions, pairs = unique_kv_rows(tensor, logical, slots, valid)
        self.emit(values, layer, "kv_indexer", name, positions=positions)
        self.emit(pairs, layer, "mapping", f"indexer_{name}_slots", raw=True, mapping_only=True)

    def kernel_factory(self, kind, tuple_output=False):
        def factory(original):
            @functools.wraps(original)
            def kernel(*args, **kwargs):
                if not self.sfa_stack:
                    if self.active:
                        raise RuntimeError("Sparse kernel ran outside an inventoried SFA")
                    return original(*args, **kwargs)
                if args:
                    raise RuntimeError("Sparse kernel positional schema is unsupported")
                layer, meta = self.sfa_stack[-1]
                if kind == "indexer":
                    for key, name in (
                        ("query", "query"),
                        ("weights", "weights"),
                        ("query_dequant_scale", "query_scale"),
                    ):
                        if key in kwargs:
                            self.emit(kwargs[key], layer, kind, name, meta=meta)
                    self.dense_indexer_kv(
                        kwargs["key"], kwargs["block_table"], kwargs["actual_seq_lengths_key"], layer, "key"
                    )
                    if "key_dequant_scale" in kwargs:
                        self.dense_indexer_kv(
                            kwargs["key_dequant_scale"],
                            kwargs["block_table"],
                            kwargs["actual_seq_lengths_key"],
                            layer,
                            "scale",
                        )
                else:
                    for key, name in (
                        ("query", "query_nope"),
                        ("query_rope", "query_rope"),
                        ("sparse_indices", "topk"),
                    ):
                        self.emit(kwargs[key], layer, kind, name, meta=meta, mapping_only=key == "sparse_indices")
                    model = self.active[-1]["model"]
                    logical_topk = self.logical_topk.get(model)
                    if logical_topk is None:
                        raise RuntimeError("Attention consumed KV without an observed logical indexer selection")
                    qpositions, _ = self.positions_for(kwargs["query"], meta, self.active[-1])
                    count = len(qpositions)
                    if count:
                        logical, slots, valid = sparse_slots(
                            cpu_tensor(kwargs["sparse_indices"])[:count],
                            cpu_tensor(kwargs["block_table"]),
                            cpu_tensor(kwargs["actual_seq_lengths_query"]).clamp_max(count),
                            cpu_tensor(kwargs["actual_seq_lengths_kv"]),
                            logical_topk[:count],
                            qpositions,
                            int(kwargs["key"].shape[1]),
                        )
                    else:
                        import torch

                        logical = logical_topk[:0].reshape(0, logical_topk.shape[-1]).to(torch.int64)
                        slots = torch.empty_like(logical)
                        valid = torch.empty_like(logical, dtype=torch.bool)
                    for key, name in (("key", "nope"), ("key_rope", "rope")):
                        values, positions, pairs = unique_kv_rows(kwargs[key], logical, slots, valid)
                        self.emit(values, layer, "kv_consumed", name, positions=positions)
                    self.emit(slots, layer, "mapping", "attention_slots", positions=qpositions, mapping_only=True)
                    self.emit(valid, layer, "mapping", "attention_valid", positions=qpositions)
                    self.emit(logical_topk, layer, "attention", "logical_topk", meta=meta)
                    self.emit(
                        kwargs["block_table"], layer, "mapping", "attention_block_table", raw=True, mapping_only=True
                    )
                result = original(*args, **kwargs)
                value = result[0] if tuple_output else result
                self.emit(value, layer, kind, "topk" if kind == "indexer" else "output", meta=meta)
                if kind == "indexer":
                    self.logical_topk[self.active[-1]["model"]] = cpu_tensor(value)
                return result

            return kernel

        return factory

    def install(self, sfa, torch_npu, engine):
        import torch

        if "mtp" in self.models:
            drafter = getattr(self.runner, "drafter", None)
            if not callable(getattr(drafter, "_run_mtp_draft_layer_with_diagnostics", None)):
                raise RuntimeError("MTP proposer does not expose its actual logits selection")

            def draft_factory(original):
                @functools.wraps(original)
                def draft(*args, **kwargs):
                    runtime = kwargs["runtime_inputs"]
                    self.mtp_sample_indices = integer_list(runtime["token_indices_to_sample"])[
                        : int(runtime["batch_size"])
                    ]
                    return original(*args, **kwargs)

                return draft

            self.patch(drafter, "_run_mtp_draft_layer_with_diagnostics", draft_factory)

        for model_name, model in self.models.items():
            self.inventories[model_name] = model_inventory(model, sfa.AscendSFAImpl)
            self.handles.append(
                model.register_forward_pre_hook(functools.partial(self.model_pre, model_name), with_kwargs=True)
            )
            self.handles.append(
                model.register_forward_hook(functools.partial(self.model_post, model_name), with_kwargs=True)
            )
            self.patch(model, "compute_logits", self.logits_factory(model_name))
            for layer, module in self.inventories[model_name][0].items():
                self.handles.append(
                    module.register_forward_pre_hook(
                        functools.partial(self.decoder_pre, model_name, layer), with_kwargs=True
                    )
                )
                self.handles.append(
                    module.register_forward_hook(
                        functools.partial(self.decoder_post, model_name, layer), with_kwargs=True
                    )
                )

        def forward_factory(original):
            @functools.wraps(original)
            def forward(impl, layer_name, hidden_states, kv_cache, attn_metadata, *args, **kwargs):
                if not self.active:
                    raise RuntimeError("SFA ran outside an observed main/MTP forward")
                entry = self.inventories[self.active[-1]["model"]][1].get(id(impl))
                if entry is None or attn_metadata is None:
                    raise RuntimeError("SFA is missing from the static main/MTP inventory")
                layer = entry[0]
                self.sfa_stack.append((layer, attn_metadata))
                try:
                    self.emit(hidden_states, layer, "sfa", "input", meta=attn_metadata)
                    result = original(impl, layer_name, hidden_states, kv_cache, attn_metadata, *args, **kwargs)
                    self.emit(result, layer, "sfa", "output", meta=attn_metadata)
                    return result
                finally:
                    self.sfa_stack.pop()

            return forward

        def index_input_factory(original):
            signature = inspect.signature(original)

            @functools.wraps(original)
            def index_input(*args, **kwargs):
                if self.sfa_stack:
                    layer, meta = self.sfa_stack[-1]
                    values = signature.bind(*args, **kwargs).arguments
                    for name in ("x", "q_c"):
                        self.emit(values[name], layer, "indexer_input", name, meta=meta)
                return original(*args, **kwargs)

            return index_input

        self.patch(sfa.AscendSFAImpl, "forward", forward_factory)
        self.patch(sfa.AscendSFAImpl, "indexer_select_post_process", index_input_factory)

        def current_kv_factory(original):
            signature = inspect.signature(original)

            @functools.wraps(original)
            def current_kv(*args, **kwargs):
                result = original(*args, **kwargs)
                if not self.sfa_stack:
                    raise RuntimeError("KV write ran outside an observed SFA")
                values = signature.bind(*args, **kwargs).arguments
                layer, meta = self.sfa_stack[-1]
                positions, _ = self.positions_for(values["kv_no_split"], meta, self.active[-1])
                slots = cpu_tensor(values["slots"]).to(torch.int64).reshape(-1)[: len(positions)]
                if slots.numel() != len(positions):
                    raise RuntimeError("Current KV slots do not cover all computed query rows")
                for plane, name in ((0, "nope"), (1, "rope")):
                    cache = values["kv_cache"][plane]
                    flat = cache.reshape(-1, *cache.shape[2:])
                    if slots.numel() and (int(slots.min()) < 0 or int(slots.max()) >= flat.shape[0]):
                        raise RuntimeError("Current KV write used an invalid live-token slot")
                    snapshot = flat.index_select(0, slots.to(cache.device))
                    self.emit(snapshot, layer, "kv_current", name, positions=positions)
                self.emit(slots, layer, "mapping", "current_slots", positions=positions, mapping_only=True)
                return result

            return current_kv

        self.patch(sfa.AscendSFAImpl, "exec_kv", current_kv_factory)
        self.patch(torch.ops._C_ascend, "npu_sparse_flash_attention", self.kernel_factory("attention"))
        for owner, name, tuple_output in (
            (torch.ops._C_ascend, "npu_lightning_indexer", False),
            (torch.ops._C_ascend, "npu_lightning_indexer_quant", False),
            (torch_npu, "npu_lightning_indexer", True),
        ):
            if hasattr(owner, name):
                self.patch(owner, name, self.kernel_factory("indexer", tuple_output))
        if callable(getattr(engine, "_run_store_pipeline", None)):

            def store_factory(original):
                @functools.wraps(original)
                def store(*args, **kwargs):
                    try:
                        return original(*args, **kwargs)
                    except Exception as error:
                        self.errors.append(f"Background store failed: {type(error).__name__}: {error}")
                        raise

                return store

            self.patch(engine, "_run_store_pipeline", store_factory)

    def restore(self):
        for handle in self.handles:
            handle.remove()
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.handles.clear()
        self.patches.clear()

    def finish(self):
        self.restore()
        if self.active or self.sfa_stack:
            self.errors.append("An observed model/SFA forward did not finish")
        for model_name, (_, _, inventory) in self.inventories.items():
            calls = [call for call in self.calls if call["model"] == model_name]
            if not calls and not (model_name == "mtp" and self.archive.root.name == "prefill"):
                self.errors.append(f"No {model_name} forwards were observed")
            if calls and not any(
                identity[0] == model_name and identity[2:] == (-1, "logits", "output")
                for identity in self.archive.identities
            ):
                self.errors.append(f"No {model_name} pre-sampling logits were observed")
            for call in calls:
                for role in inventory["required_roles"]:
                    for layer in role["layers"]:
                        identity = (model_name, call["call"], layer, role["kind"], role["name"])
                        if identity not in self.archive.identities:
                            self.errors.append(f"Missing required tensor: {identity}")
        self.archive.close()
        summary = dict(
            schema=1,
            rank=self.archive.rank,
            complete=not self.errors,
            errors=self.errors,
            records=self.archive.count,
            archived_bytes=self.archive.bytes,
            prompt_length=len(self.prompt_ids),
            models={name: entry[2] for name, entry in self.inventories.items()},
            calls=self.calls,
            scope="Eager main/MTP forwards, pre-sampling logits, actual sparse attention KV; no graph replay",
        )
        path = self.archive.directory / "coverage.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return summary


def flush_engine_stores(engine, errors=()):
    """Drain real request-store completion only after diagnostic generation."""
    engine.poll_layerwise_prefill_puts(final=True)
    condition = getattr(engine, "_store_cv", None)
    if condition is not None:
        with condition:
            if not condition.wait_for(
                lambda: not getattr(engine, "_pending_store_reqs", {}),
                timeout=float(engine.config.blocking_timeout_secs),
            ):
                raise TimeoutError("Timed out waiting for diagnostic async stores")
    states = getattr(engine, "_direct_store_states", {})
    if states:
        engine.wait_for_direct_stores(tuple(states))
    engine.poll_layerwise_prefill_puts(final=True)
    engine.wait_for_pending_sync_stores()
    if errors:
        raise RuntimeError("; ".join(errors))
    return True


class FileStoreWorker:
    def install_file_probe(self, stage_dir, prompt_ids):
        import torch_npu
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        from vllm.forward_context import get_forward_context

        from vllm_ascend.attention import sfa_v1

        if not self.vllm_config.model_config.enforce_eager:
            raise RuntimeError("File probes require eager execution in every stage")
        if hasattr(self, "_file_probe"):
            raise RuntimeError("File probe already installed")
        engine = get_kv_transfer_group()._lmcache_engine.lmcache_engine
        archive = FileTensorArchive(stage_dir, self.rank)
        probe = FileProbe(self.model_runner, archive, prompt_ids, get_forward_context)
        probe.models["main"] = self.model_runner.model
        if self.vllm_config.speculative_config is not None:
            drafter = getattr(self.model_runner, "drafter", None)
            model = getattr(drafter, "model", None)
            if model is None:
                archive.close()
                raise RuntimeError("MTP configured but its actual model is unavailable")
            if callable(getattr(model, "unwrap", None)):
                model = model.unwrap()
            probe.models["mtp"] = model
        try:
            probe.install(sfa_v1, torch_npu, engine)
        except BaseException:
            probe.restore()
            archive.close()
            raise
        self._file_probe = probe
        self._file_engine = engine
        return dict(rank=int(self.rank), models={name: entry[2] for name, entry in probe.inventories.items()})

    def finish_file_probe(self):
        summary = self._file_probe.finish()
        return {key: summary[key] for key in ("rank", "complete", "errors", "records")} | {
            "coverage_path": f"tensors/rank{self.rank}/coverage.json"
        }

    def flush_file_store(self):
        if hasattr(self, "_file_engine"):
            return flush_engine_stores(self._file_engine, self._file_probe.errors)
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        return flush_engine_stores(get_kv_transfer_group()._lmcache_engine.lmcache_engine)

    def close_file_store(self):
        from vllm.distributed.kv_transfer import ensure_kv_transfer_shutdown

        # Uses each engine's existing owner-specific close/unlink path. Never
        # enumerate or remove unrelated /dev/shm resources between stages.
        ensure_kv_transfer_shutdown()
        return True


# Explicit worker extension import is the selection mechanism, as in the
# reference file-store test. No production module imports this tools module.
install_file_sdk()
