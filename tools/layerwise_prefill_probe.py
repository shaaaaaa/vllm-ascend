# SPDX-License-Identifier: Apache-2.0
"""Opt-in worker hooks. No hooks are installed by importing this module.

The diagnostic intentionally synchronizes/copies tensors; it is NOT a benchmark.
Hooks are installed after warmup through collective_rpc. Ordinary PIECEWISE
graphs retain the eager mla_forward boundary; full/staged SFA graphs cannot
be used to observe every layer with these Python hooks.
"""

import functools
import hashlib
import json
import math
from pathlib import Path

import torch


def rows_from_slots(cache, slots):
    flat = cache.reshape(-1, *cache.shape[2:])
    slots = slots.reshape(-1).to(device=cache.device, dtype=torch.long)
    if slots.numel() and (int(slots.min()) < 0 or int(slots.max()) >= flat.shape[0]):
        raise RuntimeError("KV probe received invalid live slots (not padding)")
    return flat.index_select(0, slots).detach().cpu().clone()


def slots_for_positions(cache, block_table, positions):
    block_size = cache.shape[1]
    blocks = block_table[0].detach().cpu().long()
    return blocks[positions // block_size] * block_size + positions % block_size


class KVRecorder:
    def __init__(self, root, rank, prompt_len, flush_rows=256):
        self.root = Path(root) / "kv" / f"rank{rank}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.prompt_len = prompt_len
        self.seen = {}
        self.count = 0
        self.flush_rows = flush_rows
        self.pending = {}

    def save(self, layer, part, kind, positions, cache, slots):
        positions = positions.detach().cpu().long().reshape(-1)
        slots = slots.detach().cpu().reshape(-1)
        if positions.numel() != slots.numel():
            raise RuntimeError("KV positions/slots disagree")
        # Only store each historical row's first consumption: sparse top-k
        # may request the same row thousands of times. Current writes are
        # NEVER deduplicated: duplicate writes must be visible to the auditor.
        if kind == "loaded":
            seen = self.seen.setdefault((layer, part), set())
            keep = []
            for i, pos in enumerate(positions.tolist()):
                if 0 <= pos < self.prompt_len and pos not in seen:
                    keep.append(i)
                    seen.add(pos)
            positions, slots = positions[keep], slots[keep]
        if not positions.numel():
            return
        values = rows_from_slots(cache, slots)
        key = (layer, part, kind)
        pending = self.pending.setdefault(key, [])
        pending.append((positions.clone(), values))
        if sum(row.numel() for row, _ in pending) >= self.flush_rows:
            self._flush_key(key)

    def _flush_key(self, key):
        pending = self.pending.pop(key, [])
        if not pending:
            return
        layer, part, kind = key
        positions = torch.cat([row for row, _ in pending])
        values = torch.cat([value for _, value in pending])
        name = f"{self.count:07d}_{hashlib.sha256(layer.encode()).hexdigest()[:12]}_{part}.pt"
        torch.save({"positions": positions, "values": values}, self.root / name)
        with (self.root / "index.jsonl").open("a", encoding="utf-8") as out:
            out.write(
                json.dumps({"layer": layer, "part": part, "kind": kind, "path": name, "rows": positions.numel()}) + "\n"
            )
        self.count += 1

    def flush(self):
        for key in list(self.pending):
            self._flush_key(key)


LAYER_IO_SAMPLE_VALUES = 1024


def tensor_summary(tensor):
    """Return tolerant numerical evidence without retaining full activations.

    A raw hash is intentionally not used here.  BF16/NPU kernels may differ by
    a few ulps while remaining numerically equivalent, so the report keeps
    population statistics and a deterministic prefix sample for a later
    tolerance comparison.
    """
    value = tensor.detach().contiguous().to(device="cpu")
    flat = value.float().reshape(-1)
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    sample = flat[:LAYER_IO_SAMPLE_VALUES].tolist()
    sample = [float(item) if math.isfinite(float(item)) else None for item in sample]
    if finite_values.numel():
        mean = float(finite_values.mean())
        variance = float(finite_values.var(correction=0))
        abs_mean = float(finite_values.abs().mean())
        abs_max = float(finite_values.abs().max())
        minimum = float(finite_values.min())
        maximum = float(finite_values.max())
    else:
        mean = variance = abs_mean = abs_max = minimum = maximum = None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "mean": mean,
        "variance": variance,
        "abs_mean": abs_mean,
        "abs_max": abs_max,
        "min": minimum,
        "max": maximum,
        "nonfinite": int((~finite).sum()),
        "sample": sample,
    }


def require_tensor_summary(tensor, label):
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"Prefill layer I/O probe expected a tensor for {label}, "
            f"got {type(tensor).__name__}"
        )
    return tensor_summary(tensor)


class LayerIORecorder:
    """Persist one compact input/output summary per SFA layer and chunk."""

    def __init__(self, root, rank):
        self.path = Path(root) / "layer_io" / f"rank{rank}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pending = []

    def record(self, layer, chunk_index, token_start, token_end, inputs, outputs):
        self.pending.append(
            {
                "layer": layer,
                "chunk_index": int(chunk_index),
                "token_start": int(token_start),
                "token_end": int(token_end),
                "input": inputs if isinstance(inputs, dict) else tensor_summary(inputs),
                "output": outputs if isinstance(outputs, dict) else tensor_summary(outputs),
            }
        )

    def flush(self):
        if not self.pending:
            return
        self.pending.sort(key=lambda item: (item["chunk_index"], item["layer"]))
        with self.path.open("w", encoding="utf-8") as stream:
            for item in self.pending:
                stream.write(json.dumps(item, separators=(",", ":")) + "\n")
        self.pending.clear()


def validate_probe_graph_mode(config):
    if config.model_config.enforce_eager:
        return
    compilation = config.compilation_config
    mode = getattr(compilation.cudagraph_mode, "name", str(compilation.cudagraph_mode))
    if mode != "PIECEWISE" or "vllm::mla_forward" not in (compilation.splitting_ops or []):
        raise RuntimeError("KV probes require PIECEWISE with the mla_forward boundary, or P-node eager execution")


class PrefillValidationWorker:
    def finish_prefill_validation(self):
        restore_dma = getattr(self, "_prefill_dma_restore", None)
        if restore_dma is not None:
            restore_dma()
            self._prefill_dma_restore = None
        self.prefill_validation_recorder.flush()
        self.prefill_layer_io_recorder.flush()
        return {
            "rank": self.rank,
            "kv_files": self.prefill_validation_recorder.count,
            "layer_io_records": self.prefill_layer_io_recorder.path.as_posix(),
            "dma": self._prefill_dma_stats,
        }

    def install_prefill_validation(self, root, prompt_len, prefill_chunk_tokens=4096):
        # Lazy: this RPC runs only after NPU/TP and the model are initialized.
        from vllm.forward_context import get_forward_context

        from vllm_ascend.attention import sfa_v1 as sfa

        if hasattr(self, "prefill_validation_recorder"):
            raise RuntimeError("Validation hooks already installed")
        if getattr(self, "vllm_config", None) is not None:
            validate_probe_graph_mode(self.vllm_config)
        recorder = KVRecorder(root, self.rank, prompt_len)
        layer_io_recorder = LayerIORecorder(root, self.rank)
        self.prefill_validation_recorder = recorder
        self.prefill_layer_io_recorder = layer_io_recorder
        self._prefill_dma_stats = {
            "available": False,
            "calls": 0,
            "d2h_calls": 0,
            "h2d_calls": 0,
            "segments": 0,
            "bytes": 0,
        }
        self._prefill_dma_restore = None
        try:
            import lmcache_ascend.c_ops as lmc_ops

            original_dma = getattr(lmc_ops, "layerwise_prefill_dma_copy", None)
            if original_dma is not None:
                self._prefill_dma_stats["available"] = True

                @functools.wraps(original_dma)
                def traced_dma(copies, device_to_host):
                    stats = self._prefill_dma_stats
                    stats["calls"] += 1
                    stats["d2h_calls"] += int(bool(device_to_host))
                    stats["h2d_calls"] += int(not device_to_host)
                    stats["segments"] += len(copies)
                    stats["bytes"] += sum(int(copy[2]) for copy in copies)
                    return original_dma(copies, device_to_host)

                lmc_ops.layerwise_prefill_dma_copy = traced_dma
                self._prefill_dma_restore = lambda: setattr(
                    lmc_ops, "layerwise_prefill_dma_copy", original_dma
                )
        except (ImportError, OSError):
            # CPU unit tests do not have the Ascend extension.  The production
            # run still reports available=false instead of silently claiming
            # that a DMA call occurred.
            pass
        original_forward = sfa.AscendSFAImpl.forward
        original_wait = sfa.wait_for_kv_layer_from_connector
        active = []

        def parts(impl, layer, caches, meta):
            result = [
                ("nope", caches[0], meta.slot_mapping, meta.block_table),
                ("pe", caches[1], meta.slot_mapping, meta.block_table),
            ]
            if impl.has_indexer:
                index_caches = list(caches[2:])
                if not index_caches:
                    context = get_forward_context()
                    sibling = context.no_compile_layers[sfa._dsa_indexer_layer_name(layer)]
                    value = sibling.kv_cache[context.virtual_engine]
                    index_caches = list(value) if isinstance(value, (tuple, list)) else [value]
                if len(index_caches) not in (1, 2):
                    raise RuntimeError("Unexpected indexer cache layout")
                for name, cache in zip(("index", "scale"), index_caches):
                    result.append((name, cache, meta.indexer_slot_mapping, meta.indexer_block_table))
            return result

        @functools.wraps(original_forward)
        def forward(impl, layer_name, hidden_states, kv_cache, attn_metadata, *args, **kwargs):
            if attn_metadata is None:
                return original_forward(impl, layer_name, hidden_states, kv_cache, attn_metadata, *args, **kwargs)
            meta = attn_metadata
            # SHRINK_LATENT=0 (the P stage) does not populate req_ids. Hooks
            # are installed after warmup; use the real query/sequence spans
            # to validate this single-request run instead of skipping P.
            if (
                len(meta.seq_lens_cpu) != 1
                or len(meta.query_start_loc_cpu) != 2
                or (meta.req_ids is not None and len(meta.req_ids) != 1)
            ):
                raise RuntimeError("Three-pass validation supports one request, not padded/mixed batches")
            qlen = int(meta.query_start_loc_cpu[1] - meta.query_start_loc_cpu[0])
            end = int(meta.seq_lens_cpu[0])
            if qlen != int(meta.num_actual_tokens) or not 0 < qlen <= end:
                raise RuntimeError("Invalid single-request query span")
            active.append((impl, layer_name, kv_cache, meta, end - qlen))
            is_prefill = end <= prompt_len and qlen > 1
            io_input = require_tensor_summary(hidden_states, "SFA input") if is_prefill else None
            try:
                result = original_forward(impl, layer_name, hidden_states, kv_cache, meta, *args, **kwargs)
                torch.npu.synchronize()
                if is_prefill:
                    chunk_index = (end - 1) // prefill_chunk_tokens
                    output_tensor = kwargs.get("output")
                    if not isinstance(output_tensor, torch.Tensor):
                        output_tensor = result
                    layer_io_recorder.record(
                        layer_name,
                        chunk_index,
                        end - qlen,
                        end,
                        io_input,
                        require_tensor_summary(output_tensor, "SFA output"),
                    )
                positions = torch.arange(end - qlen, end)
                for part, cache, slots, _ in parts(impl, layer_name, kv_cache, meta):
                    recorder.save(layer_name, part, "current", positions, cache, slots[:qlen])
                return result
            finally:
                active.pop()

        @functools.wraps(original_wait)
        def wait(layer_name, *args, **kwargs):
            result = original_wait(layer_name, *args, **kwargs)
            if not active:
                return result
            impl, layer, caches, meta, prefix_len = active[-1]
            index = layer_name == sfa._dsa_indexer_layer_name(layer)
            if layer_name != layer and not index:
                return result
            selected = kwargs.get("selected_tokens")
            target = kwargs.get("target_slot_mapping")
            counts = kwargs.get("selected_token_counts")
            if selected is not None:
                if target is None or counts is None or selected.shape[0] != 1:
                    raise RuntimeError("Unsupported sparse-load diagnostic payload")
                torch.npu.synchronize()
                count = int(counts.reshape(-1)[0])
                positions = selected[0, :count].detach().cpu().long()
                sparse_slots = target[0, :count].detach().cpu()
            else:
                # Compact latent block tables do not address the whole prompt.
                # Their model-visible loads are captured above through the
                # actual selected-token/target-slot pairs, not guessed slots.
                if not index and impl.dsa_shrink_latent:
                    return result
                positions = torch.arange(min(prefix_len, prompt_len))
                sparse_slots = None
                torch.npu.synchronize()
            for part, cache, _, table in parts(impl, layer, caches, meta):
                if (part in ("index", "scale")) != index:
                    continue
                slots = sparse_slots if sparse_slots is not None else slots_for_positions(cache, table, positions)
                recorder.save(layer, part, "loaded", positions, cache, slots)
            return result

        sfa.AscendSFAImpl.forward = forward
        sfa.wait_for_kv_layer_from_connector = wait
        return {"rank": self.rank, "pid": __import__("os").getpid()}
