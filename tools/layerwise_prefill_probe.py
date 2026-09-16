# SPDX-License-Identifier: Apache-2.0
"""Opt-in worker hooks. No hooks are installed by importing this module.

The diagnostic intentionally synchronizes/copies tensors; it is NOT a benchmark.
Hooks are installed after warmup through collective_rpc, with eager execution.
"""

import functools
import hashlib
import json
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
    def __init__(self, root, rank, prompt_len):
        self.root = Path(root) / "kv" / f"rank{rank}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.prompt_len = prompt_len
        self.seen = {}
        self.count = 0

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
        name = f"{self.count:07d}_{hashlib.sha256(layer.encode()).hexdigest()[:12]}_{part}.pt"
        torch.save({"positions": positions, "values": values}, self.root / name)
        with (self.root / "index.jsonl").open("a", encoding="utf-8") as out:
            out.write(
                json.dumps({"layer": layer, "part": part, "kind": kind, "path": name, "rows": positions.numel()}) + "\n"
            )
        self.count += 1


class PrefillValidationWorker:
    def install_prefill_validation(self, root, prompt_len):
        # Lazy: this RPC runs only after NPU/TP and the model are initialized.
        from vllm.forward_context import get_forward_context

        from vllm_ascend.attention import sfa_v1 as sfa

        if hasattr(self, "prefill_validation_recorder"):
            raise RuntimeError("Validation hooks already installed")
        recorder = KVRecorder(root, self.rank, prompt_len)
        self.prefill_validation_recorder = recorder
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
            if attn_metadata is None or not attn_metadata.req_ids:
                return original_forward(impl, layer_name, hidden_states, kv_cache, attn_metadata, *args, **kwargs)
            meta = attn_metadata
            if len(meta.req_ids) != 1:
                raise RuntimeError("Three-pass validation supports one request, not padded/mixed batches")
            qlen = int(meta.query_start_loc_cpu[1] - meta.query_start_loc_cpu[0])
            end = int(meta.seq_lens_cpu[0])
            if qlen != int(meta.num_actual_tokens) or not 0 < qlen <= end:
                raise RuntimeError("Invalid single-request query span")
            active.append((impl, layer_name, kv_cache, meta, end - qlen))
            try:
                result = original_forward(impl, layer_name, hidden_states, kv_cache, meta, *args, **kwargs)
                torch.npu.synchronize()
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
