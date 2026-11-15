# SPDX-License-Identifier: Apache-2.0
"""Runner/SFA binding for the selected-prefill performance harness."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.prefill_selected_runtime import (
    SelectedKVRuntime,
)
from vllm_ascend.distributed.kv_transfer.sparse_offload.prefill_selected_workspace import (
    MIN_HISTORY_TOKENS,
    PAGE_ROWS,
)
from vllm_ascend.distributed.kv_transfer.sparse_offload.selected_prefill_test_source import (
    HostArenaSelectedStore,
    SelectedLoadStep,
    SelectedRequest,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _env_arm() -> str:
    value = os.getenv("VLLM_ASCEND_SELECTED_PREFILL_TEST_ARM", "selected").strip().lower()
    if value not in {"selected", "full", "none"}:
        raise ValueError("Selected test arm must be selected, full or none")
    return value


def _packed_group(kv_cache_config: "KVCacheConfig") -> tuple[int, list[str]]:
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        if getattr(group.kv_cache_spec, "cache_sparse_sfa_c8", False) is True:
            return gid, list(group.layer_names)
    raise ValueError("Selected test requires one packed main C8 group")


def _owners(runner: "NPUModelRunner", names: list[str]) -> dict[str, str]:
    context = runner.compilation_config.static_forward_context
    owners: dict[str, str] = {}
    previous: str | None = None
    for name in names:
        impl = context[name].impl
        if getattr(impl, "kv_lora_rank", None) != 512 or getattr(
            impl, "qk_rope_head_dim", None
        ) != 64:
            raise ValueError("Selected test requires kv_lora_rank=512 and RoPE=64")
        if not impl.skip_topk:
            if getattr(impl, "index_topk", None) != 2048:
                raise ValueError("Selected test requires 2048-wide TopK owners")
            previous = name
        if previous is None:
            raise ValueError("Selected test consumer has no earlier TopK owner")
        owners[name] = previous
    return owners


def initialize_selected_test(
    runner: "NPUModelRunner",
    kv_cache_config: "KVCacheConfig",
    kv_caches: dict[str, torch.Tensor | tuple[torch.Tensor, ...]],
) -> None:
    if os.getenv("VLLM_ASCEND_SELECTED_PREFILL_TEST", "0") != "1":
        return
    if runner.selected_kv_runtime is not None:
        raise RuntimeError("Selected test runtime must be initialized once")
    packed_gid, names = _packed_group(kv_cache_config)
    main_caches = {
        name: (kv_caches[name][0] if isinstance(kv_caches[name], tuple) else kv_caches[name])
        for name in names
    }
    max_pages = int(
        os.getenv("VLLM_ASCEND_SELECTED_PREFILL_TEST_MAX_PAGES", "1024")
    )
    store = HostArenaSelectedStore(
        runner.device, main_caches, max_pages=max_pages
    )
    runtime = SelectedKVRuntime(
        runner,
        owner_by_layer=_owners(runner, names),
        main_caches=main_caches,
        source=None,  # The test binding constructs load steps directly.
        query_capacity=8192,
    )
    runner.selected_kv_runtime = runtime
    runner._selected_test_store = store
    runner._selected_test_packed_gid = packed_gid
    runner._selected_test_names = names
    runner._selected_test_captured = False
    runner._selected_test_arm = _env_arm()
    runner._selected_test_history = int(
        os.getenv("VLLM_ASCEND_SELECTED_PREFILL_TEST_HISTORY", "0")
    )
    runner._selected_test_query = int(
        os.getenv("VLLM_ASCEND_SELECTED_PREFILL_TEST_QUERY", "100")
    )


def _history_pages(history: int) -> int:
    if history <= 0 or history % PAGE_ROWS:
        raise ValueError("Selected test history must be a positive multiple of 128")
    return history // PAGE_ROWS


def prepare_selected_test(
    runner: "NPUModelRunner", attn_metadata
) -> SelectedLoadStep | None:
    runtime = runner.selected_kv_runtime
    if runtime is None or runner._selected_test_arm == "none":
        return None
    if runner.input_batch.num_reqs != 1:
        return None
    history = int(runner.input_batch.num_computed_tokens_cpu[0])
    if history < MIN_HISTORY_TOKENS:
        return None
    starts = runner.query_start_loc.np[:2]
    query = int(starts[1] - starts[0])
    if query <= 0:
        return None
    pages = _history_pages(history)
    table = runner.input_batch.block_table[runner._selected_test_packed_gid]
    block_ids = table.block_table.np[0, :pages]
    store: HostArenaSelectedStore = runner._selected_test_store
    if runner._selected_test_arm == "full":
        for name in runner._selected_test_names:
            store.full_load(name, runtime.main_caches[name], block_ids, pages)
        return None
    ready = torch.npu.Event()
    ready.record(torch.npu.current_stream())
    bases = {
        name: store.page_bases(
            name, block_ids, pages, runtime.main_caches[name].shape[0]
        )
        for name in runner._selected_test_names
    }
    request = SelectedRequest(
        str(runner.input_batch.req_ids[0]), history, query, True
    )
    return SelectedLoadStep(request, bases, ready)


def bind_selected_test(runner: "NPUModelRunner", attn_metadata) -> None:
    runtime = runner.selected_kv_runtime
    if runtime is None:
        return
    history = int(runner.input_batch.num_computed_tokens_cpu[0])
    starts = runner.query_start_loc.np[:2]
    query = int(starts[1] - starts[0])
    if history < MIN_HISTORY_TOKENS or query <= 0:
        return
    runner._selected_test_metric = {
        "arm": runner._selected_test_arm,
        "history": history,
        "query": query,
    }
    runner._selected_test_wall_start = time.perf_counter()
    step = prepare_selected_test(runner, attn_metadata)
    if step is None:
        return
    request = step.requests[0]
    runtime.bind_step(
        step,
        [request.request_id],
        [request.history_tokens],
        [0, request.query_tokens],
        attn_metadata,
    )


def finish_selected_test(runner: "NPUModelRunner") -> None:
    runtime = runner.selected_kv_runtime
    if runtime is not None and runtime.batch is not None:
        runtime.finish_step()
    start = getattr(runner, "_selected_test_wall_start", None)
    metric = getattr(runner, "_selected_test_metric", None)
    if start is None or metric is None:
        return
    metric["forward_ms"] = (time.perf_counter() - start) * 1000.0
    pages = _history_pages(metric["history"])
    metric["full_bytes"] = pages * 128 * 656 * len(runner._selected_test_names)
    status = None
    if runtime is not None and runtime.current_plan is not None:
        status = runtime.current_plan[1]
    if status is not None:
        metric["selected_rows"] = int(status[0, 0])
        metric["selected_bytes"] = (
            metric["selected_rows"] * 656 * len(runner._selected_test_names)
        )
    else:
        metric["selected_rows"] = 0
        metric["selected_bytes"] = 0
    print("SELECTED_METRIC " + repr(metric), flush=True)
    runner._selected_test_wall_start = None
    runner._selected_test_metric = None


def capture_selected_history(runner: "NPUModelRunner", attn_metadata) -> None:
    if runner.selected_kv_runtime is None or runner._selected_test_arm == "none":
        return
    if not attn_metadata:
        return
    metadata = next(iter(attn_metadata.values()))
    seq_len = int(metadata.seq_lens_cpu[0])
    history = runner._selected_test_history
    if history <= 0:
        return
    computed = int(runner.input_batch.num_computed_tokens_cpu[0])
    actual = int(metadata.num_actual_tokens)
    # A single chunked request: the first chunk is exactly the history H.
    if computed != 0 or actual != history or seq_len != history + runner._selected_test_query:
        return
    pages = _history_pages(history)
    table = runner.input_batch.block_table[runner._selected_test_packed_gid]
    block_ids = table.block_table.np[0, :pages]
    store: HostArenaSelectedStore = runner._selected_test_store
    for name in runner._selected_test_names:
        cache = runner.selected_kv_runtime.main_caches[name]
        store.save_pages(name, cache, block_ids, pages)
        ids = torch.tensor(block_ids, dtype=torch.long, device=runner.device)
        cache.index_fill_(0, ids, 0)
    torch.npu.synchronize()
