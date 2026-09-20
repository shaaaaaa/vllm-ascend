#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile real Ascend SparseFlashAttention alongside pinned-memory DMA.

Standalone, single-NPU experiment. It does not load model weights or modify
LMCache/vLLM's production copy path. The SFA and transfer use independent NPU
buffers, as adjacent layer load/store operations would. Inspect the NPU
Memcpy and SparseFlashAttention tracks in each exported trace_view.json; wall
time alone cannot prove device-side overlap.
"""

import argparse
import json
import math
import statistics
import tempfile
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--query-tokens", type=int, default=512)
    parser.add_argument("--context-tokens", type=int, default=16384)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--copy-mib", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-repeats", type=int, default=2)
    parser.add_argument("--direction", choices=("both", "h2d", "d2h"), default="both")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def validate(args):
    if any(
        value <= 0
        for value in (
            args.query_tokens,
            args.context_tokens,
            args.topk,
            args.query_heads,
            args.copy_mib,
            args.repeats,
            args.profile_repeats,
        )
    ) or args.warmups < 0:
        raise ValueError("token/head/copy/repeat sizes must be positive; warmups may be zero")
    if args.context_tokens % 128 or args.topk > args.context_tokens:
        raise ValueError("context-tokens must be a multiple of 128 and >= topk")
    if args.query_heads not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError("query-heads must be an SFA-supported head count")


def make_sfa_inputs(torch, args, device):
    blocks = args.context_tokens // 128
    # Match the production SFA call: TND query, PA_BSND latent KV, one KV head.
    query = torch.randn((args.query_tokens, args.query_heads, 512), device=device, dtype=torch.bfloat16)
    query_rope = torch.randn((args.query_tokens, args.query_heads, 64), device=device, dtype=torch.bfloat16)
    key = torch.randn((blocks, 128, 1, 512), device=device, dtype=torch.bfloat16)
    key_rope = torch.randn((blocks, 128, 1, 64), device=device, dtype=torch.bfloat16)
    indices = torch.arange(args.topk, device=device, dtype=torch.int32)
    indices = indices.view(1, 1, -1).expand(args.query_tokens, 1, -1).contiguous()
    block_table = torch.arange(blocks, device=device, dtype=torch.int32).view(1, blocks)
    query_lens = torch.tensor([args.query_tokens], device=device, dtype=torch.int32)
    kv_lens = torch.tensor([args.context_tokens], device=device, dtype=torch.int32)
    return (query, key, indices, block_table, query_lens, kv_lens, query_rope, key_rope)


def sfa(torch, inputs):
    query, key, indices, block_table, query_lens, kv_lens, query_rope, key_rope = inputs
    return torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query,
        key=key,
        value=key,
        sparse_indices=indices,
        scale_value=1.0 / math.sqrt(576),
        sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=query_lens,
        actual_seq_lengths_kv=kv_lens,
        query_rope=query_rope,
        key_rope=key_rope,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
    )


def run_once(torch, mode, direction, inputs, host, npu_buffer, compute_stream, copy_stream):
    """One independent SFA/copy pair; no synchronization inside either operation."""
    torch.npu.synchronize()
    current = torch.npu.current_stream()
    start = torch.npu.Event(enable_timing=True)
    compute_done = torch.npu.Event()
    copy_done = torch.npu.Event()
    end = torch.npu.Event(enable_timing=True)
    start.record(current)
    output = None

    def enqueue_copy():
        # A pinned host tensor plus non_blocking=True is essential. The trace
        # must still show an NPU memcpy task before calling this a DMA result.
        if direction == "h2d":
            npu_buffer.copy_(host, non_blocking=True)
        else:
            host.copy_(npu_buffer, non_blocking=True)

    if mode in ("sfa", "serial", "parallel"):
        with torch.npu.stream(compute_stream):
            compute_stream.wait_event(start)
            with torch.autograd.profiler.record_function("SFA_COMPUTE"):
                output = sfa(torch, inputs)
            compute_done.record()
    if mode in ("dma", "serial", "parallel"):
        with torch.npu.stream(copy_stream):
            copy_stream.wait_event(start)
            if mode == "serial":
                copy_stream.wait_event(compute_done)
            with torch.autograd.profiler.record_function(f"PINNED_DMA_{direction.upper()}"):
                enqueue_copy()
            copy_done.record()
    if mode in ("sfa", "serial", "parallel"):
        current.wait_event(compute_done)
    if mode in ("dma", "serial", "parallel"):
        current.wait_event(copy_done)
    end.record(current)
    end.synchronize()
    # Keep the SFA output alive until both device streams complete.
    del output
    return start.elapsed_time(end)


def trace_candidates(path):
    """Give event names to inspect; never infer device overlap from CPU spans."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    events = raw.get("traceEvents", []) if isinstance(raw, dict) else raw
    matches = {"sfa": {}, "memcpy": {}}
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        lowered = name.lower()
        kind = (
            "sfa" if "sparseflashattention" in lowered or "sparse_flash_attention" in lowered
            else "memcpy" if "memcpy" in lowered or "mem_copy" in lowered
            else None
        )
        if kind:
            key = f"{name} [cat={event.get('cat', '')}, pid={event.get('pid', '')}]"
            matches[kind][key] = matches[kind].get(key, 0) + 1
    return matches


def main():
    args = arguments()
    validate(args)
    if args.output_dir is None:
        root = Path(tempfile.mkdtemp(prefix="dma-sfa-overlap-", dir=".")).resolve()
    else:
        root = args.output_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise ValueError(f"Output directory must be empty: {root}")
    print(f"[DMA_SFA] results: {root}", flush=True)

    import torch
    import torch_npu
    import vllm_ascend.vllm_ascend_C  # noqa: F401 - registers the actual SFA operator

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    inputs = make_sfa_inputs(torch, args, device)
    size = args.copy_mib * 1024 * 1024
    host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    if not host.is_pinned():
        raise RuntimeError("CPU buffer is not pinned; non_blocking copy would not test async DMA")
    host.fill_(7)
    npu_buffer = torch.empty(size, device=device, dtype=torch.uint8)
    npu_buffer.fill_(3)
    compute_stream = torch.npu.Stream(device=device)
    copy_stream = torch.npu.Stream(device=device)
    directions = ("h2d", "d2h") if args.direction == "both" else (args.direction,)
    summary = {"config": vars(args) | {"output_dir": str(root)}, "directions": {}}

    # Fail before profiling if the installed custom op or synthetic shape is
    # unsupported; never silently substitute a different attention kernel.
    try:
        run_once(torch, "sfa", "h2d", inputs, host, npu_buffer, compute_stream, copy_stream)
    except Exception as exc:
        raise RuntimeError("Real npu_sparse_flash_attention smoke test failed; no substitute kernel was used") from exc

    for direction in directions:
        timings = {}
        for mode in ("sfa", "dma", "serial", "parallel"):
            for _ in range(args.warmups):
                run_once(torch, mode, direction, inputs, host, npu_buffer, compute_stream, copy_stream)
            samples = [
                run_once(torch, mode, direction, inputs, host, npu_buffer, compute_stream, copy_stream)
                for _ in range(args.repeats)
            ]
            timings[mode] = {"mean_ms": statistics.mean(samples), "samples_ms": samples}
            print(f"[DMA_SFA] {direction}/{mode}: {timings[mode]['mean_ms']:.3f} ms", flush=True)

        traces = {}
        for mode in ("serial", "parallel"):
            profile_dir = root / "profile" / direction / mode
            profile_dir.mkdir(parents=True)
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(profile_dir), worker_name=f"{direction}_{mode}", analyse_flag=True
                ),
            ):
                for _ in range(args.profile_repeats):
                    run_once(torch, mode, direction, inputs, host, npu_buffer, compute_stream, copy_stream)
            paths = sorted(profile_dir.rglob("trace_view.json"))
            traces[mode] = [str(p.resolve()) for p in paths]
            print(f"[DMA_SFA] {direction}/{mode}: {len(paths)} trace_view.json -> {profile_dir}", flush=True)
            for path in paths:
                print(f"[DMA_SFA] trace: {path}", flush=True)
                print(f"[DMA_SFA] candidate NPU events: {trace_candidates(path)}", flush=True)

        serial_ms = timings["serial"]["mean_ms"]
        parallel_ms = timings["parallel"]["mean_ms"]
        summary["directions"][direction] = {
            "timings": timings,
            "serial_to_parallel_speedup": serial_ms / parallel_ms,
            "trace_view": traces,
        }
        print(
            f"[DMA_SFA] {direction}: serial/parallel={serial_ms / parallel_ms:.3f}; "
            "check the NPU SFA and Memcpy tracks in the parallel trace for actual overlap",
            flush=True,
        )
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DMA_SFA] done: {root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
