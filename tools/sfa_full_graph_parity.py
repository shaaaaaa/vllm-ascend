#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run an isolated eight-layer, TP8/DP1/MTP1 parity test on one eight-NPU host.

No HTTP server/client is needed. The eager process writes reference snapshots;
the second process compares each target forward as it runs. Temporary reference
files are managed internally; the only user-facing output is stdout/stderr.
"""

import argparse
import importlib
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

DEFAULT_MODEL = "/workspace/models/GLM-5.1-w4a8"
DEFAULT_DEVICES = "0,1,2,3,4,5,6,7"
PROMPT_TOKENS = 4351  # Beyond the 4096-token MTP scratch prefix, next to a 256 boundary.
OUTPUT_TOKENS = 16
PREFILL_CHUNK = 512
FIXED_TOKEN = 100


def preflight_dependencies() -> None:
    """Check the *imported* cross-repo API before either expensive engine starts.

    Run only in a disposable child with the graph environment. LMCache-Ascend
    patches imports, so this must not pollute the eager reference process.
    Signature binding does not instantiate connectors or allocate NPU tensors.
    Native checks verify exports, not hardware/kernel correctness.
    """
    modules = {}
    failures = []

    def load(name):
        if name not in modules:
            try:
                modules[name] = importlib.import_module(name)
            except Exception as exc:
                modules[name] = None
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        return modules[name]

    # Match the runtime's Ascend patch ordering before importing LMCache's
    # adapter; importing its CUDA implementation first is not equivalent.
    for name in ("vllm", "vllm_ascend", "lmcache_ascend", "lmcache"):
        load(name)

    # Arguments mirror the production call sites. Do not accept an old
    # singleton transfer by dropping request_capacity or bind_batch.
    contracts = (
        (
            "lmcache_ascend.v1.npu_connector.sparse_graph",
            "SparseGraphTransfer",
            (None, None, 256, 4399),
            {"request_capacity": 1},
        ),
        ("lmcache_ascend.v1.npu_connector.sparse_graph", "SparseGraphTransfer.bind_batch", (None, (), 0), {}),
        ("lmcache_ascend.v1.npu_connector.sparse_graph", "SparseGraphTransfer.load", (None, None, None, None), {}),
        (
            "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1",
            "LMCacheAscendConnectorV1Dynamic.prepare_sparse_graph_step",
            (None, ("layer",)),
            {"request_ids": ("request",), "frontiers": (4096,), "allow_empty": False},
        ),
        (
            "lmcache.integration.vllm.vllm_v1_adapter",
            "LMCacheConnectorV1Impl.prepare_sparse_graph_step",
            (None, ("layer",)),
            {"request_ids": ("request",), "frontiers": (4096,), "allow_empty": False},
        ),
        (
            "lmcache.v1.gpu_connector.sparse",
            "PreparedSparseSource",
            (),
            {"layers": (), "total_tokens": 0, "chunk_token_counts": (), "pointer_device": None},
        ),
        (
            "lmcache.v1.gpu_connector.sparse",
            "PreparedSparseSourceLayer",
            (),
            {"tensors": (), "chunk_ptrs_npu": None, "memory_objs": ()},
        ),
        (
            "lmcache_ascend.v1.npu_connector.utils",
            "prepare_sparse_direct_destination_state",
            (None, None, 6, 0, 0, 0),
            {},
        ),
        (
            "lmcache_ascend.v1.npu_connector.utils",
            "sparse_mla_dsa_batched_direct_kv_transfer_prepared",
            (None, None, None, None, 256, 4608, False, None),
            {},
        ),
    )
    for module_name, attribute, args, kwargs in contracts:
        module = load(module_name)
        if module is None:
            continue
        try:
            target = module
            for part in attribute.split("."):
                target = getattr(target, part)
            inspect.signature(target).bind(*args, **kwargs)
        except (AttributeError, TypeError, ValueError) as exc:
            failures.append(f"{module_name}.{attribute}: {exc}")

    native = load("lmcache_ascend.c_ops")
    if native is not None:
        for name in ("prepare_sparse_direct_destination_state", "sparse_mla_dsa_batched_direct_kv_transfer_prepared"):
            if not callable(getattr(native, name, None)):
                failures.append(f"lmcache_ascend.c_ops.{name}: native export missing; rebuild LMCache-Ascend")
    paths = "\n".join(f"  {name}: {getattr(module, '__file__', '<import failed>')}" for name, module in modules.items())
    if failures:
        raise RuntimeError(
            "[SFA_PARITY] dependency preflight failed BEFORE model loading:\n  "
            + "\n  ".join(failures)
            + f"\nPython: {sys.executable}\nImported modules:\n{paths}\n"
            "Update all four repos to feat/decode-full-graph and ensure this Python imports those checkouts "
            "(not stale site-packages). No inference or numerical comparison has run."
        )
    print(f"[SFA_PARITY] dependency interfaces OK; Python: {sys.executable}\n{paths}", flush=True)


def parse_devices(devices: str) -> tuple[int, ...]:
    """Require an explicit, unique single-host device list (TP1/2/4/8)."""
    parts = devices.split(",")
    if any(not part.isascii() or not part.isdecimal() for part in parts):
        raise ValueError("Devices must be comma-separated nonnegative integer indices")
    indices = tuple(int(part) for part in parts)
    if len(indices) not in (1, 2, 4, 8) or len(set(indices)) != len(indices):
        raise ValueError("Select 1, 2, 4 or 8 distinct NPU devices on one host")
    return indices


def child_environment(mode: str, devices: str) -> dict[str, str]:
    """Use local fresh CPU caches, never a running server's remote/shared store."""
    environment = {
        k: v for k, v in os.environ.items() if not k.startswith(("LMCACHE_", "VLLM_")) and k != "MOONCAKE_CONFIG_PATH"
    }
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "HCCL_DETERMINISTIC": "strict",
            "ASCEND_RT_VISIBLE_DEVICES": devices,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "LMCACHE_CHUNK_SIZE": "256",
            "LMCACHE_LOCAL_CPU": "true",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "2",
            "LMCACHE_USE_LAYERWISE": "true",
            "LMCACHE_ENABLE_SPARSE_ATTENTION": "true",
            "LMCACHE_SAVE_DECODE_CACHE": "false",
            "LMCACHE_SAVE_UNFULL_CHUNK": "true",
            "LMCACHE_SAVE_FULL_CHUNK_IN_DECODE": "false",
            "LMCACHE_DSA_TWO_GROUPS": "true",
            "LMCACHE_ENABLE_SHARED_CPU_CACHE": "false",
            "LMCACHE_EXTRA_CONFIG": '{"save_only_first_rank": false}',
            "LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE": "256",
            "VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE": "0",
            "VLLM_ASCEND_DSA_UNBUNDLE": "1",
            "VLLM_ASCEND_DSA_TWO_GROUPS": "1",
            "VLLM_ASCEND_DSA_SHRINK_LATENT": "2",
            "VLLM_ASCEND_SFA_STAGED_GRAPH": str(int(mode == "graph")),
            "VLLM_ASCEND_SFA_FULL_GRAPH": str(int(mode == "graph")),
            "VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES": "1",
            "VLLM_ASCEND_MTP_DRAFT_DEBUG": "0",
            "VLLM_ASCEND_MTP_DW_DIAG": "0",
            "VLLM_ASCEND_MTP_DW_DEEP_DIAG": "0",
            "VLLM_ASCEND_ENABLE_FLASHCOMM1": "0",
            "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": "0",
        }
    )
    return environment


def run_child(args: argparse.Namespace) -> None:
    # Lazy imports: each process sees its final environment before loading any
    # vLLM/LMCache/plugin module or allocating a device context.
    from vllm import LLM, SamplingParams

    graph = args.child == "graph"
    tp_size = len(parse_devices(args.devices))
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        load_format="dummy",
        quantization="ascend",
        hf_overrides={"num_hidden_layers": 8},
        tensor_parallel_size=tp_size,
        data_parallel_size=1,
        distributed_executor_backend="mp",
        enable_expert_parallel=False,
        max_model_len=PROMPT_TOKENS + OUTPUT_TOKENS + 32,
        max_num_seqs=1,
        max_num_batched_tokens=PREFILL_CHUNK,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=0.9,
        seed=0,
        enforce_eager=not graph,
        speculative_config={"num_speculative_tokens": 1, "method": "deepseek_mtp"},
        compilation_config={
            "mode": 3 if graph else 0,
            "cudagraph_mode": "PIECEWISE" if graph else "NONE",
            "pass_config": {"enable_sp": False},
        },
        worker_cls="vllm_ascend.worker.sfa_parity_worker.SFAParityWorker",
        kv_transfer_config={
            "kv_connector": "LMCacheAscendConnectorV1Dynamic",
            "kv_role": "kv_both",
            "kv_connector_module_path": "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1",
        },
        additional_config={
            "sfa_parity": {
                "mode": args.child,
                "reference": args.reference,
                "token_id": FIXED_TOKEN,
                "atol": args.atol,
                "rtol": args.rtol,
            }
        },
    )
    # Explicit token IDs remove tokenizer/chat-template ambiguity. Varied prompt
    # IDs avoid a degenerate repeated-token cache. Accepted and proposed tokens
    # are fixed independently, while both target and MTP computation still run.
    prompt = {"prompt_token_ids": [FIXED_TOKEN + i % 257 for i in range(PROMPT_TOKENS)]}
    outputs = llm.generate(
        prompt,
        SamplingParams(
            temperature=0,
            seed=0,
            max_tokens=OUTPUT_TOKENS,
            min_tokens=OUTPUT_TOKENS,
            ignore_eos=True,
            allowed_token_ids=[FIXED_TOKEN],
            detokenize=False,
        ),
        use_tqdm=False,
    )
    tokens = outputs[0].outputs[0].token_ids
    if list(tokens) != [FIXED_TOKEN] * OUTPUT_TOKENS:
        raise AssertionError(f"Teacher-forced target tokens were not honored: {tokens}")
    summary = llm.collective_rpc("parity_summary")
    Path(args.reference, f"{args.child}-summary.json").write_text(json.dumps(summary))
    print(f"[SFA_PARITY] {args.child}: {summary}", flush=True)


def validate_summaries(eager: list[dict], graph: list[dict], tp_size: int) -> None:
    """Require every TP rank, irrespective of RPC result ordering."""
    by_mode = []
    fields = ("steps", "decode_steps", "q2_steps", "draft_calls")
    for name, reports in (("eager", eager), ("graph", graph)):
        ranks = {report["rank"]: report for report in reports}
        if len(reports) != tp_size or set(ranks) != set(range(tp_size)):
            raise AssertionError(f"{name}: incomplete/duplicate TP rank coverage: {[r['rank'] for r in reports]}")
        for rank, report in ranks.items():
            if report["tp_size"] != tp_size or any(report[k] != ranks[0][k] for k in fields):
                raise AssertionError(f"{name}: rank={rank} has inconsistent TP/step coverage")
            if report["decode_steps"] < 2 or report["q2_steps"] < 2 or report["draft_calls"] < 2:
                raise AssertionError(f"{name}: rank={rank} lacks live decode/Q2/MTP coverage")
            if len(report["loaded_tokens_per_layer"]) != 8 or not all(x > 0 for x in report["loaded_tokens_per_layer"]):
                raise AssertionError(f"{name}: rank={rank} lacks historical KV coverage in all eight layers")
        by_mode.append(ranks)
    for rank in range(tp_size):
        if any(by_mode[0][rank][k] != by_mode[1][rank][k] for k in fields):
            raise AssertionError(f"rank={rank}: eager/graph execution coverage differs")


def run_pair(
    model: str = DEFAULT_MODEL, *, devices: str = DEFAULT_DEVICES, atol: float = 1e-7, rtol: float = 1e-2
) -> None:
    """Start two sequential fresh engines, sharing weights across selected NPUs."""
    if not Path(model, "config.json").is_file():
        raise FileNotFoundError(f"Local model config not found: {model}/config.json")
    if any(not math.isfinite(value) or value < 0 for value in (atol, rtol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    tp_size = len(parse_devices(devices))
    with TemporaryDirectory(prefix="sfa-parity-") as directory:
        for mode in ("preflight", "eager", "graph"):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--child",
                    mode,
                    "--reference",
                    directory,
                    "--model",
                    model,
                    "--devices",
                    devices,
                    "--atol",
                    str(atol),
                    "--rtol",
                    str(rtol),
                ],
                env=child_environment("graph" if mode == "preflight" else mode, devices),
                check=True,
            )
        eager = json.loads(Path(directory, "eager-summary.json").read_text())
        graph = json.loads(Path(directory, "graph-summary.json").read_text())
        # Different planner layouts can legitimately load different numbers of
        # misses. Both workers separately require positive transfer coverage.
        validate_summaries(eager, graph, tp_size)
    print(
        f"[SFA_PARITY] PASS: all {tp_size} ranks, 8 target layers, "
        f"live Q2 replays, historical KV; TP{tp_size}/DP1/MTP1",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--devices", default=DEFAULT_DEVICES)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--child", choices=("preflight", "eager", "graph"), help=argparse.SUPPRESS)
    parser.add_argument("--reference", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child == "preflight":
        preflight_dependencies()
    elif args.child:
        if not args.reference:
            parser.error("Internal child requires a reference directory")
        run_child(args)
    else:
        run_pair(args.model, devices=args.devices, atol=args.atol, rtol=args.rtol)


if __name__ == "__main__":
    main()
