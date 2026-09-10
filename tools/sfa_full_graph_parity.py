#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run an isolated eight-layer, TP1/DP1/MTP1 numerical parity test on one NPU.

No HTTP server/client is needed. The eager process writes reference snapshots;
the second process compares each target forward as it runs. Temporary reference
files are managed internally; the only user-facing output is stdout/stderr.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

DEFAULT_MODEL = "/workspace/models/GLM-5.1-w4a8"
PROMPT_TOKENS = 4351  # Beyond the 4096-token MTP scratch prefix, next to a 256 boundary.
OUTPUT_TOKENS = 16
PREFILL_CHUNK = 512
FIXED_TOKEN = 100


def child_environment(mode: str, device: str) -> dict[str, str]:
    """Use local fresh CPU caches, never a running server's remote/shared store."""
    environment = {
        k: v for k, v in os.environ.items() if not k.startswith(("LMCACHE_", "VLLM_")) and k != "MOONCAKE_CONFIG_PATH"
    }
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "HCCL_DETERMINISTIC": "strict",
            "ASCEND_RT_VISIBLE_DEVICES": device,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "LMCACHE_CHUNK_SIZE": "256",
            "LMCACHE_LOCAL_CPU": "true",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "50",
            "LMCACHE_USE_LAYERWISE": "true",
            "LMCACHE_ENABLE_SPARSE_ATTENTION": "true",
            "LMCACHE_SAVE_DECODE_CACHE": "false",
            "LMCACHE_SAVE_UNFULL_CHUNK": "true",
            "LMCACHE_SAVE_FULL_CHUNK_IN_DECODE": "false",
            "LMCACHE_DSA_TWO_GROUPS": "true",
            "LMCACHE_ENABLE_SHARED_CPU_CACHE": "false",
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
        }
    )
    return environment


def run_child(args: argparse.Namespace) -> None:
    # Lazy imports: each process sees its final environment before loading any
    # vLLM/LMCache/plugin module or allocating a device context.
    from vllm import LLM, SamplingParams

    graph = args.child == "graph"
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        load_format="dummy",
        quantization="ascend",
        hf_overrides={"num_hidden_layers": 8},
        tensor_parallel_size=1,
        data_parallel_size=1,
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
        compilation_config={"mode": 3 if graph else 0, "cudagraph_mode": "PIECEWISE" if graph else "NONE"},
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


def run_pair(model: str = DEFAULT_MODEL, *, device: str = "0", atol: float = 1e-7, rtol: float = 1e-2) -> None:
    """Start two sequential fresh engines, on the same single card."""
    if not Path(model, "config.json").is_file():
        raise FileNotFoundError(f"Local model config not found: {model}/config.json")
    if any(not math.isfinite(value) or value < 0 for value in (atol, rtol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    if not device.isdecimal():
        raise ValueError("Select exactly one NPU device index")
    with TemporaryDirectory(prefix="sfa-parity-") as directory:
        for mode in ("eager", "graph"):
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
                    "--atol",
                    str(atol),
                    "--rtol",
                    str(rtol),
                ],
                env=child_environment(mode, device),
                check=True,
            )
        eager = json.loads(Path(directory, "eager-summary.json").read_text())
        graph = json.loads(Path(directory, "graph-summary.json").read_text())
        # Different planner layouts can legitimately load different numbers of
        # misses. Both workers separately require positive transfer coverage.
        coverage_fields = ("steps", "decode_steps", "q2_steps", "draft_calls")
        if len(eager) != 1 or len(graph) != 1 or any(eager[0][k] != graph[0][k] for k in coverage_fields):
            raise AssertionError(f"Eager/graph execution coverage differs: eager={eager}, graph={graph}")
    print("[SFA_PARITY] PASS: all 8 target layers, live Q2 replays, historical KV transfers; TP1/DP1/MTP1", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0")
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--child", choices=("eager", "graph"), help=argparse.SUPPRESS)
    parser.add_argument("--reference", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        if not args.reference:
            parser.error("Internal child requires a reference directory")
        run_child(args)
    else:
        run_pair(args.model, device=args.device, atol=args.atol, rtol=args.rtol)


if __name__ == "__main__":
    main()
