# SPDX-License-Identifier: Apache-2.0
"""Start one real vLLM request through the layerwise-prefill path.

This is an NPU integration/unit smoke test rather than a CPU test.  Run it on
the Ascend host from the vllm-ascend checkout, for example::

    python -u tools/layerwise_prefill_model_test.py \
        --mode p --model /workspace/models/GLM-5.2-w4a8c8-0723 \
        --prompt-file examples/layerwise_prefill/article_summary_80k.txt \
        --prompt-tokens 80000

Use ``--mode d`` to verify the original D-node transfer path.  The script
starts one fresh model process per invocation, so it does not need stage
diagnostics or a second path-specific DMA environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

COMPUTE_CHUNK_TOKENS = 4_096
SHORT_MAX_MODEL_LEN = 16_384
MIN_PROMPT_FRACTION = 0.95
MAX_PROMPT_FIT_ATTEMPTS = 3
DEFAULT_MODEL = "/workspace/models/GLM-5.2-w4a8c8-0723"
DEFAULT_DEVICES = "0,1,2,3,4,5,6,7"
DEFAULT_PROMPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "layerwise_prefill"
    / "article_summary_80k.txt"
)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mode", choices=("p", "d"), default="p")
    cli.add_argument("--model", default=DEFAULT_MODEL)
    cli.add_argument("--devices", default=DEFAULT_DEVICES)
    cli.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    cli.add_argument(
        "--prompt-tokens",
        type=int,
        default=10_000,
        help="Target prompt length; use 80000 for the full long profile",
    )
    cli.add_argument("--cpu-cache-gb", type=float, default=24.0)
    cli.add_argument("--max-num-batched-tokens", type=int, default=COMPUTE_CHUNK_TOKENS)
    cli.add_argument("--output-json", type=Path)
    return cli


def _normalize_prompt_token_ids(encoded: Any) -> list[int]:
    """Extract one-dimensional token IDs from tokenizer/template outputs."""

    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    while (
        isinstance(encoded, (tuple, list))
        and len(encoded) == 1
        and isinstance(encoded[0], (tuple, list))
    ):
        encoded = encoded[0]
    if not isinstance(encoded, (tuple, list)):
        raise TypeError(f"unsupported chat-template output: {type(encoded)!r}")
    return [int(token) for token in encoded]


def build_prompt(tokenizer: Any, article: str, target_tokens: int) -> tuple[str, list[int]]:
    """Fit a fixed article to a chat template without an unbounded retry loop."""

    if target_tokens <= 0:
        raise ValueError("prompt-tokens must be positive")

    def encode(text: str) -> list[int]:
        return _normalize_prompt_token_ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        )

    body_ids = tokenizer.encode(article, add_special_tokens=False)
    body_budget = target_tokens - len(encode(""))
    for _ in range(MAX_PROMPT_FIT_ATTEMPTS):
        if body_budget <= 0:
            break
        text = (
            article
            if len(body_ids) <= body_budget
            else tokenizer.decode(
                body_ids[:body_budget],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        token_ids = encode(text)
        if len(token_ids) <= target_tokens:
            if len(token_ids) < target_tokens * MIN_PROMPT_FRACTION and len(token_ids) > 4_096:
                raise ValueError(
                    f"prompt tokenized to only {len(token_ids)} tokens for target={target_tokens}; "
                    "check the prompt file and tokenizer"
                )
            return text, token_ids
        body_budget -= len(token_ids) - target_tokens
    raise ValueError(
        f"could not fit prompt file within {target_tokens} tokens after "
        f"{MAX_PROMPT_FIT_ATTEMPTS} bounded attempts"
    )


def validate_args(args: argparse.Namespace) -> list[str]:
    devices = args.devices.split(",")
    if not devices or any(not device.isdigit() for device in devices):
        raise ValueError(f"devices must be a comma-separated list of IDs: {args.devices!r}")
    if len(devices) != len(set(devices)):
        raise ValueError(f"devices must be distinct: {args.devices!r}")
    if args.prompt_tokens <= 0:
        raise ValueError("prompt-tokens must be positive")
    if args.cpu_cache_gb <= 0:
        raise ValueError("cpu-cache-gb must be positive")
    if args.max_num_batched_tokens <= 0:
        raise ValueError("max-num-batched-tokens must be positive")
    if not args.prompt_file.is_file():
        raise FileNotFoundError(
            f"prompt file does not exist: {args.prompt_file}; pass --prompt-file on the Ascend host"
        )
    return devices


def case_environment(args: argparse.Namespace) -> dict[str, str]:
    """Build the P/D environment; P marker is the only path switch."""

    p_node = args.mode == "p"
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("VLLM_", "LMCACHE_", "MOONCAKE_"))
    }
    env.update(
        {
            "ASCEND_RT_VISIBLE_DEVICES": args.devices,
            "PYTHONHASHSEED": "0",
            "HCCL_DETERMINISTIC": "strict",
            "HCCL_BUFFSIZE": "200",
            "MSMONITOR_USE_DAEMON": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": str(p_node).lower(),
            "VLLM_ASCEND_DSA_UNBUNDLE": "1",
            "VLLM_ASCEND_DSA_TWO_GROUPS": "1",
            "VLLM_ASCEND_DSA_SHARED_POOL": "1",
            "VLLM_ASCEND_DSA_SHRINK_LATENT": "0",
            "VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE": "0",
            "VLLM_ASCEND_ENABLE_FLASHCOMM1": "0",
            "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "0",
            "LMCACHE_CHUNK_SIZE": "1024",
            "LMCACHE_LOCAL_CPU": "true",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": str(args.cpu_cache_gb),
            "LMCACHE_USE_LAYERWISE": "true",
            "LMCACHE_ENABLE_SPARSE_ATTENTION": "true",
            "LMCACHE_DSA_TWO_GROUPS": "true",
            "LMCACHE_STORE_ASYNC": "true",
            "LMCACHE_STORE_ASYNC_MAX_QUEUE_SIZE": "2",
            "LMCACHE_SAVE_DECODE_CACHE": "false",
            "LMCACHE_SAVE_UNFULL_CHUNK": "true",
            "LMCACHE_SAVE_FULL_CHUNK_IN_DECODE": "false",
            "LMCACHE_ENABLE_SHARED_CPU_CACHE": "true",
            "LMCACHE_SHARED_CPU_CACHE_STRICT": "true",
            "LMCACHE_SHARED_CPU_CACHE_PASSIVE_WRITABLE": "true",
            "LMCACHE_EXTRA_CONFIG": json.dumps({"save_only_first_rank": True}),
        }
    )
    # Deliberately do not set LMCACHE_LAYERWISE_PREFILL_DMA.  The adapter
    # derives raw-DMA mode from VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE; D nodes
    # must retain their original single-layer transfer path.
    return env


def install_case_environment(args: argparse.Namespace) -> dict[str, str]:
    """Replace inherited deployment knobs before importing vLLM/LMCache."""

    for key in list(os.environ):
        if key.startswith(("VLLM_", "LMCACHE_", "MOONCAKE_")):
            del os.environ[key]
    env = case_environment(args)
    os.environ.update(env)
    return env


def engine_options(args: argparse.Namespace, prompt_tokens: int) -> dict[str, Any]:
    max_model_len = max(
        SHORT_MAX_MODEL_LEN,
        prompt_tokens + args.max_num_batched_tokens,
    )
    return {
        "model": args.model,
        "trust_remote_code": True,
        "load_format": "safetensors",
        "quantization": "ascend",
        "tensor_parallel_size": len(args.devices.split(",")),
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "distributed_executor_backend": "mp",
        "enable_expert_parallel": True,
        "gpu_memory_utilization": 0.97,
        "max_model_len": max_model_len,
        "max_num_seqs": 1,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "enforce_eager": True,
        "seed": 1024,
        "speculative_config": {
            "method": "deepseek_mtp",
            "num_speculative_tokens": 1,
        },
        "kv_transfer_config": {
            "kv_connector": "LMCacheAscendConnectorV1Dynamic",
            "kv_role": "kv_producer",
            "kv_connector_module_path": "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1",
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.name != "posix":
        raise RuntimeError("run this model test on the Linux Ascend host")
    devices = validate_args(args)
    install_case_environment(args)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    article = args.prompt_file.read_text(encoding="utf-8")
    _, token_ids = build_prompt(tokenizer, article, args.prompt_tokens)
    options = engine_options(args, len(token_ids))
    print(
        f"[LAYERWISE_TEST] mode={args.mode} devices={devices} prompt_tokens={len(token_ids)} "
        f"max_model_len={options['max_model_len']} startup_begin",
        flush=True,
    )
    llm = LLM(**options)
    try:
        result = llm.generate(
            {"prompt_token_ids": token_ids},
            SamplingParams(temperature=0, seed=1024, max_tokens=1),
            use_tqdm=False,
        )[0]
        if not result.outputs or not result.outputs[0].token_ids:
            raise RuntimeError("the model returned no output token")
        report = {
            "mode": args.mode,
            "p_node_env": os.environ["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"],
            "prompt_tokens": len(token_ids),
            "num_cached_tokens": int(result.num_cached_tokens),
            "output_token_ids": list(result.outputs[0].token_ids),
            "elapsed_seconds": time.perf_counter() - started,
            "max_model_len": options["max_model_len"],
            "max_num_batched_tokens": options["max_num_batched_tokens"],
        }
        print(f"[LAYERWISE_TEST] PASS {json.dumps(report, ensure_ascii=False)}", flush=True)
        if args.output_json:
            args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        print("[LAYERWISE_TEST] model_shutdown_begin", flush=True)
        llm.llm_engine.engine_core.shutdown()
        print("[LAYERWISE_TEST] model_shutdown_complete", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"[LAYERWISE_TEST] FAIL: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
