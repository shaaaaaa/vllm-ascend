# SPDX-License-Identifier: Apache-2.0
"""Attach the existing single-request correctness probe to the real API server."""

import argparse
import functools
import json
import runpy
import sys
import time
from pathlib import Path

from layerwise_prefill_check import write_json


def capture_request(original, *, root, case, max_features, save_on_tensors, timeout, extract_tokens):
    """Tokenization/serving remain vLLM's; only the request's caller changes."""
    started = False
    case_dir = root / case

    @functools.wraps(original)
    async def generate(client, prompt, sampling_params, request_id, **kwargs):
        nonlocal started
        if started:
            raise ValueError("This correctness run expects exactly one LoCoMo request")
        started = True
        report = dict(
            case=case,
            completed=False,
            request_source="locomo",
            request_id=request_id,
            max_features=max_features,
            token_ids=[],
            text="",
            output_length=0,
        )
        installed = False
        generator = None
        try:
            from layerwise_prefill_correctness import validate_prompt_length
            from layerwise_prefill_correctness_compare import require_worker_completion

            tokens = list(extract_tokens(client, prompt))
            validate_prompt_length(len(tokens))
            if sampling_params.n != 1:
                raise ValueError("The correctness probe supports sampling n=1")
            report.update(prompt_length=len(tokens), prompt_token_ids=tokens, sampling_params=repr(sampling_params))
            if case == "off":
                write_json(root / "prompt.json", dict(length=len(tokens), token_ids=tokens, source="locomo"))
            else:
                expected = json.loads((root / "prompt.json").read_text(encoding="utf-8"))
                if tokens != expected["token_ids"]:
                    raise ValueError("LoCoMo ON prompt token IDs differ from OFF; refusing a mismatched comparison")
            write_json(case_dir / "result.json", report)
            replies = await client.collective_rpc(
                "install_correctness_probe",
                timeout=timeout,
                args=(str(case_dir), len(tokens), save_on_tensors, max_features),
            )
            installed = True
            write_json(case_dir / "installation.json", replies)
            print(f"[PREFILL_CORRECTNESS] {case}: LoCoMo prompt={len(tokens)}; max_features={max_features}", flush=True)
            begin = time.perf_counter()
            generator = original(client, prompt, sampling_params, request_id, **kwargs)
            final = None
            delta = getattr(sampling_params.output_kind, "name", "") == "DELTA"
            async for result in generator:
                if len(result.outputs) != 1:
                    raise RuntimeError("Expected exactly one completion")
                output = result.outputs[0]
                if delta:
                    report["token_ids"].extend(output.token_ids)
                    report["text"] += output.text
                else:
                    report["token_ids"] = list(output.token_ids)
                    report["text"] = output.text
                if result.finished:
                    final = result
                    # Finalize before yielding the final response, so the
                    # benchmark cannot exit ahead of the coverage files.
                    coverage = await client.collective_rpc("finish_correctness_probe", timeout=timeout)
                    installed = False
                    write_json(case_dir / "coverage.json", coverage)
                    options = json.loads((case_dir / "engine_options.json").read_text(encoding="utf-8"))
                    ranks = [row.get("rank") for row in coverage]
                    if any(type(rank) is not int for rank in ranks) or sorted(ranks) != list(
                        range(options["tensor_parallel_size"])
                    ):
                        raise RuntimeError("Missing or duplicate worker coverage ranks")
                    for row in coverage:
                        require_worker_completion(row)
                    if result.num_cached_tokens:
                        raise RuntimeError("Fresh correctness request unexpectedly reused a prompt cache hit")
                    if not report["token_ids"]:
                        raise RuntimeError("LoCoMo produced no output tokens")
                    report.update(
                        completed=True,
                        num_cached_tokens=result.num_cached_tokens,
                        output_length=len(report["token_ids"]),
                        diagnostic_seconds=time.perf_counter() - begin,
                    )
                    write_json(case_dir / "result.json", report)
                yield result
            if final is None:
                raise RuntimeError("LoCoMo request did not complete")
        except BaseException as error:
            # A consumer may close the iterator after its final response.
            if not report["completed"]:
                report.update(error=f"{type(error).__name__}: {error}")
                write_json(case_dir / "result.json", report)
            raise
        finally:
            try:
                if generator is not None:
                    await generator.aclose()
            finally:
                if installed:
                    try:
                        coverage = await client.collective_rpc("finish_correctness_probe", timeout=timeout)
                        write_json(case_dir / "coverage.json", coverage)
                    except Exception as error:
                        report["probe_cleanup_error"] = str(error)
                        write_json(case_dir / "result.json", report)

    return generate


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--capture-case", choices=("off", "on"), required=True)
    parser.add_argument("--capture-max-features", type=int, default=8)
    parser.add_argument("--capture-save-on", action="store_true")
    parser.add_argument("--capture-timeout", type=int, default=1800)
    args, server_args = parser.parse_known_args()
    from layerwise_prefill_correctness_layout import install_local_merged_layout

    install_local_merged_layout()
    from vllm.renderers.inputs.preprocess import extract_prompt_components
    from vllm.v1.engine.async_llm import AsyncLLM

    AsyncLLM.generate = capture_request(
        AsyncLLM.generate,
        root=args.capture_root.resolve(),
        case=args.capture_case,
        max_features=args.capture_max_features,
        save_on_tensors=args.capture_save_on,
        timeout=args.capture_timeout,
        extract_tokens=lambda client, prompt: extract_prompt_components(client.model_config, prompt).token_ids,
    )
    sys.argv = ["vllm.entrypoints.openai.api_server", *server_args]
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
