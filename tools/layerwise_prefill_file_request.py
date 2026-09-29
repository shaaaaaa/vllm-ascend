# SPDX-License-Identifier: Apache-2.0
"""HTTP request entry for the existing file-PD model stages and tensor probe."""

import argparse
import functools
import json
import runpy
import socket
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from layerwise_prefill_check import write_json


def check_port():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 8000))


def wait_for_server(proc, timeout):
    deadline = time.monotonic() + timeout
    opener = build_opener(ProxyHandler({}))
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("API server exited before LoCoMo; inspect server.log")
        try:
            with opener.open("http://127.0.0.1:8000/health", timeout=2):
                return
        except (URLError, TimeoutError, ConnectionError):
            time.sleep(1)
    raise TimeoutError("API server did not become ready; inspect server.log")


def mtp_snapshot():
    from prometheus_client import REGISTRY

    names = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens", "vllm:spec_decode_num_accepted_tokens")
    result = {name: None for name in names}
    for family in REGISTRY.collect():
        for sample in family.samples:
            name = sample.name.removesuffix("_total")
            if name in result:
                result[name] = (result[name] or 0) + float(sample.value)
    return result


def request_entry(
    original, *, root, stage, output_tokens, timeout, extract_tokens, build_sampling, metrics=mtp_snapshot
):
    """Keep the original probe, generation settings, store fence and coverage."""
    started = False
    directory = root / stage

    @functools.wraps(original)
    async def generate(client, prompt, sampling_params, request_id, **kwargs):
        nonlocal started
        if started:
            raise ValueError("Expected exactly one LoCoMo request per file-check stage")
        started = True
        report = dict(
            schema_version=1,
            stage=stage,
            completed=False,
            token_ids=[],
            text="",
            output_token_limit=output_tokens,
            enforce_eager=True,
            timing_is_performance_data=False,
        )
        installed = closed = False
        stream = None
        try:
            from layerwise_prefill_file_check import load_worker_coverage

            tokens = list(extract_tokens(client, prompt))
            options = json.loads((directory / "engine_options.json").read_text(encoding="utf-8"))
            if len(tokens) < 2 or len(tokens) + output_tokens > options["max_model_len"]:
                raise ValueError(f"LoCoMo prompt={len(tokens)}; require >=2 and prompt + outputs <= max-model-len")
            if sampling_params.n != 1:
                raise ValueError("File check supports one completion per request")
            if stage == "baseline":
                write_json(root / "prompt.json", dict(length=len(tokens), token_ids=tokens, source="locomo"))
            elif tokens != json.loads((root / "prompt.json").read_text(encoding="utf-8"))["token_ids"]:
                raise ValueError("LoCoMo tokenized prompt differs from OFF")
            report.update(prompt_length=len(tokens), prompt_token_ids=tokens)
            # Preserve the file-check's deterministic sampling and output limits.
            params = build_sampling(output_tokens, sampling_params.output_kind)
            replies = await client.collective_rpc("install_file_probe", timeout=timeout, args=(str(directory), tokens))
            installed = True
            write_json(directory / "installation.json", replies)
            use_mtp = bool(options.get("speculative_config"))
            before = metrics() if use_mtp else {}
            begin = time.perf_counter()
            stream = original(client, prompt, params, request_id, **kwargs)
            delta = getattr(params.output_kind, "name", "") == "DELTA"
            async for result in stream:
                if len(result.outputs) != 1:
                    raise RuntimeError("Expected exactly one completion")
                completion = result.outputs[0]
                if delta:
                    report["token_ids"].extend(completion.token_ids)
                    report["text"] += completion.text
                else:
                    report["token_ids"], report["text"] = list(completion.token_ids), completion.text
                if result.finished:
                    after = metrics() if use_mtp else {}
                    report.update(
                        num_cached_tokens=result.num_cached_tokens,
                        finish_reason=completion.finish_reason,
                        diagnostic_seconds=time.perf_counter() - begin,
                        mtp=dict(
                            configured_tokens=int(use_mtp),
                            metrics={
                                name: after[name] - before[name]
                                if after[name] is not None and before[name] is not None
                                else None
                                for name in before
                            },
                        ),
                    )
                    flushed = await client.collective_rpc("flush_file_store", timeout=timeout)
                    write_json(directory / "store_flush.json", flushed)
                    replies = await client.collective_rpc("finish_file_probe", timeout=timeout)
                    installed = False
                    coverage = load_worker_coverage(directory, replies, options["tensor_parallel_size"])
                    write_json(directory / "coverage.json", coverage)
                    if len(report["token_ids"]) != output_tokens:
                        raise RuntimeError("Generation did not reach the requested token count")
                    if use_mtp and stage != "prefill":
                        count = report["mtp"]["metrics"]["vllm:spec_decode_num_draft_tokens"]
                        if count is None or count <= 0:
                            raise RuntimeError("MTP was configured but no target verification was observed")
                    expected = len(tokens) - 1 if stage == "decode" else 0
                    if result.num_cached_tokens != expected:
                        raise RuntimeError(f"{stage}: cached_tokens={result.num_cached_tokens}, expected {expected}")
                    if any(not row.get("complete") or row.get("errors") for row in coverage):
                        raise RuntimeError("Incomplete tensor coverage; inspect coverage.json")
                    await client.collective_rpc("close_file_store", timeout=timeout)
                    closed = True
                    report["completed"] = True
                    write_json(directory / "output.json", report)
                    (directory / "output.txt").write_text(report["text"], encoding="utf-8")
                yield result
            if not report["completed"]:
                raise RuntimeError("LoCoMo request did not complete")
        except BaseException as error:
            if not report["completed"]:
                report["error"] = f"{type(error).__name__}: {error}"
                write_json(directory / "output.json", report)
            raise
        finally:
            try:
                if stream is not None:
                    await stream.aclose()
                if installed:
                    await client.collective_rpc("finish_file_probe", timeout=timeout)
            finally:
                if not closed:
                    await client.collective_rpc("close_file_store", timeout=timeout)

    return generate


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--capture-stage", choices=("baseline", "prefill", "decode"), required=True)
    parser.add_argument("--capture-output-tokens", type=int, required=True)
    parser.add_argument("--capture-rpc-timeout", type=int, required=True)
    args, rest = parser.parse_known_args()
    from layerwise_prefill_file_store import install

    install()
    from vllm import SamplingParams
    from vllm.renderers.inputs.preprocess import extract_prompt_components
    from vllm.v1.engine.async_llm import AsyncLLM

    AsyncLLM.generate = request_entry(
        AsyncLLM.generate,
        root=args.capture_root,
        stage=args.capture_stage,
        output_tokens=args.capture_output_tokens,
        timeout=args.capture_rpc_timeout,
        extract_tokens=lambda client, prompt: extract_prompt_components(client.model_config, prompt).token_ids,
        build_sampling=lambda count, kind: SamplingParams(
            temperature=0, seed=1024, max_tokens=count, ignore_eos=True, output_kind=kind
        ),
    )
    sys.argv = ["vllm.entrypoints.openai.api_server", *rest]
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
