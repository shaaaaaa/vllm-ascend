#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate capture coverage for a sequential, single-machine P/D smoke run.

This validates the test execution and raw-input inventory, not numerical
accuracy. The separate PD/off-on analyzers compare the large tensor payloads.
"""

from __future__ import annotations

import json
from pathlib import Path

from pd_tensor_analyze import load_archive


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tokens(value):
    return isinstance(value, list) and all(type(item) is int and item >= 0 for item in value)


def _required(num_layers):
    result = {
        (-1, "model_input", "input_ids"),
        (-1, "model_input", "positions"),
        (-1, "model_output", "hidden_states"),
        (-1, "logits", "output"),
    }
    for layer in range(num_layers):
        for kind, names in (
            ("decoder", ("input", "output")),
            ("sfa", ("input", "output")),
            ("attention", ("query_nope", "query_rope", "logical_topk", "output")),
            ("kv_current", ("nope", "rope")),
            ("kv_consumed", ("nope", "rope")),
        ):
            result.update((layer, kind, name) for name in names)
    return result


def _input_payload(worker, record, expected):
    # Only the small, authoritative model input IDs/positions are read here.
    # Attention/KV payloads are streamed by the subsequent numeric analyzer.
    import torch

    value = torch.load(worker.directory / record["path"], map_location="cpu", weights_only=True)
    _require(isinstance(value, torch.Tensor), "model input payload is not a tensor")
    _require(
        list(value.shape) == record["shape"] and str(value.dtype) == record["dtype"], "model input shape/dtype mismatch"
    )
    _require(value.ndim == 1 and value.tolist() == expected, "raw model input differs from call metadata")


def _validate_worker(worker, prompt_ids, expected_layerwise, min_prefill_calls, output):
    manifest = worker.manifest
    role = manifest["role"]
    _require(manifest.get("complete") is True and manifest.get("request_finished") is True, "worker did not finish")
    _require(not manifest["errors"], "worker recorded errors")
    _require(manifest["layerwise_prefill"] is expected_layerwise, "unexpected layerwise_prefill state")
    _require(manifest.get("prompt_token_ids") == prompt_ids, "worker prompt differs from requested prompt")
    _require(bool(worker.calls), "worker recorded no model calls")
    calls = [
        worker.calls[number] for number in sorted(worker.calls) if worker.calls[number].get("model", "main") == "main"
    ]
    prefill_calls = [call for call in calls if call["phase"] == "prefill"]
    decode_calls = [call for call in calls if call["phase"] == "decode"]
    required = _required(manifest["num_layers"])
    all_positions = []
    for call in calls:
        _require(call["context_complete"], "call has incomplete causal token context")
        _require(bool(call["positions"]), "call contains no query rows")
        _require(len(set(call["positions"])) == len(call["positions"]), "call repeats a logical position")
        current = [record for record in worker.records if record["call"] == call["call"]]
        identities = {(record["layer"], record["kind"], record["name"]) for record in current}
        _require(required <= identities, f"call {call['call']} missing required main-backbone capture")
        for name, expected in (("input_ids", call["token_ids"]), ("positions", call["positions"])):
            record = next(
                item for item in current if (item["layer"], item["kind"], item["name"]) == (-1, "model_input", name)
            )
            capture = record.get("row_capture")
            indices = list(range(len(call["positions"])))
            if capture:
                _require(capture["source_rows"] == len(indices), "model input source row count differs from call")
                indices = capture["source_indices"]
            _require(
                record["row_axis"] == 0 and record["positions"] == [call["positions"][i] for i in indices],
                "model input row coverage differs from call",
            )
            _input_payload(worker, record, [expected[i] for i in indices])
        for position, token in zip(call["positions"], call["token_ids"]):
            if position < len(prompt_ids):
                _require(token == prompt_ids[position], "prefill query token differs from prompt")
        all_positions.extend(call["positions"])
    if role == "P":
        _require(
            len(prefill_calls) >= min_prefill_calls, "P did not exercise the requested number of chunk-prefill calls"
        )
        _require(not decode_calls, "P unexpectedly executed decode")
        _require(
            all_positions == list(range(len(prompt_ids))),
            "P model inputs do not cover the prompt exactly once in order",
        )
    else:
        _require(bool(decode_calls), "D did not execute any main decode call")
        _require(
            min(all_positions) == len(prompt_ids) - 1,
            "D did not start at the final prompt token; prefix may have recomputed",
        )
        _require(calls[0]["positions"][0] == len(prompt_ids) - 1, "D first query is not the final prompt token")
        _require(all(position >= len(prompt_ids) - 1 for position in all_positions), "D recomputed cached prompt rows")
        _require(
            len(prefill_calls) == 1 and prefill_calls[0]["positions"] == [len(prompt_ids) - 1],
            "D prefill must only recompute the final prompt token",
        )
    sampled = [token for item in worker.sampled for token in item["token_ids"]]
    _require(worker.sampled_complete, "accepted sampled-token capture incomplete")
    api_tokens = output["token_ids"]
    mtp_tokens = output.get("mtp", {}).get("configured_tokens", 0)
    _require(type(mtp_tokens) is int and mtp_tokens >= 0, "invalid configured MTP token count")
    _require(sampled[: len(api_tokens)] == api_tokens, "worker accepted tokens differ from API output")
    excess = len(sampled) - len(api_tokens)
    _require(0 <= excess <= mtp_tokens, "worker accepted token count exceeds the final MTP bonus allowance")
    if excess:
        # The runner observes accepted tokens before the scheduler truncates
        # the final MTP target-verification result at max_tokens. Only that
        # last call may produce an unreturned bonus; prior calls remain exact.
        before_final = len(sampled) - len(worker.sampled[-1]["token_ids"])
        _require(before_final < len(api_tokens), "worker continued sampling after the API output limit")
    return {
        "tp_rank": manifest["tp_rank"],
        "calls": len(calls),
        "prefill_calls": len(prefill_calls),
        "decode_calls": len(decode_calls),
        "first_position": all_positions[0],
        "last_position": all_positions[-1],
        "records": len(worker.records),
        "sampled_tokens": len(sampled),
        "api_output_tokens": len(api_tokens),
        "clipped_final_tokens": excess,
        "directory": str(worker.directory),
    }


def validate_case(
    case_root, tp_size, prompt_ids, output_tokens=16, expect_layerwise=False, min_prefill_calls=2, model_id=None
):
    """Return JSON-safe coverage evidence; never claim a numeric accuracy PASS.

    ``case_root`` contains ``capture/``, ``prefill/output.json``,
    ``decode/output.json``. The launcher validates its sealed file-backed
    Mooncake SDK store separately with ``validate_store``.
    All failures are retained in ``errors`` rather than silently ignored.
    """
    root = Path(case_root).resolve()
    errors = []
    details = {"roles": {}, "outputs": {}}
    report = {
        "schema_version": 1,
        "complete": False,
        "scope": "execution and capture coverage; numeric accuracy assessed separately",
        "errors": errors,
        "details": details,
        "paths": {"case": str(root), "capture": str(root / "capture")},
    }
    try:
        _require(type(tp_size) is int and tp_size > 0, "tp_size must be a positive integer")
        _require(_tokens(prompt_ids) and len(prompt_ids) >= 2, "prompt must contain at least two token IDs")
        _require(
            type(output_tokens) is int and output_tokens >= 2, "output_tokens must be at least two to exercise decode"
        )
        _require(type(expect_layerwise) is bool, "expect_layerwise must be boolean")
        _require(type(min_prefill_calls) is int and min_prefill_calls > 0, "min_prefill_calls must be positive")
    except ValueError as exc:
        errors.append(str(exc))
        return report

    expected_case = "on" if expect_layerwise else "off"
    for role, stage, count, cached in (("P", "prefill", 1, 0), ("D", "decode", output_tokens, len(prompt_ids) - 1)):
        try:
            output = json.loads((root / stage / "output.json").read_text(encoding="utf-8"))
            _require(output.get("completed") is True, "stage did not complete")
            _require(output.get("stage") == stage and output.get("case") == expected_case, "stage/case label mismatch")
            _require(isinstance(output.get("request_id"), str) and bool(output["request_id"]), "missing API request ID")
            _require(
                output.get("prompt_token_ids") == prompt_ids and output.get("prompt_length") == len(prompt_ids),
                "API prompt mismatch",
            )
            _require(
                _tokens(output.get("token_ids")) and len(output["token_ids"]) == count,
                "API output token count incomplete",
            )
            _require(output.get("output_token_limit") == count, "API output token limit differs from requested limit")
            _require(output.get("num_cached_tokens") == cached, "unexpected cached token count")
            _require(output.get("finish_reason") == "length", "generation did not stop at the requested output limit")
            details["outputs"][role] = output
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append(f"{role} output: {exc}")
    if len(details["outputs"]) == 2:
        p_output, d_output = details["outputs"]["P"], details["outputs"]["D"]
        if p_output["request_id"] != d_output["request_id"]:
            errors.append("P/D external API request IDs differ")
        if p_output["token_ids"][0] != d_output["token_ids"][0]:
            errors.append("P/D first output token differs")

    archive = load_archive([root / "capture"])
    details["archive_issues"] = [issue for issue in archive.issues if issue["status"] != "additional_tensor"]
    errors.extend(
        f"capture: {issue['status']}: {issue.get('reason', issue.get('worker_directory', issue.get('path', '')))}"
        for issue in details["archive_issues"]
    )
    request_ids = {worker.manifest["request_id"] for worker in archive.workers.values()}
    if len(request_ids) != 1:
        errors.append(f"expected exactly one external request ID shared by P/D, observed {sorted(request_ids)}")
    else:
        details["request_id"] = next(iter(request_ids))
    inventories = {(worker.manifest["model_id"], worker.manifest["num_layers"]) for worker in archive.workers.values()}
    if len(inventories) != 1:
        errors.append("P/D/TP model ID or layer inventories differ")
    if model_id is not None and any(item[0] != model_id for item in inventories):
        errors.append("captured model ID differs from requested model")
    for role in ("P", "D"):
        workers = sorted(
            (worker for worker in archive.workers.values() if worker.manifest["role"] == role),
            key=lambda worker: worker.manifest["tp_rank"],
        )
        if [worker.manifest["tp_rank"] for worker in workers] != list(range(tp_size)):
            errors.append(f"{role} TP rank coverage differs from 0..{tp_size - 1}")
        summaries = []
        details["roles"][role] = summaries
        for worker in workers:
            try:
                _require(worker.manifest["tp_size"] == tp_size, "worker TP size mismatch")
                output = details["outputs"].get(role)
                _require(output is not None, "no valid API output")
                _require(worker.manifest["request_id"] == output["request_id"], "capture and API request IDs differ")
                summaries.append(
                    _validate_worker(
                        worker, prompt_ids, expect_layerwise if role == "P" else False, min_prefill_calls, output
                    )
                )
            except Exception as exc:
                errors.append(f"{role} TP{worker.manifest['tp_rank']}: {exc}")
    report["complete"] = not errors
    return report
