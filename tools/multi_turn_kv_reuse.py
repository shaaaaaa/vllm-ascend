#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Send a fixed long prefix, then append actual answers and prewritten questions.

python3 tools/multi_turn_kv_reuse.py --base-url http://127.0.0.1:8000 --model-name GLM-5.3-falcon
Uses Python 3.9+ standard library only; connects to an already running service.
The default 50 follow-up rounds mean 51 sequential completion requests in total.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

QUESTION_PREFIX = "\n\n用户："
QUESTION_SUFFIX = "\n助手：\n"
DEFAULT_ROUNDS = 50
METRIC_FIELDS = (
    "round",
    "phase",
    "request_id",
    "response_id",
    "question",
    "prompt_chars",
    "new_input_chars",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "uncached_prompt_tokens",
    "latency_s",
    "ttft_s",
    "generation_s",
    "mean_token_gap_ms",
    "finish_reason",
)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    """Preserve CR/LF and all answer whitespace, including on Windows."""
    return path.read_bytes().decode("utf-8")


def sse_data(response: BinaryIO) -> Iterator[str]:
    """Read SSE data, including comments, CRLF and multiline JSON events."""
    pending = []
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if pending:
                yield "\n".join(pending)
                pending = []
        elif line.startswith("data:"):
            data = line[len("data:") :]
            pending.append(data[1:] if data.startswith(" ") else data)
    if pending:
        yield "\n".join(pending)


class Client:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[: -len("/v1")]
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def model_name(self) -> str:
        with self.opener.open(self.base_url + "/v1/models", timeout=self.timeout) as response:
            models = json.load(response).get("data", [])
        if not models or not isinstance(models[0].get("id"), str):
            raise RuntimeError("No served model found; specify --model-name")
        return models[0]["id"]

    def complete(self, payload: dict[str, Any], directory: Path) -> tuple[str, dict[str, Any]]:
        request = urllib.request.Request(
            self.base_url + "/v1/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        chunks, usage, response_id, finish_reason = [], {}, None, None
        first_text, last_text, done = None, None, False
        started = time.perf_counter()
        # Output and events survive a timeout or truncated response; never retry
        # silently, advance the dialogue, or substitute an unfinished answer.
        with (
            directory.joinpath("output.txt").open("wb") as output,
            directory.joinpath("events.jsonl").open("w", encoding="utf-8") as events,
            self.opener.open(request, timeout=self.timeout) as response,
        ):
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" not in content_type:
                raise RuntimeError(f"Expected an SSE completion stream, got {content_type!r}")
            for data in sse_data(response):
                if data.strip() == "[DONE]":
                    done = True
                    break
                received = time.perf_counter()
                event = json.loads(data)
                events.write(json.dumps(event, ensure_ascii=False) + "\n")
                if event.get("error"):
                    raise RuntimeError(f"Completion error: {event['error']}")
                response_id = event.get("id") or response_id
                if event.get("usage") is not None:
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("index", 0) != 0:
                        raise RuntimeError("Expected exactly one completion choice")
                    text = choice.get("text", "")
                    if not isinstance(text, str):
                        raise RuntimeError("Completion text must be a string")
                    if text:
                        if first_text is None:
                            first_text = received
                        last_text = received
                        chunks.append(text)
                        output.write(text.encode("utf-8"))
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
        elapsed = time.perf_counter() - started
        text = "".join(chunks)
        save_json(
            directory / "response.json",
            {
                "id": response_id,
                "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
                "usage": usage,
                "stream_done": done,
            },
        )
        if not done or finish_reason not in ("stop", "length"):
            raise RuntimeError(f"Incomplete completion: done={done}, finish_reason={finish_reason!r}")
        if not text:
            raise RuntimeError("Completion returned no text; the next round was not sent")
        details = usage.get("prompt_tokens_details") or {}
        prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
        cached_tokens = details.get("cached_tokens")
        generation_s = last_text - first_text
        stats = {
            "response_id": response_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "uncached_prompt_tokens": (
                prompt_tokens - cached_tokens if prompt_tokens is not None and cached_tokens is not None else None
            ),
            "latency_s": elapsed,
            "ttft_s": first_text - started,
            "generation_s": generation_s,
            "mean_token_gap_ms": (
                generation_s * 1000 / (completion_tokens - 1)
                if completion_tokens is not None and completion_tokens > 1
                else None
            ),
            "finish_reason": finish_reason,
        }
        return text, stats


def run(args: argparse.Namespace, root: Path, initial_prompt: str, questions: list[str]) -> None:
    client = Client(args.base_url, args.timeout)
    run_id = uuid.uuid4().hex[:12]
    summary = {
        "status": "running",
        "run_id": run_id,
        "base_url": client.base_url,
        "model": args.model,
        "followup_rounds": args.rounds,
        "total_requests": args.rounds + 1,
        "max_tokens": args.max_tokens,
        "initial_prompt_chars": len(initial_prompt),
        "composition": "next_prompt = previous_prompt + previous_actual_output + added_input.txt",
        "timing_scope": "Client stream timing includes HTTP transfer and queueing; token gaps are estimates.",
        "rounds": [],
    }
    save_json(root / "summary.json", summary)
    current_round = None
    try:
        model = args.model or client.model_name()
        summary["model"] = model
        print(f"[MULTI_TURN] model={model} requests={args.rounds + 1} artifacts={root.resolve()}", flush=True)
        print("[MULTI_TURN] Fixed text is unchanged; actual token count comes from server usage.", flush=True)
        print("[MULTI_TURN] cached_tokens=null means not reported; it does not mean a cache miss.", flush=True)
        history, previous_output, previous_prompt_chars = initial_prompt, "", 0
        with root.joinpath("metrics.csv").open("w", newline="", encoding="utf-8") as metrics_file:
            metrics = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
            metrics.writeheader()
            for current_round in range(args.rounds + 1):
                question = questions[current_round - 1] if current_round else ""
                added = QUESTION_PREFIX + question + QUESTION_SUFFIX if current_round else ""
                if current_round:
                    history += previous_output + added
                directory = root / "turns" / f"{current_round:03d}"
                directory.mkdir(parents=True)
                directory.joinpath("added_input.txt").write_bytes(added.encode("utf-8"))
                request_id = f"multi-{run_id}-{current_round:03d}"
                payload = {
                    "model": model,
                    "prompt": history,
                    "request_id": request_id,
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                # The initial prompt is stored once. Each exact later input can
                # be reconstructed without writing another 500k-token copy.
                save_json(
                    directory / "request.json",
                    {
                        **{key: value for key, value in payload.items() if key != "prompt"},
                        "prompt_chars": len(history),
                        "prompt_source": "summary.json composition",
                    },
                )
                print(
                    f"[MULTI_TURN] round={current_round:02d} begin request_id={request_id} chars={len(history)}",
                    flush=True,
                )
                previous_output, stats = client.complete(payload, directory)
                row = {
                    "round": current_round,
                    "phase": "followup" if current_round else "initial",
                    "request_id": request_id,
                    "question": question,
                    "prompt_chars": len(history),
                    "new_input_chars": len(history) - previous_prompt_chars,
                    **stats,
                }
                previous_prompt_chars = len(history)
                summary["rounds"].append(row)
                metrics.writerow(row)
                metrics_file.flush()
                save_json(root / "summary.json", summary)
                print("[MULTI_TURN] " + json.dumps(row, ensure_ascii=False), flush=True)
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            failed_round=current_round,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        save_json(root / "summary.json", summary)


def main(argv: list[str] | None = None) -> int:
    assets = Path(__file__).resolve().parent / "multi_turn_kv_assets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Service or PD proxy address")
    parser.add_argument("--model-name", "--model", dest="model", help="Served model name; default: /v1/models")
    parser.add_argument("--prefix-file", type=Path, default=assets / "prefix_500k.txt")
    parser.add_argument("--questions-file", type=Path, default=assets / "questions_50.txt")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="Follow-ups AFTER the initial request")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum answer tokens per request (default: 128)")
    parser.add_argument("--timeout", type=float, default=3600, help="HTTP socket timeout in seconds (default: 3600)")
    parser.add_argument("--output-dir", type=Path, help="New or empty artifact directory")
    args = parser.parse_args(argv)
    if args.rounds < 0 or args.max_tokens <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("rounds must be nonnegative; max-tokens and timeout must be positive")
    try:
        initial_prompt = read_text(args.prefix_file)
        questions_text = read_text(args.questions_file)
        questions = [line.strip() for line in questions_text.splitlines() if line.strip()]
        if not initial_prompt.strip():
            raise ValueError("Prefix file is empty")
        if len(questions) < args.rounds:
            raise ValueError(f"Need {args.rounds} prewritten questions, found {len(questions)}")
        root = args.output_dir or Path(f"multi-turn-kv-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}")
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise ValueError("--output-dir must be empty")
        root.joinpath("initial_prompt.txt").write_bytes(initial_prompt.encode("utf-8"))
        root.joinpath("questions.txt").write_bytes(questions_text.encode("utf-8"))
        run(args, root, initial_prompt, questions)
    except urllib.error.HTTPError as error:
        print(f"[MULTI_TURN] HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[MULTI_TURN] FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
