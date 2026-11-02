#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Send a fixed long prefix, then append actual answers and prewritten questions.

python3 tools/multi_turn_kv_reuse_concurrent.py --base-url http://127.0.0.1:8000 --model-name GLM-5.3-falcon
Uses Python 3.9+; optional real-token truncation also needs tokenizers.
Connects to an already running service.
Each session warms its prefix before the measured, concurrent follow-up rounds.
After all warmups finish, each session advances independently without round barriers.
Use multi_turn_kv_reuse.py for the original serial run including round 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, BinaryIO

QUESTION_PREFIX = "\n\n用户："
QUESTION_SUFFIX = "\n助手：\n"
DEFAULT_ROUNDS = 50
METRIC_FIELDS = (
    "round",
    "session_id",
    "concurrency",
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
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_text(path: Path) -> str:
    """Preserve CR/LF and all answer whitespace, including on Windows."""
    return path.read_bytes().decode("utf-8")


def context_token_count(value: str) -> int:
    """Accept a positive token count, optionally using decimal k/m suffixes."""
    match = re.fullmatch(r"([1-9][0-9]*)([km]?)", value.strip().lower().replace("_", ""))
    if match is None:
        raise argparse.ArgumentTypeError("Use a positive token count, e.g. 200000 or 200k")
    return int(match[1]) * {"": 1, "k": 1000, "m": 1000000}[match[2]]


def truncate_prefix(text: str, count: int | None, tokenizer_path: Path | None) -> tuple[str, dict[str, Any]]:
    """Take a real-token prefix locally; never download a tokenizer or generate text."""
    if count is None:
        return text, {"requested_context_tokens": None}
    if tokenizer_path is None:
        raise ValueError(
            "--context-tokens requires --tokenizer pointing to the served model's tokenizer directory/json"
        )
    path = tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
    if not path.is_file():
        raise ValueError(f"Tokenizer JSON does not exist: {path}")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    tokens = tokenizer.encode(text, add_special_tokens=False).ids
    selected = tokens[:count]
    result = tokenizer.decode(selected, skip_special_tokens=False) if len(tokens) > count else text
    # Cutting the article also cuts its original final question. Keep a fixed
    # reading task so warmup is a useful completion rather than a continuation.
    if len(tokens) > count:
        result += "\n\n用户：请解读以上资料，简述主要结论和依据。\n助手：\n"
    return result, {
        "requested_context_tokens": count,
        "source_tokens": len(tokens),
        "selected_prefix_tokens": len(selected),
        "tokenizer": str(path.resolve()),
        "context_note": (
            "First N article tokens plus session marker/reading task; actual length is server prompt_tokens."
        ),
    }


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
        "concurrency": args.concurrency,
        "scheduling": "independent_sessions_after_all_warmups",
        "total_requests": args.concurrency * (args.rounds + 1),
        "max_tokens": args.max_tokens,
        "initial_prompt_chars": len(initial_prompt),
        "composition": "next_prompt = previous_prompt + previous_actual_output + added_input.txt",
        "timing_scope": "Client stream timing includes HTTP transfer and queueing; token gaps are estimates.",
        "context": args.context_metadata,
        "warmups": [],
        "sessions": [],
        "completed_followup_rounds": 0,
        "rounds": [],
    }
    save_json(root / "summary.json", summary)
    current_round = None
    current_session = None
    try:
        model = args.model or client.model_name()
        summary["model"] = model
        print(
            f"[MULTI_TURN] model={model} concurrency={args.concurrency} "
            f"warmups={args.concurrency} measured={args.concurrency * args.rounds} artifacts={root.resolve()}",
            flush=True,
        )
        print(
            "[MULTI_TURN] All session warmups finish before round 1; warmup is excluded from metrics.csv.", flush=True
        )
        print(
            "[MULTI_TURN] Follow-up sessions advance independently; there is no cross-session round barrier.",
            flush=True,
        )
        print("[MULTI_TURN] cached_tokens=null means not reported; it does not mean a cache miss.", flush=True)
        sessions = []
        for session_id in range(args.concurrency):
            directory = root if args.concurrency == 1 else root / "sessions" / f"{session_id:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            marker = f"session={session_id:04d}; timestamp_ns={time.time_ns()}; run={run_id}\n\n"
            directory.joinpath("session_prefix.txt").write_bytes(marker.encode("utf-8"))
            summary["sessions"].append(
                dict(
                    session_id=session_id,
                    directory=str(directory.relative_to(root)),
                    marker=marker,
                    completed_followup_rounds=0,
                )
            )
            sessions.append(
                dict(
                    client=Client(args.base_url, args.timeout),
                    directory=directory,
                    history=marker + initial_prompt,
                    previous_output="",
                    previous_prompt_chars=0,
                )
            )
        with (
            ThreadPoolExecutor(max_workers=args.concurrency) as executor,
            root.joinpath("metrics.csv").open("w", newline="", encoding="utf-8") as metrics_file,
        ):
            metrics = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
            metrics.writeheader()
            pending = {}

            def submit_turn(session_id: int, turn: int) -> None:
                nonlocal current_session, current_round
                current_session, current_round = session_id, turn
                session = sessions[session_id]
                question = questions[turn - 1] if turn else ""
                added = QUESTION_PREFIX + question + QUESTION_SUFFIX if turn else ""
                if turn:
                    session["history"] += session["previous_output"] + added
                history = session["history"]
                directory = session["directory"] / "turns" / f"{turn:03d}"
                directory.mkdir(parents=True)
                directory.joinpath("added_input.txt").write_bytes(added.encode("utf-8"))
                request_id = f"multi-{run_id}-s{session_id:03d}-{turn:03d}"
                payload = {
                    "model": model,
                    "prompt": history,
                    "request_id": request_id,
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                save_json(
                    directory / "request.json",
                    {
                        **{key: value for key, value in payload.items() if key != "prompt"},
                        "session_id": session_id,
                        "round": turn,
                        "prompt_chars": len(history),
                        "prompt_source": "session_prefix.txt + initial_prompt.txt + prior answers/questions",
                    },
                )
                print(
                    f"[MULTI_TURN] round={turn:02d} session={session_id} "
                    f"begin request_id={request_id} chars={len(history)}",
                    flush=True,
                )
                future = executor.submit(session["client"].complete, payload, directory)
                pending[future] = (
                    session_id,
                    {
                        "round": turn,
                        "session_id": session_id,
                        "concurrency": args.concurrency,
                        "phase": "followup" if turn else "warmup",
                        "request_id": request_id,
                        "question": question,
                        "prompt_chars": len(history),
                        "new_input_chars": len(history) - session["previous_prompt_chars"],
                    },
                )

            # Only warmup has a global barrier. Each measured completion
            # releases its own session's next turn, regardless of other rounds.
            for first_turn in (0, 1):
                if first_turn > args.rounds:
                    break
                for session_id in range(args.concurrency):
                    submit_turn(session_id, first_turn)
                while pending:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    ready, failure = [], None
                    for future in completed:
                        current_session, base_row = pending.pop(future)
                        current_round = base_row["round"]
                        session = sessions[current_session]
                        try:
                            session["previous_output"], stats = future.result()
                        except Exception as error:
                            if failure is None:
                                failure = (current_session, current_round, error)
                            continue
                        row = {**base_row, **stats}
                        session["previous_prompt_chars"] = row["prompt_chars"]
                        if current_round:
                            summary["rounds"].append(row)
                            metrics.writerow(row)
                            metrics_file.flush()
                            summary["sessions"][current_session]["completed_followup_rounds"] = current_round
                            summary["completed_followup_rounds"] = min(
                                item["completed_followup_rounds"] for item in summary["sessions"]
                            )
                            if current_round < args.rounds:
                                ready.append((current_session, current_round + 1))
                        else:
                            summary["warmups"].append(row)
                        save_json(root / "summary.json", summary)
                        print("[MULTI_TURN] " + json.dumps(row, ensure_ascii=False), flush=True)
                        output_label = (
                            f"round={current_round:02d} session={current_session} "
                            f"phase={row['phase']} request_id={row['request_id']}"
                        )
                        # Print complete answers from the coordinator so concurrent
                        # streams never interleave their text in the terminal/tee log.
                        print(
                            f"[MULTI_TURN_OUTPUT] {output_label} begin\n"
                            f"{session['previous_output']}\n"
                            f"[MULTI_TURN_OUTPUT] {output_label} end",
                            flush=True,
                        )
                    if failure is not None:
                        current_session, current_round, error = failure
                        raise error
                    for session_id, turn in ready:
                        submit_turn(session_id, turn)
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            failed_round=current_round,
            failed_session=current_session,
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
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Independent sessions; all warm up before measured rounds"
    )
    parser.add_argument(
        "--context-tokens", type=context_token_count, help="Use only the first N prefix tokens, e.g. 200k"
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Served model's local tokenizer directory or tokenizer.json; needed for truncation",
    )
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum answer tokens per request (default: 128)")
    parser.add_argument("--timeout", type=float, default=3600, help="HTTP socket timeout in seconds (default: 3600)")
    parser.add_argument("--output-dir", type=Path, help="New or empty artifact directory")
    args = parser.parse_args(argv)
    if (
        args.rounds < 0
        or args.concurrency < 1
        or args.max_tokens <= 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error("rounds must be nonnegative; concurrency, max-tokens and timeout must be positive")
    try:
        initial_prompt = read_text(args.prefix_file)
        questions_text = read_text(args.questions_file)
        questions = [line.strip() for line in questions_text.splitlines() if line.strip()]
        if not initial_prompt.strip():
            raise ValueError("Prefix file is empty")
        if len(questions) < args.rounds:
            raise ValueError(f"Need {args.rounds} prewritten questions, found {len(questions)}")
        initial_prompt, args.context_metadata = truncate_prefix(initial_prompt, args.context_tokens, args.tokenizer)
        root = args.output_dir or Path(
            f"multi-turn-kv-concurrent-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        )
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise ValueError("--output-dir must be empty")
        root.joinpath("initial_prompt.txt").write_bytes(initial_prompt.encode("utf-8"))
        root.joinpath("questions.txt").write_bytes(questions_text.encode("utf-8"))
        run(args, root, initial_prompt, questions)
    except urllib.error.HTTPError as error:
        print(f"[MULTI_TURN] HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"[MULTI_TURN] FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
