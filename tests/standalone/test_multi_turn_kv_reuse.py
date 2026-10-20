# SPDX-License-Identifier: Apache-2.0
"""Exercise the sequential streaming client over HTTP without a real model."""

import csv
import json
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@contextmanager
def service(initial, questions, failure=None, discover=False, report_cache=True):
    state = {"requests": [], "get_models": 0, "expected": initial, "outputs": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state["get_models"] += 1
            assert self.path == "/v1/models" and discover
            data = b'{"data":[{"id":"discovered-model"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            assert self.path == "/v1/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            turn = len(state["requests"])
            state["requests"].append(body)
            assert body["prompt"] == state["expected"], f"history changed at round {turn}"
            if failure == "http" and turn == 1:
                data = b'{"error":"service failed"}'
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def send(event):
                # The SSE envelope can use CRLF and split a JSON object across
                # data fields. Initial empty chunks must not become the TTFT.
                first, rest = json.dumps(event, ensure_ascii=False).split(",", 1)
                self.wfile.write(f": keepalive\r\ndata: {first},\r\ndata: {rest}\r\n\r\n".encode())

            answer = "" if failure == "empty" else f" \r\n真实输出 {turn}：保留空白。\n\n"
            state["outputs"].append(answer)
            identifier = "cmpl-" + body["request_id"]
            send({"id": identifier, "choices": [{"index": 0, "text": "", "finish_reason": None}]})
            for text in (answer[:3], answer[3:]):
                send({"id": identifier, "choices": [{"index": 0, "text": text, "finish_reason": None}]})
            if failure == "truncated" and turn == 1:
                return
            finish = "length" if turn % 2 else "stop"
            send({"id": identifier, "choices": [{"index": 0, "text": "", "finish_reason": finish}]})
            usage = {"prompt_tokens": len(body["prompt"]), "completion_tokens": 8}
            if report_cache:
                usage["prompt_tokens_details"] = {"cached_tokens": len(initial) if turn else 0}
            send({"id": identifier, "choices": [], "usage": usage})
            self.wfile.write(b"data: [DONE]\r\n\r\n")
            if turn < len(questions):
                state["expected"] += answer + "\n\n用户：" + questions[turn] + "\n助手：\n"

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1/", state
        finally:
            server.shutdown()
            worker.join(timeout=5)


def invoke(tmp_path, url, questions, *arguments):
    prefix = tmp_path / "prefix.txt"
    prefix.write_bytes("用户：固定前缀\r\n资料。\n助手：\n".encode())
    questions_file = tmp_path / "questions.txt"
    questions_file.write_text("\n".join(questions) + "\n", encoding="utf-8")
    output = tmp_path / "artifacts"
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(REPO / "tools/multi_turn_kv_reuse.py"),
            "--base-url",
            url,
            "--prefix-file",
            str(prefix),
            "--questions-file",
            str(questions_file),
            "--output-dir",
            str(output),
            *arguments,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    return result, output


def test_initial_request_and_50_followups_keep_every_actual_answer(tmp_path):
    initial = "用户：固定前缀\r\n资料。\n助手：\n"
    assets = REPO / "tools/multi_turn_kv_assets"
    questions = assets.joinpath("questions_50.txt").read_text(encoding="utf-8").splitlines()
    assert len(questions) == 50
    with service(initial, questions) as (url, state):
        result, output = invoke(tmp_path, url, questions, "--model-name", "GLM-5.3-falcon")
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(state["requests"]) == 51
    assert state["get_models"] == 0
    assert len({r["request_id"] for r in state["requests"]}) == 51
    history = initial
    for turn, body in enumerate(state["requests"]):
        directory = output / "turns" / f"{turn:03d}"
        assert body["model"] == "GLM-5.3-falcon"
        assert body["max_tokens"] == 128
        assert body["stream_options"] == {"include_usage": True}
        if turn:
            history += state["outputs"][turn - 1] + directory.joinpath("added_input.txt").read_bytes().decode()
        assert body["prompt"] == history
        assert directory.joinpath("output.txt").read_bytes().decode() == state["outputs"][turn]
        metadata = json.loads(directory.joinpath("request.json").read_text(encoding="utf-8"))
        assert "prompt" not in metadata
        assert metadata["prompt_chars"] == len(history)
    summary = json.loads(output.joinpath("summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert len(summary["rounds"]) == 51
    assert summary["rounds"][1]["cached_tokens"] == len(initial)
    assert summary["rounds"][0]["cached_tokens"] == 0
    assert summary["rounds"][0]["completion_tokens"] == 8
    assert summary["rounds"][1]["new_input_chars"] == len(state["outputs"][0]) + len(
        "\n\n用户：" + questions[0] + "\n助手：\n"
    )
    for row in summary["rounds"]:
        assert row["latency_s"] >= row["ttft_s"] >= 0
        assert row["generation_s"] >= 0
        assert row["mean_token_gap_ms"] == pytest.approx(row["generation_s"] * 1000 / 7)
    with output.joinpath("metrics.csv").open(encoding="utf-8") as source:
        assert len(list(csv.DictReader(source))) == 51


def test_discovery_and_missing_cache_usage_are_supported(tmp_path):
    initial = "用户：固定前缀\r\n资料。\n助手：\n"
    questions = ["依据呢？", "结论呢？"]
    with service(initial, questions, discover=True, report_cache=False) as (url, state):
        result, output = invoke(tmp_path, url, questions, "--rounds", "2", "--max-tokens", "4000")
    assert result.returncode == 0, result.stderr
    assert state["get_models"] == 1
    assert all(r["model"] == "discovered-model" and r["max_tokens"] == 4000 for r in state["requests"])
    summary = json.loads(output.joinpath("summary.json").read_text(encoding="utf-8"))
    assert all(r["cached_tokens"] is None and r["uncached_prompt_tokens"] is None for r in summary["rounds"])


@pytest.mark.parametrize("failure", ["http", "truncated"])
def test_failed_round_is_not_retried_or_appended(tmp_path, failure):
    initial = "用户：固定前缀\r\n资料。\n助手：\n"
    questions = ["依据呢？", "结论呢？"]
    with service(initial, questions, failure=failure) as (url, state):
        result, output = invoke(tmp_path, url, questions, "--rounds", "2", "--model-name", "served")
    assert result.returncode == 1
    assert len(state["requests"]) == 2
    summary = json.loads(output.joinpath("summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["failed_round"] == 1
    assert len(summary["rounds"]) == 1
    assert not output.joinpath("turns/002").exists()
    if failure == "truncated":
        assert output.joinpath("turns/001/output.txt").read_bytes().decode() == state["outputs"][1]
        assert "Incomplete completion" in result.stderr
    else:
        assert "HTTP 503" in result.stderr and "service failed" in result.stderr


def test_not_enough_prewritten_questions_fails_before_http(tmp_path):
    result, output = invoke(tmp_path, "http://127.0.0.1:1", ["依据呢？"], "--rounds", "2")
    assert result.returncode == 1
    assert "Need 2 prewritten questions, found 1" in result.stderr
    assert not output.exists()


def test_empty_answer_is_archived_but_not_used_for_the_next_round(tmp_path):
    initial = "用户：固定前缀\r\n资料。\n助手：\n"
    questions = ["依据呢？"]
    with service(initial, questions, failure="empty") as (url, state):
        result, output = invoke(tmp_path, url, questions, "--rounds", "1", "--model-name", "served")
    assert result.returncode == 1
    assert len(state["requests"]) == 1
    assert "returned no text" in result.stderr
    response = json.loads(output.joinpath("turns/000/response.json").read_text(encoding="utf-8"))
    assert response["stream_done"] is True
    assert response["choices"][0]["text"] == ""
    assert not output.joinpath("turns/001").exists()


def test_default_prefix_is_a_committed_text_asset_with_reference_count():
    assets = REPO / "tools/multi_turn_kv_assets"
    manifest = json.loads(assets.joinpath("manifest.json").read_text(encoding="utf-8"))
    prefix = assets.joinpath("prefix_500k.txt").read_bytes()
    assert len(prefix) == manifest["prefix_utf8_bytes"]
    assert len(prefix.decode()) == manifest["prefix_characters"]
    assert 499_000 <= manifest["reference_token_count"] <= 501_000
    assert manifest["runtime_text_generation"] is False
    assert prefix.decode().endswith("助手：\n")


def test_default_large_prefix_is_sent_unchanged_without_tokenizer_or_resize(tmp_path):
    assets = REPO / "tools/multi_turn_kv_assets"
    initial = assets.joinpath("prefix_500k.txt").read_bytes().decode()
    questions = assets.joinpath("questions_50.txt").read_text(encoding="utf-8").splitlines()[:1]
    output = tmp_path / "default-assets"
    with service(initial, questions) as (url, state):
        result = subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                str(REPO / "tools/multi_turn_kv_reuse.py"),
                "--base-url",
                url,
                "--model-name",
                "served",
                "--rounds",
                "1",
                "--output-dir",
                str(output),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(state["requests"]) == 2
    assert state["requests"][0]["prompt"] == initial
    assert state["requests"][1]["prompt"] == initial + state["outputs"][0] + "\n\n用户：" + questions[0] + "\n助手：\n"
    assert output.joinpath("initial_prompt.txt").read_bytes() == assets.joinpath("prefix_500k.txt").read_bytes()
