# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual client against a local HTTP service without model dependencies."""

import json
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@contextmanager
def mock_service(tokenize=True, first_output=" \n第一轮的真实输出。\r\n\n", models_available=True):
    completions = []
    tokenizations = []

    class Handler(BaseHTTPRequestHandler):
        def respond(self, body, status=200):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models" and models_available:
                self.respond({"data": [{"id": "mock-model"}]})
            else:
                self.respond({"error": "unknown endpoint"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                tokenizations.append(body)
                self.respond(
                    {"count": len(body["prompt"])} if tokenize else {"error": "unavailable"}, 200 if tokenize else 404
                )
            elif self.path == "/v1/completions":
                completions.append(body)
                turn = len(completions)
                self.respond(
                    {
                        "id": f"round-{turn}",
                        "choices": [{"text": first_output if turn == 1 else "第二轮输出", "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": len(body["prompt"]),
                            "completion_tokens": 12,
                            "prompt_tokens_details": {"cached_tokens": 0 if turn == 1 else 2816},
                        },
                    }
                )
            else:
                self.respond({"error": "unknown endpoint"}, 404)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1/", completions, tokenizations
        finally:
            server.shutdown()
            worker.join(timeout=5)


def run_client(url, output_dir, *extra_args):
    script = Path(__file__).resolve().parents[2] / "tools/two_turn_kv_reuse.py"
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(script), "--base-url", url, "--output-dir", str(output_dir), *extra_args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )


@pytest.mark.parametrize("tokenize", [True, False])
@pytest.mark.parametrize("model_option", [None, "--model-name", "--model"])
def test_two_sequential_requests_preserve_real_history(tmp_path, tokenize, model_option):
    output_dir = tmp_path / "case"
    first_output = " \n第一轮的真实输出。\r\n\n"
    model_name = "mock-model" if model_option is None else "GLM-5.3-falcon"
    extra_args = [] if model_option is None else [model_option, model_name]
    with mock_service(tokenize=tokenize, first_output=first_output, models_available=model_option is None) as (
        url,
        requests,
        tokenizations,
    ):
        result = run_client(url, output_dir, *extra_args)
    assert result.returncode == 0, result.stderr
    assert len(requests) == 2
    first, second = requests
    assert first["model"] == second["model"] == model_name
    assert all(request["model"] == model_name for request in tokenizations)
    assert first["temperature"] == second["temperature"] == 0
    added = (output_dir / "round2_added_input.txt").read_bytes().decode("utf-8")
    assert second["prompt"] == first["prompt"] + first_output + added
    assert "公共交通与城市更新" in first["prompt"]
    assert "城市低碳转型与公共治理" in added
    articles = Path(__file__).resolve().parents[2] / "tools/two_turn_kv_articles"
    assert articles.joinpath("round1.txt").read_text(encoding="utf-8") in first["prompt"]
    assert articles.joinpath("round2.txt").read_text(encoding="utf-8") in added
    for turn, request in enumerate(requests, 1):
        assert json.loads((output_dir / f"round{turn}_request.json").read_text(encoding="utf-8")) == request
        assert (output_dir / f"round{turn}_input.txt").read_bytes().decode("utf-8") == request["prompt"]
    assert (output_dir / "round1_output.txt").read_bytes().decode("utf-8") == first_output
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["round2"]["cached_tokens"] == 2816
    if tokenize:
        assert len(tokenizations) == 2
        assert summary["first_input_tokens"] == len(first["prompt"])
        assert summary["second_new_input_tokens"] == len(added)
    else:
        assert len(tokenizations) == 1
        assert summary["first_input_tokens"] is None
        assert summary["second_new_input_tokens"] is None
        assert "keeping the prewritten articles unchanged" in result.stdout


def test_empty_first_output_stops_before_second_request(tmp_path):
    output_dir = tmp_path / "case"
    with mock_service(first_output="") as (url, requests, _):
        result = run_client(url, output_dir)
    assert result.returncode != 0
    assert "Round 1 returned no text" in result.stderr
    assert len(requests) == 1
    assert (output_dir / "round1_response.json").exists()
    assert not (output_dir / "round2_request.json").exists()
