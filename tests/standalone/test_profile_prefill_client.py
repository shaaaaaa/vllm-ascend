# SPDX-License-Identifier: Apache-2.0
"""Exercise the HTTP profiling client on CPU, including its default workload."""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def client_tool():
    path = Path(__file__).resolve().parents[2] / "tools/profile_prefill_client.py"
    spec = importlib.util.spec_from_file_location("profile_prefill_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def service():
    state = SimpleNamespace(calls=[], completions=[], missing_details=False, fail_capture=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, body, status=200):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            state.calls.append(self.path)
            if self.path == "/openapi.json":
                self.reply(
                    {"paths": dict.fromkeys(["/start_profile", "/stop_profile", "/tokenize", "/v1/completions"])}
                )
            elif self.path == "/v1/models":
                self.reply({"data": [{"id": "glm-test", "max_model_len": 600000}]})
            else:
                self.reply({"error": "unexpected GET"}, 404)

        def do_POST(self):
            state.calls.append(self.path)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                self.reply({"tokens": [11, 12, 13, 14]})
            elif self.path in ("/start_profile", "/stop_profile"):
                self.reply({})
            elif self.path == "/v1/completions":
                state.completions.append(body)
                count = len(state.completions)
                if state.fail_capture and count == 4:
                    self.reply({"error": "capture failed"}, 500)
                    return
                tokens = len(body["prompt"])
                cached = [0, 1024, 1024, 512000][count - 1]
                usage = {"prompt_tokens": tokens, "completion_tokens": 1}
                if cached and not state.missing_details:
                    usage["prompt_tokens_details"] = {"cached_tokens": cached}
                self.reply({"id": f"request-{count}", "choices": [{}], "usage": usage})
            else:
                self.reply({"error": "unexpected POST"}, 404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()


def test_default_workload_and_profile_boundaries(client_tool, service, tmp_path):
    output = tmp_path / "capture"
    client_tool.main(["--base-url", service.url + "/v1", "--output", str(output)])
    short, probe, warm, measured = service.completions
    assert [len(item["prompt"]) for item in service.completions] == [1024, 1152, 512000, 512128]
    assert measured["prompt"][:512000] == warm["prompt"]
    assert measured["prompt"][512000] != probe["prompt"][1024]
    assert service.calls[-3:] == ["/start_profile", "/v1/completions", "/stop_profile"]
    assert all(item["max_tokens"] == 1 and item["add_special_tokens"] is False for item in service.completions)
    assert short["kv_transfer_params"] == {"do_remote_decode": True, "do_remote_prefill": False}
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "captured"
    assert result["measurement"]["cached_tokens"] == 512000
    assert result["measurement"]["new_prompt_tokens"] == 128


def test_missing_usage_stops_before_long_warmup(client_tool, service, tmp_path):
    service.missing_details = True
    with pytest.raises(RuntimeError, match="enable-prompt-tokens-details"):
        client_tool.main(["--base-url", service.url, "--output", str(tmp_path / "capture")])
    assert len(service.completions) == 2
    assert "/start_profile" not in service.calls


def test_request_failure_stops_profiler(client_tool, service, tmp_path):
    service.fail_capture = True
    with pytest.raises(RuntimeError, match="capture failed"):
        client_tool.main(["--base-url", service.url, "--output", str(tmp_path / "capture")])
    assert service.calls[-1] == "/stop_profile"
    result = json.loads((tmp_path / "capture/result.json").read_text())
    assert result["status"] == "failed"


def test_unaligned_prefix_rejected_before_network(client_tool, tmp_path):
    with pytest.raises(ValueError, match="divisible"):
        client_tool.main(["--prefix-tokens", "500000", "--output", str(tmp_path / "capture")])
    assert not (tmp_path / "capture").exists()


def test_wrong_hit_is_reported(client_tool):
    with pytest.raises(RuntimeError, match="actual new prompt tokens=416"):
        client_tool.require_hit({"phase": "profiled", "cached_tokens": 499712, "new_prompt_tokens": 416}, 500000)


def test_export_only_new_runs(client_tool, tmp_path, monkeypatch):
    old = tmp_path / "old_ascend_pt"
    old.mkdir()
    before = client_tool.profile_runs(tmp_path)
    new = tmp_path / "new_ascend_pt"
    new.mkdir()
    (new / "trace_view.json").write_text("{}")
    calls = []
    monkeypatch.setattr(client_tool.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    traces = client_tool.analyse_runs(client_tool.profile_runs(tmp_path) - before, tmp_path)
    assert traces == [str(new / "trace_view.json")]
    assert str(old) not in calls[0]
    assert calls[0][-1] == str(new)
