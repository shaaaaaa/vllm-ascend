# SPDX-License-Identifier: Apache-2.0
"""CPU HTTP integration tests for complete decode profiling and cleanup."""

import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


@pytest.fixture
def client_tool(monkeypatch):
    tools = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("profile_decode_client", tools / "profile_decode_client.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def service():
    state = SimpleNamespace(
        calls=[],
        completions=[],
        events=[],
        profiling=False,
        fail_request=False,
        fail_start=False,
        fail_stop=False,
        output_count=None,
        raw_dir=None,
        role="kv_consumer",
        profiler={"profiler": "torch", "torch_profiler_dir": "/profiles/D0"},
        routes=[],
        received_headers=None,
        request_headers=[],
        maximum=100000,
        max_num_seqs=16,
        local_dp_size=1,
        concurrency_gate=None,
        require_profile=True,
        dynamic_route=False,
        start_lock=threading.Lock(),
        started_batches={},
        capability_version=3,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, body=None, status=200):
            data = b"" if body is None else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            state.calls.append(self.path)
            if self.path == "/openapi.json":
                self.reply({"paths": dict.fromkeys(["/tokenize", "/start_profile", "/stop_profile", "/server_info"])})
            elif self.path == "/v1/models":
                self.reply({"data": [{"id": "glm-test", "max_model_len": state.maximum}]})
            elif self.path == "/profile_routes":
                self.reply(
                    {
                        "version": state.capability_version,
                        "target_decoder": True,
                        "prefill_batch_barrier": True,
                        "deferred_profile_start": True,
                        "decoder_urls": [state.url],
                    }
                )
            elif self.path.startswith("/server_info"):
                self.reply(
                    {
                        "vllm_config": {
                            "profiler_config": state.profiler,
                            "kv_transfer_config": {"kv_role": state.role},
                            "parallel_config": {
                                "data_parallel_size_local": state.local_dp_size,
                                "data_parallel_size": state.local_dp_size,
                            },
                            "scheduler_config": {"max_num_seqs": state.max_num_seqs},
                        }
                    }
                )
            else:
                self.reply({"error": "unexpected GET"}, 404)

        def do_POST(self):
            state.calls.append(self.path)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                tokens = [11, 12, 13, 14, 15, 16, 17]
                if body["prompt"].startswith("Decode profiling session "):
                    index = int(body["prompt"].split()[-1].strip("."))
                    tokens = [100 + index]
                self.reply({"tokens": tokens})
            elif self.path == "/start_profile":
                state.profiling = True
                self.reply({"error": "start failed"} if state.fail_start else None, 500 if state.fail_start else 200)
            elif self.path == "/stop_profile":
                state.events.append("stopped")
                state.profiling = False
                if state.raw_dir:
                    state.raw_dir.mkdir()
                self.reply({"error": "stop failed"} if state.fail_stop else None, 500 if state.fail_stop else 200)
            elif self.path == "/v1/completions":
                state.completions.append(body)
                state.received_headers = dict(self.headers)
                state.request_headers.append(dict(self.headers))
                if state.concurrency_gate:
                    state.concurrency_gate.wait(timeout=10)
                start_header = self.headers.get("X-VLLM-Profile-Start")
                if start_header:
                    key = self.headers["X-VLLM-Profile-Batch"]
                    with state.start_lock:
                        if key not in state.started_batches:
                            requested_at = time.time_ns()
                            try:
                                for url in json.loads(start_header):
                                    with urlopen(Request(url + "/start_profile", data=b"{}"), timeout=5):
                                        pass
                            except HTTPError as error:
                                self.reply(json.loads(error.read()), error.code)
                                return
                            state.started_batches[key] = {
                                "requested_at_ns": requested_at,
                                "ready_at_ns": time.time_ns(),
                                "decoder_urls": json.loads(start_header),
                            }
                if state.require_profile:
                    assert state.profiling
                if state.fail_request:
                    self.reply({"error": "decode failed"}, 500)
                    return
                state.events.extend(range(body["max_tokens"]))
                self.reply(
                    {
                        "id": "decode-test",
                        "choices": [{"index": 0, "text": "answer", "finish_reason": "length"}],
                        "usage": {
                            "prompt_tokens": len(body["prompt"]),
                            "completion_tokens": state.output_count or body["max_tokens"],
                        },
                        "vllm_profile_routes": [
                            {**route, **({"profile_start": state.started_batches[key]} if start_header else {})}
                            for route in (
                                [{"decoder_url": state.url + "/v1", "request_id": "request", "decoder_dp_rank": 0}]
                                if state.dynamic_route or not state.routes else state.routes
                            )
                        ],
                    }
                )
            else:
                self.reply({"error": "unexpected POST"}, 404)

    class Server(ThreadingHTTPServer):
        request_queue_size = 64

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def invoke(tool, service, output, *extra):
    tool.main(["--base-url", service.url + "/v1", "--d-url", service.url, "--output", str(output), *extra])


def test_default_exact_tokens_and_complete_profile(client_tool, service, tmp_path):
    output = tmp_path / "capture"
    invoke(client_tool, service, output)
    (payload,) = service.completions
    assert len(payload["prompt"]) == 50000
    assert payload["max_tokens"] == payload["min_tokens"] == 128
    assert payload["ignore_eos"] is True
    assert payload["stream"] is False
    assert payload["add_special_tokens"] is False
    assert service.calls[-3:] == ["/v1/completions", "/start_profile", "/stop_profile"]
    assert service.events == [*range(128), "stopped"]
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "captured"
    assert result["profile_start_trigger"] == "proxy_after_prefill"
    assert result["profile_started_at_ns"] == result["profile_start"]["ready_at_ns"]
    assert result["measurement"]["completion_tokens"] == 128
    assert state_header(service, "x-vllm-profile-routes") == "1"
    assert result["decoder_instances"][0]["profile_dir"] == "/profiles/D0"
    assert (output / "output.txt").read_text() == "answer"


def state_header(service, name):
    return next(value for key, value in service.received_headers.items() if key.lower() == name)


def test_route_from_proxy_is_matched_to_server_profile_directory(client_tool, service, tmp_path, capsys):
    service.routes = [
        {
            "request_id": "request-routed-to-D0",
            "prefiller_url": "http://P0:7910/v1",
            "decoder_url": service.url + "/v1",
            "decoder_dp_rank": 2,
        }
    ]
    invoke(client_tool, service, tmp_path / "capture")
    result = json.loads((tmp_path / "capture/result.json").read_text())
    assert result["routing_status"] == "reported_by_proxy"
    route = result["executed_routes"][0]
    assert route["profile_dir"] == "/profiles/D0"
    assert route["trace_path_pattern"] == "/profiles/D0/*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json"
    assert route["listed_for_profiling"] is True
    assert "executed decoder=" + service.url in capsys.readouterr().out


def test_custom_lengths_and_repeatable_prompt(client_tool, service, tmp_path):
    for name in ("one", "two"):
        invoke(client_tool, service, tmp_path / name, "--prefix-tokens", "50001", "--output-tokens", "256")
    assert service.completions[0]["prompt"] == service.completions[1]["prompt"]
    assert len(service.completions[0]["prompt"]) == 50001
    assert service.completions[0]["max_tokens"] == 256


@pytest.mark.parametrize("flag, message", [("fail_request", "decode failed"), ("fail_start", "start failed")])
def test_http_failure_stops_profiler(client_tool, service, tmp_path, flag, message):
    setattr(service, flag, True)
    with pytest.raises(RuntimeError, match=message):
        invoke(client_tool, service, tmp_path / "capture")
    assert service.calls[-1] == "/stop_profile"
    assert not service.profiling
    assert json.loads((tmp_path / "capture/result.json").read_text())["status"] == "failed"


def test_stop_failure_preserves_original_error(client_tool, service, tmp_path):
    service.fail_request = service.fail_stop = True
    with pytest.raises(RuntimeError, match="decode failed"):
        invoke(client_tool, service, tmp_path / "capture")
    result = json.loads((tmp_path / "capture/result.json").read_text())
    assert "decode failed" in result["error"]
    assert "stop failed" in result["profiler_stop_errors"][0]


def test_stop_failure_marks_capture_failed(client_tool, service, tmp_path):
    service.fail_stop = True
    with pytest.raises(RuntimeError, match="Profiler stop failed"):
        invoke(client_tool, service, tmp_path / "capture")
    assert json.loads((tmp_path / "capture/result.json").read_text())["status"] == "failed"


def test_short_response_still_stops_profiler(client_tool, service, tmp_path):
    service.output_count = 7
    with pytest.raises(RuntimeError, match="128 output tokens"):
        invoke(client_tool, service, tmp_path / "capture")
    assert service.calls[-1] == "/stop_profile"
    assert (tmp_path / "capture/response.json").exists()


@pytest.mark.parametrize("setting", ["limit", "producer", "length"])
def test_preflight_fails_before_start(client_tool, service, tmp_path, setting):
    if setting == "limit":
        service.profiler["max_iterations"] = 3
    elif setting == "producer":
        service.role = "kv_producer"
    else:
        service.maximum = 50000
    with pytest.raises(ValueError):
        invoke(client_tool, service, tmp_path / "capture")
    assert "/start_profile" not in service.calls


def test_exports_only_new_profiles_after_stop(client_tool, service, tmp_path, monkeypatch):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    old = profiles / "old_ascend_pt"
    old.mkdir()
    service.raw_dir = profiles / "new_ascend_pt"
    exports = []

    def export(runs, root, prefix):
        assert service.calls[-1] == "/stop_profile"
        assert not service.profiling
        assert prefix == "[DECODE_PROFILE]"
        exports.extend(runs)
        return [str(service.raw_dir / "ASCEND_PROFILER_OUTPUT/trace_view.json")]

    monkeypatch.setattr(client_tool, "analyse_runs", export)
    invoke(client_tool, service, tmp_path / "capture", "--profile-dir", str(profiles))
    assert exports == [service.raw_dir.resolve()]
    result = json.loads((tmp_path / "capture/result.json").read_text())
    assert result["status"] == "exported"
    assert result["raw_profiles"] == [str(service.raw_dir.resolve())]


def test_proxy_receives_four_start_targets_and_client_stops_after_decode(client_tool, monkeypatch):
    events = []

    class Decoder:
        def __init__(self, name):
            self.base_url = name

        def request(self, path, body):
            events.append((self.base_url, path))

    decoders = [Decoder(f"D{index}") for index in range(4)]
    def run(*args, profile_urls):
        assert profile_urls == [decoder.base_url for decoder in decoders]
        events.append(("proxy", "decode completed"))
        return [{"routes": [{"profile_start": {"ready_at_ns": 123}}]}]

    monkeypatch.setattr(client_tool, "run_requests", run)
    client_tool.profile_request(decoders, None, "model", [[1]], 128, None, {})
    assert events == [
        ("proxy", "decode completed"),
        ("D3", "/stop_profile"),
        ("D2", "/stop_profile"),
        ("D1", "/stop_profile"),
        ("D0", "/stop_profile"),
    ]


def test_proxy_start_error_stops_requested_decoders(client_tool, monkeypatch):
    events = []

    class Decoder:
        def __init__(self, name):
            self.base_url = name

        def request(self, path, body):
            events.append((self.base_url, path))

    def fail(*args, **kwargs):
        raise TimeoutError("start reply lost")

    monkeypatch.setattr(client_tool, "run_requests", fail)

    with pytest.raises(TimeoutError, match="start reply lost"):
        client_tool.profile_request([Decoder("D0"), Decoder("D1")], None, "model", [1], 128, None, {})
    assert events == [
        ("D1", "/stop_profile"),
        ("D0", "/stop_profile"),
    ]


def test_invalid_length_fails_before_network(client_tool, tmp_path):
    with pytest.raises(ValueError, match="positive"):
        client_tool.main(
            ["--d-url", "http://127.0.0.1:1", "--prefix-tokens", "0", "--output", str(tmp_path / "capture")]
        )
    assert not (tmp_path / "capture").exists()


def test_sixteen_http_requests_run_concurrently_on_target(client_tool, service, tmp_path):
    service.dynamic_route = True
    service.concurrency_gate = threading.Barrier(16)
    invoke(
        client_tool,
        service,
        tmp_path / "capture",
        "--concurrency",
        "16",
        "--target-d-url",
        service.url,
    )
    assert len(service.completions) == 16
    assert len({tuple(item["prompt"]) for item in service.completions}) == 16
    assert all(len(item["prompt"]) == 50000 for item in service.completions)
    assert service.calls.count("/start_profile") == service.calls.count("/stop_profile") == 1
    assert service.events[-1] == "stopped"
    result = json.loads((tmp_path / "capture/result.json").read_text())
    assert result["concurrency"] == 16
    assert len(result["measurements"]) == 16
    assert all(route["decoder_url"] == service.url + "/v1" for route in result["executed_routes"])
    assert len(list((tmp_path / "capture").glob("session-*/response.json"))) == 16
    headers = [{key.lower(): value for key, value in item.items()} for item in service.request_headers]
    assert len({item["x-vllm-profile-batch"] for item in headers}) == 1
    assert {int(item["x-vllm-profile-member"]) for item in headers} == set(range(16))
    assert all(item["x-vllm-profile-target"] == service.url for item in headers)
    assert all(item["x-vllm-profile-batch-size"] == "16" for item in headers)


def test_warmup_completes_before_capture(client_tool, service, tmp_path):
    service.dynamic_route = True
    service.require_profile = False
    service.concurrency_gate = threading.Barrier(4)
    invoke(
        client_tool,
        service,
        tmp_path / "capture",
        "--concurrency",
        "4",
        "--target-d-url",
        service.url,
        "--warmup",
    )
    assert len(service.started_batches) == 1
    assert all(
        not any(key.lower() == "x-vllm-profile-start" for key in headers)
        for headers in service.request_headers[:4]
    )
    assert all(any(key.lower() == "x-vllm-profile-start" for key in headers) for headers in service.request_headers[4:])
    assert len(service.completions) == 8
    result = json.loads((tmp_path / "capture/result.json").read_text())
    assert len(result["warmup"]) == len(result["measurements"]) == 4
    assert sorted(item["prompt"] for item in service.completions[:4]) == sorted(
        item["prompt"] for item in service.completions[4:]
    )


def test_concurrent_failure_drains_other_requests_before_stop(client_tool, monkeypatch, tmp_path):
    done = threading.Event()
    events = []

    class Decoder:
        base_url = "http://D:7920"

        def request(self, path, payload):
            if path == "/stop_profile":
                assert done.is_set()
            events.append(path)

    def complete(client, model, prompt, output_tokens, root, headers):
        if prompt == [0]:
            raise RuntimeError("session failed")
        time.sleep(0.02)
        done.set()
        return {"routes": []}

    monkeypatch.setattr(client_tool, "measured_request", complete)
    with pytest.raises(RuntimeError, match="1 of 2"):
        client_tool.profile_request([Decoder()], None, "model", [[0], [1]], 128, tmp_path, {})
    assert events == ["/stop_profile"]
    assert (tmp_path / "session-000/error.json").exists()
    assert (tmp_path / "session-001/measurement.json").exists()


def test_target_capacity_rejected_before_profile(client_tool, service, tmp_path):
    service.max_num_seqs = 4
    with pytest.raises(ValueError, match="max_num_seqs=4"):
        invoke(client_tool, service, tmp_path / "capture", "--target-d-url", service.url, "--concurrency", "16")
    assert "/start_profile" not in service.calls


def test_wrong_target_rejected_before_profile(client_tool, service, tmp_path):
    with pytest.raises(ValueError, match="absent from the proxy"):
        invoke(
            client_tool,
            service,
            tmp_path / "capture",
            "--target-d-url",
            "http://wrong:7920",
            "--d-url",
            "http://wrong:7920",
        )
    assert "/start_profile" not in service.calls


def test_old_proxy_is_rejected_before_prefill_or_profile(client_tool, service, tmp_path):
    service.capability_version = 2
    with pytest.raises(ValueError, match="after prefill"):
        invoke(client_tool, service, tmp_path / "capture")
    assert "/v1/completions" not in service.calls
    assert "/start_profile" not in service.calls
