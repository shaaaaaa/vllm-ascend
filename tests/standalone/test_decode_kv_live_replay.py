# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import decode_kv_live_replay as R  # noqa: E402
import prefill_kv_seed as S  # noqa: E402
from test_prefill_kv_replay import log_file, snapshot  # noqa: E402


def source(tmp_path, tail_token=False):
    p = snapshot(tmp_path / "snapshots")
    log = tmp_path / "D_dp0.log"
    log_file(log)
    with log.open("a") as stream:
        stream.write(f"kv_connector_extra_config={{'skip_last_n_tokens': {int(tail_token)}}}\n")
    for path in (tmp_path / "snapshots").glob("*/rank*/replay.json"):
        data = dict(
            request=dict(
                schema=1,
                prompt_token_ids=list(range(11)) + ([99] if tail_token else []),
                sampling=dict(
                    temperature=0, max_tokens=128, detokenize=True, bad_words=[], skip_reading_prefix_cache=False
                ),
                cache_salt=None,
            ),
            runtime=dict(engine_options=dict(model="/model", tensor_parallel_size=2)),
        )
        path.write_text(json.dumps(data))
        manifest = path.with_name("manifest.json")
        row = json.loads(manifest.read_text())
        row["replay_file"] = "replay.json"
        manifest.write_text(json.dumps(row))
    return dict(request_id="response-0", snapshot_root=p["snapshot_root"], d_logs=[str(log)])


def legacy_source(tmp_path):
    options = source(tmp_path)
    for path in (tmp_path / "snapshots").glob("*/rank*/replay.json"):
        path.unlink()
        manifest = path.with_name("manifest.json")
        row = json.loads(manifest.read_text())
        row.pop("replay_file")
        manifest.write_text(json.dumps(row))
    root = tmp_path / "multi-turn-concurrent"
    directory = root / "sessions" / "011" / "turns" / "001"
    directory.mkdir(parents=True)
    request = dict(
        model="glm",
        request_id="client-11-r1",
        session_id=11,
        round=1,
        prompt_chars=123,
        prompt_source="session_prefix.txt + initial_prompt.txt + prior answers/questions",
        max_tokens=128,
        temperature=0,
        stream=True,
        stream_options=dict(include_usage=True),
    )
    response = dict(
        id="response-0",
        usage=dict(prompt_tokens=11),
        stream_done=True,
        choices=[dict(index=0, text="old output", finish_reason="length")],
    )
    (directory / "request.json").write_text(json.dumps(request))
    (directory / "response.json").write_text(json.dumps(response))
    (root / "summary.json").write_text(
        json.dumps(
            dict(
                warmups=[],
                rounds=[dict(request_id="client-11-r1", response_id="response-0", session_id=11, round=1)],
            )
        )
    )
    return options, directory


def args(tmp_path):
    return R.parser().parse_args(
        [
            "--request-id",
            "response-0",
            "--d-url",
            "http://D:7920",
            "--workspace",
            str(tmp_path),
        ]
    )


@pytest.mark.parametrize("tail_token", [False, True])
def test_metadata_prepare_uses_complete_receive_and_isolated_keys(tmp_path, monkeypatch, tail_token):
    options = source(tmp_path, tail_token)
    import decode_kv_replay

    monkeypatch.setattr(decode_kv_replay, "equal_payload", lambda *_: pytest.fail("metadata scanned full KV"))
    meta = dict(R.metadata(options), snapshot_root=options["snapshot_root"])
    cli = args(tmp_path)
    models = {"data": [{"id": "GLM-5.3-falcon", "root": "/model", "max_model_len": 1000}]}
    payload, plan, cached = R.prepare(cli, meta, models, tmp_path)
    assert plan["cached_tokens"] == 11  # not rounded down to eight!
    assert plan["seed_mode"] == "decode_receive"
    assert cached == (11 if tail_token else 10)
    assert payload["prompt"] == list(range(11)) + ([99] if tail_token else [])
    assert payload["kv_transfer_params"] == {"lmcache.tag.decode_replay": plan["replay_tag"]}
    assert plan["replay_tag"] in payload["cache_salt"]
    assert payload["model"] == "GLM-5.3-falcon"
    tokens, _, _ = S.snapshot_metadata(plan)
    assert tokens == list(range(11))
    _, second, _ = R.prepare(cli, meta, models, tmp_path)
    assert second["replay_tag"] != plan["replay_tag"]


def test_decode_partial_exception_does_not_relax_prefill_frontier(tmp_path):
    plan = snapshot(tmp_path / "snapshots")
    plan.update(cached_tokens=11, prefix_digest=S.token_digest(list(range(11))))
    with pytest.raises(ValueError, match="full cache-block boundary"):
        S.snapshot_metadata(plan)
    plan["seed_mode"] = "decode_receive"
    assert S.snapshot_metadata(plan)[0] == list(range(11))
    plan["cached_tokens"] = 9
    with pytest.raises(ValueError, match="full cache-block boundary"):
        S.snapshot_metadata(plan)


def test_partial_page_key_hash_tags_and_plane_bytes(tmp_path, monkeypatch):
    import hashlib

    from test_prefill_kv_replay import test_ascend_hash_input_and_lmcache_key_serializer

    # Install the same serializer boundary mocks used by the P replay tests.
    test_ascend_hash_input_and_lmcache_key_serializer(monkeypatch)
    layout = sys.modules["lmcache.v1.mooncake_layout"]
    layout.MOONCAKE_VALID_TOKENS_TAG = "lmcache.tag.internal.valid_tokens"
    hashes = []

    def digest(value):
        hashes.append(value)
        return hashlib.sha256(repr(value).encode()).digest()

    monkeypatch.setitem(sys.modules, "vllm.utils.hashing", NS(get_hash_fn_by_name=lambda _: digest))
    plan = snapshot(tmp_path / "snapshots")
    plan.update(
        seed_mode="decode_receive",
        cached_tokens=11,
        prefix_digest=S.token_digest(list(range(11))),
        replay_tag="one-replay",
    )
    tokens, rows, _ = S.snapshot_metadata(plan)
    pages = list(S.page_keys(plan, tokens))
    assert [(g, a, b) for _, g, a, b in pages][-2:] == [(0, 8, 11), (1, 8, 11)]
    assert hashes[-1][1] == (8, 9, 10)
    for (identity, _), group, start, end in pages:
        tags = identity[5]
        assert tags["lmcache.tag.decode_replay"] == "one-replay"
        assert tags.get(layout.MOONCAKE_VALID_TOKENS_TAG) == (3 if start == 8 else None)
        buf = bytearray(32)
        n = S.fill_page(rows, group, start, end, memoryview(buf))
        expected = b"".join(
            p.read_bytes()[start * row["row_bytes"] : end * row["row_bytes"]]
            for (g, _, _), (row, p) in sorted(rows.items())
            if g == group
        )
        assert bytes(buf[:n]) == expected


def test_payload_preserves_bootstrap_stops_and_actual_token_ids():
    original = dict(
        prompt_token_ids=[10, 11, 12],
        sampling=dict(
            n=1,
            temperature=0,
            seed=42,
            max_tokens=127,
            min_tokens=4,
            stop=["用户："],
            stop_token_ids=[3],
            extra_args={"kv_transfer_params": {"bootstrap_final_hidden": {"data": "encoded-hidden"}}},
        ),
    )
    before = json.dumps(original)
    value = R.request_payload(original, "glm", "salt", None)
    assert value["prompt"] == [10, 11, 12]
    assert value["kv_transfer_params"]["bootstrap_final_hidden"] == {"data": "encoded-hidden"}
    assert (value["seed"], value["max_tokens"], value["stop_token_ids"]) == (42, 127, [3])
    assert json.dumps(original) == before
    assert not value["add_special_tokens"]


@pytest.mark.parametrize(
    "change",
    [
        {"bad_words": ["no"]},
        {"future_sampler": True},
        {"extra_args": {"kv_transfer_params": {"remote_fill_result": {}}}},
    ],
)
def test_unsupported_sampler_does_not_silently_change_request(change):
    with pytest.raises(ValueError):
        R.request_payload(
            {"sampling": dict(temperature=0, max_tokens=128, **change), "prompt_token_ids": [1]}, "glm", "salt", None
        )


@pytest.mark.parametrize("change,match", [({"skip_last_n_tokens": 1}, "lookup frontier"), ({"tp_size": 4}, "TP sizes")])
def test_incompatible_live_d_fails_before_seed(tmp_path, change, match):
    options = source(tmp_path)
    meta = dict(R.metadata(options), snapshot_root=options["snapshot_root"], **change)
    with pytest.raises(ValueError, match=match):
        R.prepare(args(tmp_path), meta, {"data": [{"id": "glm", "root": "/model"}]}, tmp_path)


@pytest.mark.parametrize(
    "cached,expected_status", [(10, "replay_complete"), (0, "failed"), (None, "replay_unverified")]
)
def test_existing_d_lifecycle_no_model_or_service_launch(tmp_path, monkeypatch, cached, expected_status):
    options = source(tmp_path)
    meta = dict(R.metadata(options), snapshot_root=options["snapshot_root"])
    monkeypatch.setattr(R, "REPO", tmp_path)
    events = []

    def http(url, path, **kwargs):
        assert url == "http://D:7920" and path == "/v1/models"
        return {"data": [{"id": "glm", "root": "/model"}]}

    @contextmanager
    def seed(args, plan, output):
        assert plan["cached_tokens"] == 11
        events.append("seed")
        yield NS(poll=lambda: None)
        events.append("released")

    class Client:
        def __init__(self, url, timeout):
            assert url == "http://D:7920"

        def complete(self, payload, output):
            assert events == ["seed"]
            assert payload["prompt"] == list(range(11))
            (output / "output.txt").write_text("saved output")
            events.append("D completed")
            return "saved output", {"prompt_tokens": 11, "cached_tokens": cached, "response_id": "cmpl-new"}

    monkeypatch.setattr(R, "http_json", http)
    monkeypatch.setattr(R, "collect_metadata", lambda *_: meta)
    monkeypatch.setattr(R, "seeded", seed)
    monkeypatch.setattr(R, "Client", Client)
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("unexpected process launch"))
    command = ["--request-id", "response-0", "--d-url", "http://D:7920", "--workspace", str(tmp_path)]
    if cached == 0:
        with pytest.raises(RuntimeError, match="Cache mismatch"):
            R.main(command)
    else:
        assert R.main(command) == (2 if cached is None else 0)
        assert events[-1] == "released"
    report = next((tmp_path / "decode-kv-replays").glob("*/report.json"))
    assert json.loads(report.read_text())["status"] == expected_status
    assert report.with_name("output.txt").read_text() == "saved output"


def test_entrypoint_does_not_import_accelerator_or_vllm():
    code = (
        f"import sys; sys.path.insert(0, {str(R.REPO / 'tools')!r}); "
        "import decode_kv_replay, decode_kv_live_replay; "
        "assert not any(x in sys.modules for x in ('torch', 'torch_npu', 'vllm'))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("legacy", [False, True])
def test_metadata_child_binary_protocol(tmp_path, legacy):
    import base64
    import gzip

    options = legacy_source(tmp_path)[0] if legacy else source(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(R.REPO / "tools" / "decode_kv_replay.py"),
            "--metadata-child",
            base64.b64encode(json.dumps(options).encode()).decode(),
        ],
        capture_output=True,
        check=True,
    )
    value = json.loads(gzip.decompress(result.stdout))
    assert value["received_tokens"] == list(range(11))
    assert value["contract"]["master"] == "10.0.0.1:50051"
    assert value["skip_last_n_tokens"] == 0
    assert (value["request"] is None) == legacy
    assert value["runtime"]["engine_options"]["model"] == "/model"


@pytest.mark.parametrize("legacy", [False, True])
def test_real_http_single_request_retains_seed_until_done(tmp_path, monkeypatch, legacy):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    options = legacy_source(tmp_path)[0] if legacy else source(tmp_path)
    monkeypatch.chdir(tmp_path)
    meta = dict(R.metadata(options), snapshot_root=options["snapshot_root"])
    monkeypatch.setattr(R, "REPO", tmp_path)
    monkeypatch.setattr(R, "collect_metadata", lambda *_: meta)
    active, requests, failures = [], [], []

    @contextmanager
    def seed(args, plan, output):
        # The full main path must hand the legacy raw snapshot to the native
        # seed's real metadata/layout validator, including the partial tail.
        assert S.snapshot_metadata(plan)[0] == list(range(11))
        active.append(plan["replay_tag"])
        try:
            yield NS(poll=lambda: None)
        finally:
            active.clear()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "glm", "root": "/model"}]}).encode())

        def do_POST(self):
            requests.append(self.path)
            value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not active or value["kv_transfer_params"]["lmcache.tag.decode_replay"] != active[0]:
                failures.append("seed absent")
            if value["prompt"] != list(range(11)):
                failures.append("tokens changed")
            event = dict(
                id="cmpl-replay",
                choices=[dict(index=0, text="test output", finish_reason="length")],
                usage=dict(prompt_tokens=11, completion_tokens=2, prompt_tokens_details=dict(cached_tokens=10)),
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setattr(R, "seeded", seed)
    try:
        assert (
            R.main(
                [
                    "--request-id",
                    "response-0",
                    "--d-url",
                    f"http://127.0.0.1:{server.server_port}",
                    "--workspace",
                    str(tmp_path),
                ]
            )
            == 0
        )
    finally:
        server.shutdown()
        worker.join()
        server.server_close()
    assert requests == ["/v1/models", "/v1/completions"]
    assert not failures and not active
    response = next((tmp_path / "decode-kv-replays").glob("*/response.json"))
    assert json.loads(response.read_text())["choices"][0]["text"] == "test output"
    report = json.loads(response.with_name("report.json").read_text())
    if legacy:
        assert report["request_reconstruction"]["session_id"] == 11
        assert report["request_reconstruction"]["round"] == 1
        assert report["request_reconstruction"]["input_token_ids"] == "snapshot:tokens.json"
    assert not list((tmp_path / "snapshots").rglob("replay.json")) if legacy else True


@pytest.mark.parametrize("problem", ["wrong_request", "wrong_length", "model_missing", "tail_missing", "duplicate"])
def test_legacy_rejects_mismatched_evidence_without_guessing(tmp_path, problem):
    options, directory = legacy_source(tmp_path)
    if problem == "model_missing":
        manifest = next((tmp_path / "snapshots").glob("*/rank*/manifest.json"))
        data = json.loads(manifest.read_text())
        data.pop("model")
        manifest.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="model identity"):
            R.metadata(options)
        return
    meta = R.metadata(options)
    if problem == "wrong_request":
        path = directory / "request.json"
        data = json.loads(path.read_text())
        data["request_id"] = "different-request"
        path.write_text(json.dumps(data))
    elif problem == "wrong_length":
        path = directory / "response.json"
        data = json.loads(path.read_text())
        data["usage"]["prompt_tokens"] = 12
        path.write_text(json.dumps(data))
    elif problem == "tail_missing":
        meta["skip_last_n_tokens"] = 1
    else:
        (tmp_path / "multi-turn-concurrent" / "response.json").write_bytes((directory / "response.json").read_bytes())
    with pytest.raises(ValueError):
        R.restore_legacy_request(meta, tmp_path / "multi-turn-concurrent", "response-0")


def test_legacy_missing_client_data_gives_existing_data_instruction(tmp_path):
    options, _ = legacy_source(tmp_path)
    with pytest.raises(ValueError, match="set --run-dir to the original"):
        R.restore_legacy_request(R.metadata(options), tmp_path / "not-the-run", "response-0")
