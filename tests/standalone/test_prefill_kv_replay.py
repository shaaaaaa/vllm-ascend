# SPDX-License-Identifier: Apache-2.0
"""CPU validation of exact turn reconstruction and snapshot validation."""

import hashlib
import json
import shlex
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
import kv_replay_snapshot as S  # noqa: E402
import prefill_kv_replay as R  # noqa: E402


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_run(root):
    prefix, article, answer, added = "unique\r\n", "原文\n", "回答\r\n", "\n用户：问题\n助手：\n"
    history = prefix + article
    session = root / "sessions" / "000"
    session.mkdir(parents=True)
    (root / "initial_prompt.txt").write_bytes(article.encode())
    (session / "session_prefix.txt").write_bytes(prefix.encode())
    rows = []
    for turn in range(2):
        folder = session / "turns" / f"{turn:03}"
        folder.mkdir(parents=True)
        if turn:
            history += answer + added
        (folder / "added_input.txt").write_bytes((added if turn else "").encode())
        (folder / "output.txt").write_bytes(("坏答案" if turn else answer).encode())
        payload = dict(
            model="alias",
            request_id=f"request-{turn}",
            temperature=0,
            max_tokens=128,
            stream=True,
            stream_options={"include_usage": True},
            session_id=0,
            round=turn,
            prompt_chars=len(history),
            prompt_source="pieces",
        )
        save(folder / "request.json", payload)
        rows.append(
            dict(
                session_id=0,
                round=turn,
                response_id=f"response-{turn}",
                request_id=f"request-{turn}",
                prompt_tokens=11 + turn * 3,
                cached_tokens=8 if turn else None,
            )
        )
    save(
        root / "summary.json",
        dict(
            sessions=[dict(session_id=0, directory="sessions/000")],
            warmups=rows[:1],
            rounds=rows[1:],
        ),
    )
    return history


def schema():
    return dict(
        version=3,
        use_mla=True,
        use_layerwise=True,
        dsa_two_groups=True,
        page_first_multi_buffer=True,
        save_chunk_meta=False,
        chunk_size=4,
        dtype="torch.bfloat16",
        group0_raw_token_dim=2,
        group1_raw_token_dim=2,
        runtime_kv_groups=[[2, ["l0", "l1"]], [1, ["i0"]]],
    )


def contract():
    value = schema()
    signature = hashlib.blake2b(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(), digest_size=8
    ).hexdigest()
    return dict(schema=value, signature=signature, algorithm="sha256_cbor", hash_root=repr(b"root"))


def log_file(path, hash_root=b"root", pid=1):
    c = contract()
    path.write_text(
        f"(Worker_DP0_TP1_EP1 pid={pid}) Initialized NONE_HASH={hash_root!r} from vLLM (>= PR#20511)\n"
        f"(Worker_DP0_TP1_EP1 pid={pid}) Using hash algorithm=sha256_cbor; Mooncake payload layout: "
        f"signature={c['signature']} role=worker rank=1 schema={c['schema']!r} (token_database.py)\n"
        "[KV_FILE_REPLAY] enabled role=P\n[KV_FILE_REPLAY] enabled role=D\n",
        encoding="utf-8",
    )


def snapshot(root):
    tokens = list(range(11))
    for rank in range(2):
        folder = root / "response-0" / f"rank{rank}"
        folder.mkdir(parents=True)
        save(folder / "tokens.json", tokens)
        files = []
        for group, count in enumerate((2, 1)):
            for layer in range(count):
                name = f"g{group}-l{layer}-c0.bin"
                data = bytes((i + group * 30 + layer * 20) % 256 for i in range(44))
                (folder / name).write_bytes(data)
                files.append(
                    dict(
                        group=group,
                        layer=layer,
                        component=0,
                        row_bytes=4,
                        dtype="torch.bfloat16",
                        start=0,
                        end=11,
                        bytes=44,
                        path=name,
                    )
                )
        save(
            folder / "manifest.json",
            dict(
                format="decode-kv-raw-v1",
                request_id="response-0",
                model="/model",
                status="complete",
                tp_rank=rank,
                tp_size=2,
                byteorder=sys.byteorder,
                tokens=11,
                groups=[2, 1],
                layer_names=[["l0", "l1"], ["i0"]],
                files=files,
            ),
        )
        save(folder / "replay.json", dict(runtime=dict(engine_options=dict(model="/model"))))
    return dict(
        snapshot_root=str(root),
        previous="response-0",
        cached_tokens=8,
        prefix_digest=S.token_digest(tokens[:8]),
        previous_prompt_tokens=11,
        previous_prompt_digest=S.token_digest(tokens),
        model_path="/model",
        contract=contract(),
        tp_size=2,
        workers=2,
        release_file=str(root / "lease.release"),
        hold_seconds=2,
        local_hostname="127.0.0.1",
    )


def test_reconstruct_exact_payload_and_no_new_session_prefix(tmp_path):
    history = make_run(tmp_path)
    result = R.restore_turn(tmp_path, 0, 1)
    assert result["payload"]["prompt"] == history
    assert "\r\n" in history
    assert result["previous_text"] == "unique\r\n原文\n"
    assert result["payload"]["temperature"] == 0
    assert result["payload"]["max_tokens"] == 128
    assert result["payload"]["request_id"].startswith("replay-")
    assert "prompt_chars" not in result["payload"]
    assert result["cached_tokens"] == 8


def test_changed_client_files_fail_before_file_setup(tmp_path):
    make_run(tmp_path)
    (tmp_path / "initial_prompt.txt").write_text("changed")
    with pytest.raises(ValueError, match="length differs"):
        R.restore_turn(tmp_path, 0, 1)


def test_parse_live_abi_and_conflicting_hash_roots(tmp_path):
    log_file(tmp_path / "a.log")
    c = S.live_contract([tmp_path / "a.log"])
    assert c["hash_root"] == repr(b"root")
    assert c["tp_size"] == 2
    log_file(tmp_path / "b.log", hash_root=b"other", pid=2)
    with pytest.raises(ValueError, match="one common"):
        S.live_contract([tmp_path / "a.log", tmp_path / "b.log"])


def test_tail_excluded_and_group_plane_order(tmp_path):
    plan = snapshot(tmp_path)
    tokens, rows = S.validate_snapshot(plan)
    assert tokens == list(range(8))  # Never source the saved 8..11 tail; it must be recomputed.
    target = b"".join(p.read_bytes()[16:32] for (g, _, _), (r, p) in sorted(rows.items()) if g == 0)
    assert target == bytes(range(16, 32)) + bytes(range(36, 52))


def test_legacy_manifest_without_replay_metadata(tmp_path):
    plan = snapshot(tmp_path)
    for path in tmp_path.rglob("replay.json"):
        path.unlink()
    tokens, rows = S.validate_snapshot(plan)
    assert tokens == list(range(8))
    assert len(rows) == 3


@pytest.mark.parametrize("runtime", [None, {"error": "old metadata capture failed"}])
def test_manifest_does_not_require_d_launch_metadata(tmp_path, runtime):
    plan = snapshot(tmp_path)
    for path in tmp_path.rglob("replay.json"):
        save(path, dict(runtime=runtime))
    assert S.validate_snapshot(plan)[0] == list(range(8))


@pytest.mark.parametrize("failure", ["missing_model", "manifest_conflict", "replay_conflict", "layout", "tokens"])
def test_legacy_compatibility_keeps_model_and_payload_checks(tmp_path, failure):
    plan = snapshot(tmp_path)
    for path in tmp_path.rglob("replay.json"):
        path.unlink()
    # Check TP1 as well: its identity must not be discarded when reading TP0.
    directory = tmp_path / "response-0/rank1"
    path = directory / "manifest.json"
    meta = json.loads(path.read_text())
    if failure == "missing_model":
        meta.pop("model")
    elif failure == "manifest_conflict":
        meta["model"] = "/different-model"
        save(directory / "replay.json", dict(runtime=dict(engine_options=dict(model="/model"))))
    elif failure == "replay_conflict":
        save(directory / "replay.json", dict(runtime=dict(engine_options=dict(model="/different-model"))))
    elif failure == "layout":
        meta["files"][0]["dtype"] = "torch.float16"
    else:
        plan["prefix_digest"] = "wrong"
    save(path, meta)
    with pytest.raises(ValueError):
        S.validate_snapshot(plan)


def test_replay_model_remains_a_supported_identity_source(tmp_path):
    plan = snapshot(tmp_path)
    for path in tmp_path.rglob("manifest.json"):
        meta = json.loads(path.read_text())
        meta.pop("model")
        save(path, meta)
    assert S.validate_snapshot(plan)[0] == list(range(8))


def test_child_failure_message_reaches_caller(tmp_path):
    log = tmp_path / "source.log"
    log.write_text(
        S.PREFIX + '{"status": "validating"}\n' + R.PREFIX + " FAILED: source model differs\n", encoding="utf-8"
    )
    process = NS(poll=lambda: 1, returncode=1)
    with pytest.raises(RuntimeError, match="File source failed: source model differs"):
        R.wait_file_source(process, log, 1)


@pytest.mark.parametrize("failure", ["tokens", "tp_bytes", "layout", "model", "partial"])
def test_incompatible_snapshots_rejected(tmp_path, failure):
    plan = snapshot(tmp_path)
    if failure == "tokens":
        plan["prefix_digest"] = "wrong"
    elif failure == "tp_bytes":
        path = tmp_path / "response-0/rank1/g1-l0-c0.bin"
        data = bytearray(path.read_bytes())
        data[1] ^= 1
        path.write_bytes(data)
    elif failure == "layout":
        plan["contract"]["schema"]["runtime_kv_groups"][0][1][0] = "wrong"
    elif failure == "model":
        plan["model_path"] = "/wrong"
    else:
        plan["cached_tokens"] = 7
    with pytest.raises(ValueError):
        S.validate_snapshot(plan)


def test_prepare_uses_live_tokenizer_and_does_not_run_inference(tmp_path, monkeypatch):
    make_run(tmp_path / "run")
    log_file(tmp_path / "p.log")
    calls = []

    def http(url, path, body=None):
        calls.append(path)
        if path == "/v1/models":
            return {"data": [{"id": "alias", "root": "/model"}]}
        return {"tokens": list(range(14 if "问题" in body["prompt"] else 11))}

    monkeypatch.setattr(R, "http_json", http)
    args = R.parser().parse_args(
        [
            "--local-hostname",
            "127.0.0.1",
            "--run-dir",
            str(tmp_path / "run"),
            "--base-url",
            "http://proxy:8000",
            "--p-url",
            "http://P:7910",
            "--p-logs",
            str(tmp_path / "p.log"),
        ]
    )
    (tmp_path / "out").mkdir()
    restored, plan = R.prepare(args, tmp_path / "out")
    assert calls == ["/v1/models", "/tokenize", "/tokenize"]
    assert plan["cached_tokens"] == 8
    assert plan["file_host"] == "127.0.0.1"
    tail = json.loads((tmp_path / "out/input_tail_tokens.json").read_text())
    assert tail["token_ids"] == list(range(8, 14))
    assert "kv_transfer_params" not in restored["payload"]


def test_prefill_source_log_paths_belong_to_remote_container(tmp_path, monkeypatch):
    make_run(tmp_path / "run")
    log_file(tmp_path / "p.log")

    def http(url, path, body=None):
        if path == "/v1/models":
            return {"data": [{"id": "alias", "root": "/model"}]}
        return {"tokens": list(range(14 if "问题" in body["prompt"] else 11))}

    monkeypatch.setattr(R, "http_json", http)
    args = R.parser().parse_args(
        [
            "--local-hostname",
            "127.0.0.1",
            "--run-dir",
            str(tmp_path / "run"),
            "--p-url",
            "http://P:7910",
            "--p-logs",
            str(tmp_path / "p.log"),
            "--snapshot-host",
            "root@D",
            "--workspace",
            "/workspace/sqh/vllm-ascend",
        ]
    )
    (tmp_path / "out").mkdir()
    _, plan = R.prepare(args, tmp_path / "out")
    assert plan["snapshot_root"] == "/workspace/sqh/vllm-ascend/decode-kv-snapshots"
    assert "transport" not in plan  # Never reuse the client's P-local IP/interface.


def test_release_cannot_touch_serving_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "REPO", tmp_path)
    with pytest.raises(ValueError):
        R.release_child(str(tmp_path / "tp4_dp4_logs" / "P.log"))
    path = tmp_path / "prefill-kv-replays/leases" / ("a" * 32 + ".release")
    R.release_child(str(path))
    assert path.exists()


def indexer_snapshot(tmp_path, policy):
    plan = snapshot(tmp_path)
    value = plan["contract"]["schema"]
    # Production resolve_mooncake_dsa_raw_token_dims returns max packet BYTES
    # for C8, including mixed precision. It does not return head_dim here.
    value["group1_raw_token_dim"] = 130 if all(policy) else 256
    names = [f"i{i}" for i in range(len(policy))]
    value["runtime_kv_groups"][1] = [len(policy), names]
    value["indexer_execution"] = dict(
        layout="dsa-index-c8-planar-v1" if all(policy) else "dsa-index-mixed-planar-v1",
        quantization="hadamard128-dynamic-int8-fp16-scale-v1",
        head_dim=128,
        key_dtype="int8" if all(policy) else "per-layer-int8-or-model-dtype",
        scale_dtype="float16",
        storage_dtype="uint8",
    )
    if not all(policy):
        value["indexer_execution"]["c8_layers"] = list(policy)
    components = []
    for layer, c8 in enumerate(policy):
        if c8:
            components.extend([(layer, 0, 128, "torch.int8", 21), (layer, 1, 2, "torch.float16", 77)])
        else:
            components.append((layer, 0, 256, "torch.bfloat16", 109))
    for rank in range(2):
        folder = tmp_path / f"response-0/rank{rank}"
        (folder / "replay.json").unlink()  # The reported failure used legacy snapshots.
        meta = json.loads((folder / "manifest.json").read_text())
        meta["groups"][1] = len(policy)
        meta["layer_names"][1] = names
        meta["files"] = [r for r in meta["files"] if r["group"] == 0]
        for layer, component, width, dtype, byte in components:
            name = f"g1-l{layer}-c{component}.bin"
            (folder / name).write_bytes(bytes([byte]) * 11 * width)
            meta["files"].append(
                dict(
                    group=1,
                    layer=layer,
                    component=component,
                    row_bytes=width,
                    dtype=dtype,
                    start=0,
                    end=11,
                    bytes=11 * width,
                    path=name,
                )
            )
        save(folder / "manifest.json", meta)
    return plan, components


@pytest.mark.parametrize(
    "policy",
    [(False, True), (True, False), (True, True), (False, False), (False,) * 3 + (True,) * 16 + (False,) * 3],
)
def test_mixed_indexer_wire_order(tmp_path, policy):
    plan, components = indexer_snapshot(tmp_path, policy)
    _, rows = S.validate_snapshot(plan)
    target = b"".join(
        p.read_bytes()[4 * r["row_bytes"] : 8 * r["row_bytes"]] for (g, _, _), (r, p) in sorted(rows.items()) if g == 1
    )
    assert target == b"".join(bytes([byte]) * (4 * width) for _, _, width, _, byte in components)


@pytest.mark.parametrize(
    "layer,component,dtype,width",
    [
        (0, 0, "torch.bfloat16", 512),
        (1, 0, "torch.int8", 256),
        (1, 1, "torch.float32", 4),
    ],
)
def test_mixed_indexer_still_rejects_wrong_planes(tmp_path, layer, component, dtype, width):
    plan, _ = indexer_snapshot(tmp_path, (False, True))
    folder = tmp_path / "response-0/rank0"
    path = folder / "manifest.json"
    meta = json.loads(path.read_text())
    for row in meta["files"]:
        if (row["group"], row["layer"], row["component"]) == (1, layer, component):
            row.update(dtype=dtype, row_bytes=width, bytes=11 * width)
            (folder / row["path"]).write_bytes(bytes(11 * width))
    save(path, meta)
    with pytest.raises(ValueError, match="actual=.*expected="):
        S.validate_snapshot(plan)


def test_real_case_frontier_recomputes_372_not_90(tmp_path, monkeypatch):
    make_run(tmp_path / "run")
    summary_path = tmp_path / "run/summary.json"
    summary = json.loads(summary_path.read_text())
    summary["warmups"][0]["prompt_tokens"] = 499994
    summary["rounds"][0].update(prompt_tokens=500084, cached_tokens=499712)
    save(summary_path, summary)
    log_file(tmp_path / "p.log")

    def http(url, path, body=None):
        if path == "/v1/models":
            return {"data": [{"id": "alias", "root": "/model"}]}
        return {"tokens": list(range(500084 if "问题" in body["prompt"] else 499994))}

    monkeypatch.setattr(R, "http_json", http)
    args = R.parser().parse_args(
        ["--local-hostname", "127.0.0.1", "--run-dir", str(tmp_path / "run"), "--p-logs", str(tmp_path / "p.log")]
    )
    out = tmp_path / "out"
    out.mkdir()
    _, plan = R.prepare(args, out)
    tail = json.loads((out / "input_tail_tokens.json").read_text())
    assert plan["cached_tokens"] == 499712
    assert tail["token_ids"] == list(range(499712, 500084))
    assert len(tail["token_ids"]) == 372


@pytest.mark.parametrize("cached", [8, 0, None])
def test_existing_pd_orchestration_and_cache_miss_verdict(tmp_path, monkeypatch, cached):
    make_run(tmp_path / "run")
    snapshot(tmp_path / "snapshots")
    log_file(tmp_path / "p.log")
    monkeypatch.setattr(R, "REPO", tmp_path)
    active = []
    calls = []
    connected = []

    def check_proxy(url, timeout):
        assert url == "http://proxy:8000"
        connected.append(url)

    def http(url, path, body=None):
        calls.append((url, path))
        if path == "/v1/models":
            return {"data": [{"id": "alias", "root": "/model"}]}
        return {"tokens": list(range(14 if "问题" in body["prompt"] else 11))}

    @contextmanager
    def file_source(args, plan, output):
        assert connected == ["http://proxy:8000"]
        active.append("source")
        yield NS(poll=lambda: None, file_info=dict(file_url="http://snapshot:1234/test", role="PD"))
        active.clear()

    class Client:
        def __init__(self, url, timeout):
            assert url == "http://proxy:8000"

        def complete(self, payload, output):
            assert active == ["source"]
            assert payload["prompt"].startswith("unique\r\n原文")
            return "回放回答", dict(
                cached_tokens=cached,
                prompt_tokens=14,
                uncached_prompt_tokens=6,
                ttft_s=0.1,
                response_id="replay-response",
            )

    monkeypatch.setattr(R, "http_json", http)
    monkeypatch.setattr(R, "check_proxy_connection", check_proxy)
    monkeypatch.setattr(R, "file_source", file_source)
    monkeypatch.setattr(R, "source_stats", lambda *_: {"tested_elsewhere": True})
    monkeypatch.setattr(R, "Client", Client)
    argv = [
        "--local-hostname",
        "127.0.0.1",
        "--run-dir",
        str(tmp_path / "run"),
        "--snapshot-root",
        str(tmp_path / "snapshots"),
        "--p-logs",
        str(tmp_path / "p.log"),
        "--base-url",
        "http://proxy:8000",
        "--p-url",
        "http://P:7910",
    ]
    if cached in (8, None):
        assert R.main(argv) == (0 if cached == 8 else 2)
        assert not active
    else:
        with pytest.raises(RuntimeError, match="Cache mismatch"):
            R.main(argv)
    (path,) = (tmp_path / "prefill-kv-replays").glob("*/report.json")
    report = json.loads(path.read_text())
    expected_status = {8: "replay_complete", 0: "failed", None: "replay_unverified"}[cached]
    assert report["status"] == expected_status
    assert report["request_completed"]
    if cached is None:
        assert report["count_validation"]["missing"] == ["cached_tokens"]
        assert report["metrics"]["cached_tokens"] is None
        assert "error" not in report
    assert report["accuracy_verdict"] == "not_assessed"
    assert calls == [("http://P:7910", p) for p in ("/v1/models", "/tokenize", "/tokenize")]


@pytest.mark.parametrize(
    "prompt,cached,missing",
    [
        (14, 8, []),
        (14, None, ["cached_tokens"]),
        (None, 8, ["prompt_tokens"]),
        (None, None, ["prompt_tokens", "cached_tokens"]),
    ],
)
def test_response_count_validation_distinguishes_missing(prompt, cached, missing):
    metrics = dict(prompt_tokens=prompt, cached_tokens=cached)
    result = R.validate_response_counts(metrics, 14, 8)
    assert result == dict(
        status="unverified" if missing else "matched", missing=missing, expected=dict(prompt_tokens=14, cached_tokens=8)
    )
    assert metrics == dict(prompt_tokens=prompt, cached_tokens=cached)


@pytest.mark.parametrize(
    "metrics",
    [
        dict(prompt_tokens=15, cached_tokens=None),
        dict(prompt_tokens=None, cached_tokens=0),
        dict(prompt_tokens=14, cached_tokens="8"),
    ],
)
def test_response_count_mismatch_still_fails_when_other_count_missing(metrics):
    with pytest.raises(RuntimeError, match="mismatch"):
        R.validate_response_counts(metrics, 14, 8)


def test_proxy_check_connects_without_sending_a_request():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(3)
        R.check_proxy_connection(f"http://127.0.0.1:{listener.getsockname()[1]}/v1", 3)
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(3)
            assert connection.recv(1) == b""


def test_proxy_refusal_prevents_snapshot_file_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "REPO", tmp_path)
    monkeypatch.setattr(R, "prepare", lambda *_: pytest.fail("Prepared snapshots despite proxy refusal"))
    monkeypatch.setattr(R, "file_source", lambda *_: pytest.fail("Started file setup despite proxy refusal"))
    with socket.socket() as unlistened:
        unlistened.bind(("127.0.0.1", 0))
        base = f"http://127.0.0.1:{unlistened.getsockname()[1]}"
        with pytest.raises(RuntimeError, match="Cannot connect to PD proxy.*KV file reading has not started"):
            R.main(["--run-dir", str(tmp_path), "--base-url", base, "--p-url", "http://P:7910"])
    (report_path,) = (tmp_path / "prefill-kv-replays").glob("*/report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert "KV file reading has not started" in report["error"]


@pytest.mark.parametrize("base", ["", "localhost:8000", "ftp://localhost:8000"])
def test_proxy_check_rejects_invalid_urls(base):
    with pytest.raises(ValueError, match="http"):
        R.check_proxy_connection(base, 3)


def test_remote_child_argv_preserves_linux_path_and_quoting(tmp_path):
    args = R.parser().parse_args(
        [
            "--local-hostname",
            "127.0.0.1",
            "--snapshot-host",
            "root@host",
            "--workspace",
            "/work space/repo",
            "--container",
            "serving",
            "--remote-python",
            "/usr/bin/python3",
        ]
    )
    command = R.child_argv(args, "--release-child", "/work space/repo/leases/test.release")
    assert command[:2] == ["ssh", "-T"]
    assert shlex.split(command[-1]) == [
        "docker",
        "exec",
        "--",
        "serving",
        "/usr/bin/python3",
        "/work space/repo/tools/prefill_kv_replay.py",
        "--release-child",
        "/work space/repo/leases/test.release",
    ]


def test_failure_message_preserves_extension_loader_cause():
    try:
        raise RuntimeError("Failed to load backend torch_npu") from ImportError(
            "test_native.so: cannot open shared object"
        )
    except RuntimeError as error:
        message = R.format_failure(error)
    assert "RuntimeError: Failed to load backend torch_npu" in message
    assert "ImportError: test_native.so: cannot open shared object" in message
