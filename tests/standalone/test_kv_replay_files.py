# SPDX-License-Identifier: Apache-2.0
"""Read real snapshot fixtures over HTTP without loading Torch or a native SDK."""

import base64
import gzip
import json
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace as NS
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import decode_kv_live_replay as D  # noqa: E402
import kv_replay_files as F  # noqa: E402
import prefill_kv_replay as P  # noqa: E402
from test_decode_kv_live_replay import source as decode_source  # noqa: E402
from test_prefill_kv_replay import log_file, make_run, snapshot  # noqa: E402


@contextmanager
def serving(plan):
    files = F.SnapshotFiles(plan)
    with ThreadingHTTPServer(("127.0.0.1", 0), files.handler()) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        owner = NS(file_info=dict(file_url=f"http://127.0.0.1:{server.server_port}/{files.nonce}", role=files.role))
        try:
            yield files, owner, build_opener(ProxyHandler({}))
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("decode", [False, True])
def test_snapshot_bytes_frontier_and_completion(tmp_path, decode):
    plan = snapshot(tmp_path / "snapshots")
    if decode:
        plan.update(replay_mode="decode_receive", cached_tokens=11, prefix_digest=P.token_digest(list(range(11))))
    with serving(plan) as (source, owner, http):
        url = owner.file_info["file_url"]
        manifest = json.load(http.open(url + "/manifest"))
        assert manifest["tokens"] == list(range(11 if decode else 8))
        assert manifest["role"] == ("D" if decode else "PD")
        for group in (0, 1):
            for start in range(0, len(source.tokens), 4):
                end = min(start + 4, len(source.tokens))
                suffix = f"/{group}/{start}/{end}"
                data = http.open(url + "/page" + suffix).read()
                expected = b"".join(
                    path.read_bytes()[start * r["row_bytes"] : end * r["row_bytes"]]
                    for r, path in source.page(group, start, end)
                )
                assert data == expected
                if start == 0 and group == 0:
                    assert source.stats()["completed_pages"] == 0  # GET does not imply a completed H2D.
                assert (
                    http.open(
                        Request(
                            url + "/ack" + suffix,
                            data=json.dumps(
                                dict(host="worker", pid=1, rank=0, role="D" if owner.file_info["role"] == "D" else "P")
                            ).encode(),
                        )
                    ).status
                    == 204
                )
        report = F.source_stats(owner, tmp_path)
        assert report["completed_pages"] == (6 if decode else 4)
        with pytest.raises(HTTPError):
            http.open(url + "/page/0/8/12")
        with pytest.raises(HTTPError):
            http.open(url + "/page/0/../manifest")


def test_incomplete_read_is_not_silently_verified(tmp_path):
    with serving(snapshot(tmp_path / "snapshots")) as (_, owner, http):
        http.open(owner.file_info["file_url"] + "/page/0/0/4").read()
        with pytest.raises(RuntimeError, match="entire selected KV prefix"):
            F.source_stats(owner, tmp_path)
        assert json.loads((tmp_path / "file_reads.json").read_text())["completed_pages"] == 0


def test_individual_layer_fallback_requires_every_layer(tmp_path):
    with serving(snapshot(tmp_path / "snapshots")) as (source, owner, http):
        url = owner.file_info["file_url"]
        for layer in (1, 0):
            data = http.open(f"{url}/layer/0/{layer}/0/4").read()
            assert data == next(path.read_bytes()[:16] for row, path in source.page(0, 0, 4, layer))
            http.open(
                Request(
                    f"{url}/ack/0/0/4/{layer}",
                    data=json.dumps(
                        dict(host="worker", pid=1, rank=0, role="D" if owner.file_info["role"] == "D" else "P")
                    ).encode(),
                )
            ).close()
            assert source.stats()["completed_pages"] == (1 if layer == 0 else 0)


def test_attach_only_changes_cache_identity_and_preserves_input(tmp_path):
    request = dict(
        prompt=[1, 2, 3],
        temperature=0,
        max_tokens=128,
        kv_transfer_params={"bootstrap_final_hidden": "original"},
    )
    owner = NS(file_info=dict(file_url="http://3-60:1234/random", role="PD"))
    F.attach_source(request, owner, tmp_path)
    assert request["prompt"] == [1, 2, 3] and request["max_tokens"] == 128
    assert request["kv_transfer_params"]["bootstrap_final_hidden"] == "original"
    tag = request["kv_transfer_params"][F.TAG]
    assert json.loads(base64.urlsafe_b64decode(tag + "=" * (-len(tag) % 4))) == dict(
        url=owner.file_info["file_url"], role="PD"
    )


def test_file_source_process_does_not_import_accelerator_or_native(tmp_path):
    plan = snapshot(tmp_path / "snapshots")
    plan.update(file_host="127.0.0.1", file_port=0, hold_seconds=0.05)
    encoded = base64.b64encode(gzip.compress(json.dumps(plan).encode())).decode()
    code = """
import builtins, sys
real = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torch_npu', 'mooncake', 'vllm', 'lmcache'}:
        raise RuntimeError('forbidden runtime import: '+name)
    return real(name, *args, **kwargs)
builtins.__import__ = guarded
sys.path.insert(0, sys.argv[1])
import prefill_kv_replay
prefill_kv_replay.main(['--file-child', sys.argv[2]])
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(P.REPO / "tools"), encoded], capture_output=True, text=True, timeout=30
    )
    assert '"status": "ready"' in result.stdout, result.stderr
    assert '"native_transport": false' in result.stdout
    assert "File replay lease expired" in result.stderr
    assert "forbidden runtime import" not in result.stderr


def test_file_metadata_does_not_read_proc_environment(tmp_path, monkeypatch):
    options = decode_source(tmp_path)
    log = Path(options["d_logs"][0])
    with log.open("a") as stream:
        stream.write("[KV_FILE_REPLAY] enabled role=D\n")
    options["proc_root"] = str(tmp_path / "no-processes")
    meta = D.metadata(options)
    assert "transport" not in meta


def test_prefill_file_plan_keeps_original_frontier(tmp_path, monkeypatch):
    make_run(tmp_path / "run")
    log_file(tmp_path / "p.log")
    with (tmp_path / "p.log").open("a") as stream:
        stream.write("[KV_FILE_REPLAY] enabled role=P\n")

    def http(url, path, body=None):
        if path == "/v1/models":
            return {"data": [{"id": "alias", "root": "/model"}]}
        return {"tokens": list(range(14 if "问题" in body["prompt"] else 11))}

    monkeypatch.setattr(P, "http_json", http)
    args = P.parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--p-logs", str(tmp_path / "p.log"), "--snapshot-host", "root@7.150.3.60"]
    )
    assert not hasattr(args, "source") and not hasattr(D.parser().parse_args([]), "source")
    _, plan = P.prepare(args, tmp_path)
    assert plan["cached_tokens"] == 8 and plan["file_host"] == "7.150.3.60"


def test_decode_reads_cannot_mask_missing_prefill_reads(tmp_path):
    with serving(snapshot(tmp_path / "snapshots")) as (files, owner, http):
        url = owner.file_info["file_url"]
        for role, host in (("D", "6-177"), ("P", "3-11")):
            for group in (0, 1):
                for start in (0, 4):
                    suffix = f"/{group}/{start}/{start + 4}"
                    # Neither reader has any local snapshot paths; only HTTP.
                    assert http.open(url + "/page" + suffix).read()
                    identity = json.dumps(dict(host=host, role=role, pid=22, rank=0)).encode()
                    http.open(Request(url + "/ack" + suffix, data=identity)).close()
            if role == "D":
                assert files.stats()["completed_pages"] == 4
                with pytest.raises(RuntimeError, match="entire selected KV prefix"):
                    F.source_stats(owner, tmp_path)
        report = F.source_stats(owner, tmp_path)
        assert report["role_completed_pages"] == {"P": 4, "D": 4}
        assert {r["host"] for r in report["readers"]} == {"3-11", "6-177"}


@pytest.mark.parametrize("parser", [P.parser, D.parser])
@pytest.mark.parametrize("flag", ["--source", "--master", "--seed-device", "--store-gb", "--seed-child"])
def test_removed_injection_arguments_are_rejected(parser, flag):
    with pytest.raises(SystemExit):
        parser().parse_args([flag, "old-value"])
