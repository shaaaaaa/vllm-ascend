# SPDX-License-Identifier: Apache-2.0
"""CPU/file and actual compressed RPC tests for bandwidth-bounded comparison."""

import ast
import base64
import builtins
import gzip
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_decode_kv_compare import archive

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import decode_kv_compare as cli  # noqa: E402
import decode_kv_compare_segments as client  # noqa: E402
import decode_kv_remote as worker  # noqa: E402


class FilesystemClient:
    """Execute the production RPC handlers against independent host roots."""

    def __init__(self, roots):
        self.roots = roots
        self.calls = []
        self.received_bytes = 0

    def call(self, host, payload):
        self.calls.append((host, payload["action"]))
        payload = json.loads(json.dumps(payload))
        if payload["action"] == "inventory":
            payload["workspace"] = str(self.roots[host])
        result = worker.dispatch(payload)
        compressed = client.pack(result)
        self.received_bytes += len(compressed)
        return json.loads(gzip.decompress(compressed))


def setup(tmp_path, *, same_host=False, count=10000, width=2, size=1):
    roots = {"p": tmp_path / "host-a"}
    if not same_host:
        roots["q"] = tmp_path / "host-b"
    target = "p" if same_host else "q"
    for rank in range(size):
        data = bytes(i % 251 for i in range(count * width))
        archive(
            roots["p"] / "decode-kv-snapshots", "before", list(range(count)), data, rank=rank, size=size, width=width
        )
        archive(
            roots[target] / "decode-kv-snapshots",
            "after",
            list(range(count + 3)),
            data + b"X" * (3 * width),
            rank=rank,
            size=size,
            width=width,
        )
    output = tmp_path / "report"
    output.mkdir()
    args = NS(
        hosts=list(roots),
        workspace="unused",
        snapshot_dir=None,
        previous="before",
        current="after",
        output=output,
        segment_tokens=4096,
        max_diff_mib=64,
        workers=2,
    )
    return args, FilesystemClient(roots), roots[target] / "decode-kv-snapshots/after"


def corrupt(path, position):
    with path.open("r+b") as stream:
        stream.seek(position)
        old = stream.read(1)
        stream.seek(position)
        stream.write(bytes([old[0] ^ 1]))


def test_500k_full_prefix_partial_segment_and_appended_tail(tmp_path):
    args, rpc, _ = setup(tmp_path, count=500000, size=4)
    report = client.analyze_remote(args, rpc)
    assert report["status"] == "history_checksums_match"
    assert report["common_prefix_tokens"] == 500000
    assert report["current_uncompared_range"] == [500000, 500003]
    assert report["checksummed_bytes_per_snapshot"] == 4_000_000
    assert report["segments"] == 4 * 123
    assert report["tp_ranks"] == [0, 1, 2, 3]
    assert report["raw_cross_host_bytes_selected"] == 0
    assert not any(action in ("blocks", "compare") for _, action in rpc.calls)


def test_one_bit_cross_host_exact_token_byte_and_reuse(tmp_path):
    args, rpc, after = setup(tmp_path)
    corrupt(after / "host-tp0/data.bin", 4096 * 2 + 1)
    report = client.analyze_remote(args, rpc)
    assert report["status"] == "history_differs"
    assert report["different_bytes"] == 1
    assert report["different_segments"] == 1
    assert report["raw_cross_host_bytes_selected"] == 2 * 4096 * 2
    first = report["first_differences"][0]["first"][0]
    assert (first["token"], first["byte_in_component"]) == (4096, 1)
    rpc.calls.clear()
    second = client.analyze_remote(args, rpc)
    assert second["raw_cached_bytes_reused"] == 2 * 4096 * 2
    assert not any(action in ("checksums", "blocks") for _, action in rpc.calls)
    assert second["different_bytes"] == 1


def test_same_host_compares_in_place_even_with_zero_download_budget(tmp_path):
    args, rpc, after = setup(tmp_path, same_host=True)
    args.max_diff_mib = 0
    corrupt(after / "host-tp0/data.bin", 9999 * 2)
    report = client.analyze_remote(args, rpc)
    assert report["different_bytes"] == 1
    assert report["difference_counts_complete"]
    assert report["raw_cross_host_bytes_selected"] == 0
    assert any(action == "compare" for _, action in rpc.calls)
    assert not any(action == "blocks" for _, action in rpc.calls)


@pytest.mark.parametrize("budget", [0, 1])
def test_widespread_difference_cannot_bypass_download_budget(tmp_path, budget):
    args, rpc, after = setup(tmp_path, count=600000)
    args.max_diff_mib = budget
    (after / "host-tp0/data.bin").write_bytes(b"X" * (600003 * 2))
    report = client.analyze_remote(args, rpc)
    assert report["status"] == "history_differs"
    assert report["raw_cross_host_bytes_selected"] <= budget * 1024 * 1024
    assert report["different_segments"] == 147
    assert report["unfetched_differing_segments"] > 0
    assert not report["difference_counts_complete"]
    assert report["different_bytes"] is None
    assert report["first_differing_segment"]["start"] == 0


def test_token_mismatch_limits_checksums_and_does_not_flag_unshared_suffix(tmp_path):
    args, rpc, after = setup(tmp_path)
    tokens = list(range(10003))
    tokens[4097] = -1
    (after / "host-tp0/tokens.json").write_text(json.dumps(tokens))
    corrupt(after / "host-tp0/data.bin", 4097 * 2)
    report = client.analyze_remote(args, rpc)
    assert report["common_prefix_tokens"] == 4097
    assert report["status"] == "history_checksums_match"
    assert not report["previous_is_prefix"]


def test_missing_rank_does_not_start_expensive_scan(tmp_path):
    args, rpc, after = setup(tmp_path, size=2)
    (after / "host-tp1/manifest.json").unlink()
    with pytest.raises(ValueError, match="Incomplete TP"):
        client.analyze_remote(args, rpc)
    assert all(action == "inventory" for _, action in rpc.calls)


@pytest.mark.parametrize("change", ["writing", "truncate"])
def test_invalid_capture_is_never_a_checksum_match(tmp_path, change):
    args, rpc, after = setup(tmp_path)
    if change == "writing":
        path = after / "host-tp0/manifest.json"
        meta = json.loads(path.read_text())
        meta["status"] = "writing"
        path.write_text(json.dumps(meta))
    else:
        (after / "host-tp0/data.bin").write_bytes(b"bad")
    with pytest.raises(ValueError, match="not complete|Truncated"):
        client.analyze_remote(args, rpc)


def test_source_change_after_scan_detected_even_when_checksums_match(tmp_path):
    args, rpc, after = setup(tmp_path)
    original = rpc.call

    def call(host, payload):
        if host == "q" and payload["action"] == "verify":
            corrupt(after / "host-tp0/data.bin", 0)
        return original(host, payload)

    rpc.call = call
    with pytest.raises(ValueError, match="changed during"):
        client.analyze_remote(args, rpc)


def test_source_change_after_scan_prevents_fetching_other_bytes(tmp_path):
    args, rpc, after = setup(tmp_path)
    corrupt(after / "host-tp0/data.bin", 0)
    original = rpc.call

    def call(host, payload):
        if host == "q" and payload["action"] == "blocks":
            corrupt(after / "host-tp0/data.bin", 1)
        return original(host, payload)

    rpc.call = call
    with pytest.raises(ValueError, match="changed during"):
        client.analyze_remote(args, rpc)


def test_actual_compressed_rpc_works_with_python39_syntax(tmp_path):
    args, _, _ = setup(tmp_path, count=3)
    source = Path(worker.__file__).read_bytes()
    ast.parse(source, feature_version=(3, 9))
    payload = dict(action="inventory", workspace=str(tmp_path / "host-a"), requests=["before"])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            client.BOOTSTRAP,
            base64.b64encode(gzip.compress(source)).decode(),
            base64.b64encode(client.pack(payload)).decode(),
        ],
        capture_output=True,
        check=True,
    )
    inventory = json.loads(gzip.decompress(result.stdout))
    assert len(inventory) == 1
    assert inventory[0]["tokens"] == [0, 1, 2]
    assert inventory[0]["meta"]["files"][0]["bytes"] == 6


def test_cli_defaults_to_remote_checksums_not_full_download(tmp_path, monkeypatch):
    args, rpc, _ = setup(tmp_path)
    monkeypatch.setattr(client, "RemoteClient", lambda ignored: rpc)
    monkeypatch.setattr(cli, "collect", lambda *a: pytest.fail("must not collect full archives"))
    assert (
        cli.main(["--previous", "before", "--current", "after", "--hosts", "p", "q", "--output", str(args.output)]) == 0
    )
    report = json.loads((args.output / "report.json").read_text())
    assert report["mode"] == "offline_segments"
    assert report["raw_cross_host_bytes_selected"] == 0


def test_raw_fallback_without_numpy_keeps_bit_positions(monkeypatch):
    original = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("bare SSH host")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    result = worker.byte_difference(b"abcdeF", b"abCdef", 4096, 2)
    assert result["different_bytes"] == result["different_tokens"] == 2
    assert [(x["token"], x["byte_in_component"]) for x in result["first"]] == [(4097, 0), (4098, 1)]


def test_c8_key_and_scale_are_separate_components(tmp_path):
    args, rpc, after = setup(tmp_path, count=4097)
    for host, request, count in [("host-a", "before", 4097), ("host-b", "after", 4100)]:
        directory = tmp_path / host / "decode-kv-snapshots" / request / "host-tp0"
        path = directory / "manifest.json"
        meta = json.loads(path.read_text())
        meta["groups"] = [1, 1]
        for component, width, dtype in [(0, 128, "torch.int8"), (1, 2, "torch.float16")]:
            name = f"indexer-{component}.bin"
            data = bytes(i % 251 for i in range(count * width))
            (directory / name).write_bytes(data)
            meta["files"].append(
                dict(
                    group=1,
                    layer=0,
                    component=component,
                    path=name,
                    row_bytes=width,
                    dtype=dtype,
                    start=0,
                    end=count,
                    bytes=len(data),
                )
            )
        path.write_text(json.dumps(meta))
    corrupt(after / "host-tp0/indexer-1.bin", 4096 * 2 + 1)
    report = client.analyze_remote(args, rpc)
    assert report["different_segments"] == 1
    first = report["first_differences"][0]
    assert (first["group"], first["layer"], first["component"]) == (1, 0, 1)
    assert first["first"][0]["token"] == 4096
    assert report["raw_cross_host_bytes_selected"] == 4


def test_changed_files_invalidate_cached_hashes(tmp_path):
    args, rpc, after = setup(tmp_path)
    assert client.analyze_remote(args, rpc)["status"] == "history_checksums_match"
    rpc.calls.clear()
    corrupt(after / "host-tp0/data.bin", 42)
    report = client.analyze_remote(args, rpc)
    assert report["different_bytes"] == 1
    assert sum(action == "checksums" for _, action in rpc.calls) == 1


def test_full_transfer_remains_explicit_opt_in(tmp_path, monkeypatch):
    args, rpc, _ = setup(tmp_path)
    calls = []

    def collect(ignored, host):
        calls.append(host)
        return rpc.roots[host]

    monkeypatch.setattr(cli, "collect", collect)
    assert (
        cli.main(
            [
                "--previous",
                "before",
                "--current",
                "after",
                "--hosts",
                "p",
                "q",
                "--full-transfer",
                "--output",
                str(args.output),
            ]
        )
        == 0
    )
    assert sorted(calls) == ["p", "q"]
    assert json.loads((args.output / "report.json").read_text())["status"] == "received_history_identical"


def test_incompatible_layout_fails_before_reading_payloads(tmp_path):
    args, rpc, after = setup(tmp_path)
    path = after / "host-tp0/manifest.json"
    meta = json.loads(path.read_text())
    meta["files"][0]["dtype"] = "torch.bfloat16"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="incompatible layout"):
        client.analyze_remote(args, rpc)
    assert all(action == "inventory" for _, action in rpc.calls)
