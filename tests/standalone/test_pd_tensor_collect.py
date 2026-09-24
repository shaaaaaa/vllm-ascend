# SPDX-License-Identifier: Apache-2.0
"""CPU-only collection tests; SSH is mocked and archives contain inert bytes."""

import importlib.util
import io
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "pd_tensor_collect.py"
SPEC = importlib.util.spec_from_file_location("pd_tensor_collect_test", MODULE_PATH)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def tar_bytes(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for entry, payload in entries:
            member = tarfile.TarInfo(entry) if isinstance(entry, str) else entry
            member.size = len(payload) if member.isfile() else 0
            archive.addfile(member, io.BytesIO(payload) if member.isfile() else None)
    return stream.getvalue()


def argv(tmp_path, hosts=("user@192.0.2.10",), **options):
    result = [
        "--hosts",
        *hosts,
        "--repo-path",
        "/workspace/vllm-ascend",
        "--run-id",
        "check-001",
        "--output",
        str(tmp_path / "collected"),
    ]
    for key, value in options.items():
        result.extend([f"--{key.replace('_', '-')}", str(value)])
    return result


def read_manifest(tmp_path):
    return json.loads((tmp_path / "collected" / "collection.json").read_text(encoding="utf-8"))


def stub_ssh(monkeypatch, payload, *, returncode=0, error=b"", commands=None):
    def run(command, *, stdout, stderr, check):
        assert check is False
        assert command[:4] == ["ssh", "-T", "-o", "BatchMode=yes"]
        if commands is not None:
            commands.append(command)
        stdout.write(payload)
        stderr.write(error)
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(collector.subprocess, "run", run)


def test_collect_four_hosts_preserves_role_request_worker_tree(tmp_path, monkeypatch):
    entries = [
        ("./prefill/request-a/rank0/index.jsonl", b'{"shape":[3]}\n'),
        ("./prefill/request-a/rank0/value.pt", b"inert tensor bytes"),
        ("./decode/request-a/rank1/coverage.json", b'{"complete":true}'),
    ]
    commands = []
    stub_ssh(monkeypatch, tar_bytes(entries), commands=commands)
    hosts = tuple(f"user@192.0.2.{i}" for i in range(10, 14))
    assert collector.main(argv(tmp_path, hosts)) == 0
    manifest = read_manifest(tmp_path)
    assert manifest["complete"] is True
    assert manifest["source_path"] == "/workspace/vllm-ascend/pd-tensor-dump/check-001"
    assert {host["target"] for host in manifest["hosts"]} == set(hosts)
    assert len(commands) == 4
    for host in manifest["hosts"]:
        assert host["status"] == "success"
        assert host["files"] == len(entries)
        assert host["bytes"] == sum(len(payload) for _, payload in entries)
        for relative, payload in entries:
            assert (tmp_path / "collected" / host["directory"] / relative).read_bytes() == payload
        assert "archive_path" not in host["attempts"][0]
    assert not list((tmp_path / "collected" / "partials").glob("*.tar.partial"))
    assert not (tmp_path / "collected" / ".collection.lock").exists()


def test_remote_argv_quotes_paths_and_preserves_ssh_verification(tmp_path):
    repo = "/workspace/a 'quoted'; literal directory"
    key = tmp_path / "private key"
    args = collector.parser().parse_args(
        argv(tmp_path, repo_path=repo, container="inference-1", ssh_port=2222, identity_file=key)
    )
    command = collector.ssh_command(args, args.hosts[0], collector.source_path(repo, args.run_id))
    assert command == [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-p",
        "2222",
        "-i",
        str(key.resolve()),
        "--",
        "user@192.0.2.10",
        command[-1],
    ]
    assert shlex.split(command[-1]) == [
        "docker",
        "exec",
        "--",
        "inference-1",
        "tar",
        "-C",
        repo + "/pd-tensor-dump/check-001",
        "-cf",
        "-",
        "--",
        ".",
    ]
    assert "StrictHostKeyChecking" not in " ".join(command)
    assert "UserKnownHostsFile" not in " ".join(command)
    assert not key.exists()  # Collector does not read, create or persist key material.


@pytest.mark.parametrize(
    "target", ["-oProxyCommand=id", "x\ny", "user@-evil", "@host", "a@b@c", "x y", "x;id", "x$(id)"]
)
def test_reject_unsafe_targets(target):
    with pytest.raises(collector.argparse.ArgumentTypeError):
        collector.ssh_target(target)


@pytest.mark.parametrize(
    "target", ["inference-alias", "192.0.2.1", "user@192.0.2.1", "2001:db8::1", "user@[2001:db8::1]", "[2001:db8::1]"]
)
def test_accept_ssh_targets(target):
    assert collector.ssh_target(target) == target


@pytest.mark.parametrize("name", ["../escape", "a/b", "-flag", ".", "a\nb", "a;id", "trailing."])
def test_run_and_container_are_simple_components(name):
    with pytest.raises(collector.argparse.ArgumentTypeError):
        collector.safe_component(name)


@pytest.mark.parametrize("path", ["relative/path", "/workspace/../other", "/work\nspace", "C:\\work"])
def test_reject_unsafe_repo_path(path):
    with pytest.raises(ValueError):
        collector.source_path(path, "run")


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "/outside",
        "nested/../../outside",
        "C:/outside",
        "a\\outside",
        "NUL.txt",
        "a:ads",
        "trailing. ",
        "a\nb",
    ],
)
def test_reject_archive_path_escape_and_platform_aliases(tmp_path, monkeypatch, name):
    stub_ssh(monkeypatch, tar_bytes([("good/item.pt", b"first"), (name, b"untrusted")]))
    assert collector.main(argv(tmp_path)) == 1
    manifest = read_manifest(tmp_path)
    assert manifest["complete"] is False
    host = manifest["hosts"][0]
    assert host["status"] == "failed"
    assert not (tmp_path / "collected" / host["directory"]).exists()
    attempt = host["attempts"][0]
    assert (tmp_path / "collected" / attempt["archive_path"]).is_file()
    partial = tmp_path / "collected" / attempt["partial_directory"]
    assert (partial / "good/item.pt").read_bytes() == b"first"
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE])
def test_archive_links_and_special_files_rejected(tmp_path, kind):
    member = tarfile.TarInfo("link")
    member.type = kind
    member.linkname = "../outside"
    archive = tmp_path / "payload.tar"
    archive.write_bytes(tar_bytes([(member, b"")]))
    destination = tmp_path / "extracted"
    destination.mkdir()
    with pytest.raises(ValueError, match="links and special"):
        collector.safe_extract(archive, destination)
    assert not list(destination.iterdir())


@pytest.mark.parametrize("names", [("x", "x"), ("x", "X"), ("./x", "x")])
def test_archive_duplicate_files_rejected_without_overwriting(tmp_path, names):
    archive = tmp_path / "payload.tar"
    archive.write_bytes(tar_bytes([(names[0], b"original"), (names[1], b"replacement")]))
    destination = tmp_path / "extracted"
    destination.mkdir()
    with pytest.raises(ValueError, match="duplicate"):
        collector.safe_extract(archive, destination)
    assert (destination / "x").read_bytes() == b"original"


def test_one_ssh_failure_does_not_hide_other_host_success_or_extract_partial_tar(tmp_path, monkeypatch):
    good_tar = tar_bytes([("decode/request/rank0/x.pt", b"payload")])

    def run(command, *, stdout, stderr, check):
        stdout.write(good_tar)
        failed = command[-2] == "bad-host"
        stderr.write(b"permission denied" if failed else b"")
        return subprocess.CompletedProcess(command, 255 if failed else 0)

    monkeypatch.setattr(collector.subprocess, "run", run)
    assert collector.main(argv(tmp_path, ("bad-host", "good-host"))) == 1
    manifest = read_manifest(tmp_path)
    assert manifest["complete"] is False
    bad, good = manifest["hosts"]
    assert [bad["status"], good["status"]] == ["failed", "success"]
    assert "permission denied" in bad["error"]
    assert bad["attempts"][0]["ssh_returncode"] == 255
    assert "partial_directory" not in bad["attempts"][0]
    assert (tmp_path / "collected" / bad["attempts"][0]["archive_path"]).read_bytes() == good_tar


@pytest.mark.parametrize(
    "payload",
    [b"", b"not tar", tar_bytes([]), tar_bytes([("x.pt", b"x" * 2000)])[:1000]],
    ids=["no-bytes", "invalid-header", "no-files", "truncated-payload"],
)
def test_empty_or_corrupt_archive_is_not_success(tmp_path, monkeypatch, payload):
    stub_ssh(monkeypatch, payload)
    assert collector.main(argv(tmp_path)) == 1
    assert read_manifest(tmp_path)["complete"] is False


def test_transport_exception_is_recorded_and_download_retained(tmp_path, monkeypatch):
    def run(*args, **kwargs):
        raise FileNotFoundError("ssh unavailable")

    monkeypatch.setattr(collector.subprocess, "run", run)
    assert collector.main(argv(tmp_path)) == 1
    host = read_manifest(tmp_path)["hosts"][0]
    assert "ssh unavailable" in host["error"]
    assert (tmp_path / "collected" / host["attempts"][0]["archive_path"]).exists()


def test_successful_host_is_skipped_and_failed_host_retries_without_deleting_partial(tmp_path, monkeypatch):
    payload = tar_bytes([("prefill/request/rank0/value.pt", b"original")])
    calls = []
    stub_ssh(monkeypatch, payload, commands=calls)
    assert collector.main(argv(tmp_path, ("good",))) == 0
    stub_ssh(monkeypatch, b"interrupted", returncode=255, commands=calls)
    assert collector.main(argv(tmp_path, ("good", "retry"))) == 1
    failed_attempt = read_manifest(tmp_path)["hosts"][1]["attempts"][0]
    old_partial = tmp_path / "collected" / failed_attempt["archive_path"]
    assert old_partial.read_bytes() == b"interrupted"
    stub_ssh(monkeypatch, tar_bytes([("new.pt", b"new")]), commands=calls)
    assert collector.main(argv(tmp_path, ("good", "retry"))) == 0
    manifest = read_manifest(tmp_path)
    assert len(calls) == 3
    good, retried = manifest["hosts"]
    assert (tmp_path / "collected" / good["directory"] / "prefill/request/rank0/value.pt").read_bytes() == b"original"
    assert len(good["attempts"]) == 1
    assert len(retried["attempts"]) == 2
    assert old_partial.read_bytes() == b"interrupted"


def test_existing_untracked_host_directory_is_never_overwritten(tmp_path, monkeypatch):
    destination = tmp_path / "collected/hosts" / collector.host_directory("user@192.0.2.10")
    destination.mkdir(parents=True)
    (destination / "original").write_text("keep", encoding="utf-8")
    commands = []
    stub_ssh(monkeypatch, tar_bytes([("original", b"replace")]), commands=commands)
    assert collector.main(argv(tmp_path)) == 1
    assert not commands
    assert (destination / "original").read_text(encoding="utf-8") == "keep"
    assert read_manifest(tmp_path)["hosts"][0]["status"] == "failed"


def test_different_run_cannot_reuse_collection_directory(tmp_path, monkeypatch):
    commands = []
    stub_ssh(monkeypatch, tar_bytes([("original", b"keep")]), commands=commands)
    assert collector.main(argv(tmp_path)) == 0
    original_manifest = (tmp_path / "collected/collection.json").read_bytes()
    assert collector.main(argv(tmp_path, run_id="another-run")) == 1
    assert len(commands) == 1
    assert (tmp_path / "collected/collection.json").read_bytes() == original_manifest


def test_failed_unrequested_host_keeps_overall_collection_incomplete(tmp_path, monkeypatch):
    stub_ssh(monkeypatch, b"partial", returncode=255)
    assert collector.main(argv(tmp_path, ("bad",))) == 1
    stub_ssh(monkeypatch, tar_bytes([("x.pt", b"complete")]))
    assert collector.main(argv(tmp_path, ("good",))) == 1
    assert read_manifest(tmp_path)["complete"] is False


def test_missing_previous_success_is_detected_even_if_not_requested(tmp_path, monkeypatch):
    stub_ssh(monkeypatch, tar_bytes([("x.pt", b"complete")]))
    assert collector.main(argv(tmp_path, ("old",))) == 0
    manifest = read_manifest(tmp_path)
    directory = tmp_path / "collected" / manifest["hosts"][0]["directory"]
    directory.rename(directory.with_name(directory.name + "-moved"))
    assert collector.main(argv(tmp_path, ("new",))) == 1
    manifest = read_manifest(tmp_path)
    assert manifest["complete"] is False
    assert manifest["hosts"][0]["status"] == "failed"


def test_keyboard_interrupt_records_failure_and_returns_130(tmp_path, monkeypatch):
    def run(command, *, stdout, **kwargs):
        stdout.write(b"unfinished")
        raise KeyboardInterrupt

    monkeypatch.setattr(collector.subprocess, "run", run)
    assert collector.main(argv(tmp_path, ("first", "second"))) == 130
    manifest = read_manifest(tmp_path)
    assert manifest["complete"] is False
    assert [host["status"] for host in manifest["hosts"]] == ["failed", "pending"]
    assert (tmp_path / "collected" / manifest["hosts"][0]["attempts"][0]["archive_path"]).read_bytes() == b"unfinished"
    assert not (tmp_path / "collected/.collection.lock").exists()


def test_existing_lock_prevents_concurrent_collection(tmp_path, monkeypatch):
    root = tmp_path / "collected"
    root.mkdir()
    (root / ".collection.lock").write_text("existing collector", encoding="utf-8")
    commands = []
    stub_ssh(monkeypatch, tar_bytes([("x", b"x")]), commands=commands)
    assert collector.main(argv(tmp_path)) == 1
    assert not commands
    assert (root / ".collection.lock").read_text(encoding="utf-8") == "existing collector"


@pytest.mark.parametrize(
    ("report_status", "expected_exit"),
    [("analysis_complete", 0), ("incomplete_or_incomparable", 2)],
)
def test_analyze_pd_runs_after_complete_collection_with_same_roots(
    tmp_path, monkeypatch, capsys, report_status, expected_exit
):
    stub_ssh(monkeypatch, tar_bytes([("P/request/worker/value.pt", b"complete")]))
    calls = []

    def analyze(reference, candidate, *, mode, output):
        manifest = read_manifest(tmp_path)
        assert manifest["complete"] is True
        calls.append((reference, candidate, mode, output))
        output.mkdir()
        report = {"status": report_status, "compared_tensors": 17, "counts": {"different": 1}, "issues": 0}
        (output / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setitem(sys.modules, "pd_tensor_analyze", SimpleNamespace(analyze=analyze))
    assert collector.main([*argv(tmp_path), "--analyze-pd"]) == expected_exit
    root = (tmp_path / "collected").resolve()
    assert calls == [([root], [root], "pd-kv", root / "report-pd")]
    lines = capsys.readouterr().out.splitlines()
    summary = json.loads(
        next(line.removeprefix("[PD_TENSOR_ANALYZE] ") for line in lines if "PD_TENSOR_ANALYZE" in line)
    )
    assert summary == {
        "status": report_status,
        "compared_tensors": 17,
        "different": 1,
        "issues": 0,
        "report": str(root / "report-pd/report.json"),
    }


def test_download_failure_does_not_import_analyzer(tmp_path, monkeypatch, capsys):
    stub_ssh(monkeypatch, b"partial", returncode=255)
    monkeypatch.setitem(sys.modules, "pd_tensor_analyze", None)
    assert collector.main([*argv(tmp_path), "--analyze-pd"]) == 1
    assert "PD_TENSOR_ANALYZE" not in capsys.readouterr().err
    assert not (tmp_path / "collected/report-pd").exists()


def test_default_collection_does_not_import_analyzer(tmp_path, monkeypatch):
    stub_ssh(monkeypatch, tar_bytes([("x.pt", b"data")]))
    monkeypatch.setitem(sys.modules, "pd_tensor_analyze", None)
    assert collector.main(argv(tmp_path)) == 0
    assert not (tmp_path / "collected/report-pd").exists()


def test_analysis_import_error_preserves_successful_collection_and_returns_2(tmp_path, monkeypatch, capsys):
    stub_ssh(monkeypatch, tar_bytes([("x.pt", b"data")]))
    monkeypatch.setitem(sys.modules, "pd_tensor_analyze", None)
    assert collector.main([*argv(tmp_path), "--analyze-pd"]) == 2
    assert read_manifest(tmp_path)["complete"] is True
    assert "[PD_TENSOR_ANALYZE] FAILED: ModuleNotFoundError" in capsys.readouterr().err


def test_analysis_can_be_added_to_previous_collection_without_downloading_again(tmp_path, monkeypatch):
    commands = []
    stub_ssh(monkeypatch, tar_bytes([("x.pt", b"data")]), commands=commands)
    assert collector.main(argv(tmp_path)) == 0
    calls = []

    def analyze(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "analysis_complete"}

    monkeypatch.setitem(sys.modules, "pd_tensor_analyze", SimpleNamespace(analyze=analyze))
    assert collector.main([*argv(tmp_path), "--analyze-pd"]) == 0
    assert len(commands) == len(calls) == 1


def test_analysis_error_returns_2_without_changing_successful_collection(tmp_path, monkeypatch, capsys):
    stub_ssh(monkeypatch, tar_bytes([("x.pt", b"data")]))

    def analyze(*args, **kwargs):
        raise ValueError("invalid tensor schema")

    monkeypatch.setitem(sys.modules, "pd_tensor_analyze", SimpleNamespace(analyze=analyze))
    assert collector.main([*argv(tmp_path), "--analyze-pd"]) == 2
    assert read_manifest(tmp_path)["complete"] is True
    assert "invalid tensor schema" in capsys.readouterr().err
