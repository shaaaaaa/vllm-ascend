# SPDX-License-Identifier: Apache-2.0
"""CPU-only collection tests; SSH is mocked and archives contain inert bytes."""

import gzip
import importlib.util
import io
import json
import os
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
    compressed = gzip.compress(tar_bytes(entries), compresslevel=1)
    stub_ssh(monkeypatch, compressed, commands=commands)
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
        assert host["attempts"][0]["archive_bytes"] == len(compressed)
    assert not list((tmp_path / "collected" / "partials").glob("*.gz.partial"))
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
        "python3",
        "-c",
        collector.REMOTE_COLLECT,
        repo,
        "check-001",
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


def test_idle_hosts_are_empty_and_retried_without_redownloading_success(tmp_path, monkeypatch):
    calls = []
    ready = False

    def run(command, *, stdout, stderr, check):
        host = command[-2]
        calls.append(host)
        if host == "absent" and not ready:
            stderr.write(collector.MISSING_RUN_MARKER.encode())
            return subprocess.CompletedProcess(command, collector.MISSING_RUN_EXIT)
        stdout.write(tar_bytes([] if host == "empty" and not ready else [("x.pt", b"data")]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(collector.subprocess, "run", run)
    hosts = ("p-active", "empty", "d-active", "absent")
    assert collector.main(argv(tmp_path, hosts)) == 0
    manifest = read_manifest(tmp_path)
    assert manifest["complete"]
    assert [h["status"] for h in manifest["hosts"]] == ["success", "empty", "success", "empty"]
    ready = True
    assert collector.main(argv(tmp_path, hosts)) == 0
    assert calls == [*hosts, "empty", "absent"]
    assert all(h["status"] == "success" for h in read_manifest(tmp_path)["hosts"])


@pytest.mark.parametrize(
    "code,error,payload",
    [(44, b"docker failed", b""), (255, b"PD_TENSOR_RUN_ABSENT", b""), (44, b"PD_TENSOR_RUN_ABSENT", b"partial")],
)
def test_missing_run_requires_unambiguous_remote_result(tmp_path, monkeypatch, code, error, payload):
    stub_ssh(monkeypatch, payload, returncode=code, error=error)
    assert collector.main(argv(tmp_path)) == 1
    assert read_manifest(tmp_path)["hosts"][0]["status"] == "failed"


def test_all_hosts_absent_is_not_success_and_does_not_analyze(tmp_path, monkeypatch):
    stub_ssh(monkeypatch, b"", returncode=44, error=collector.MISSING_RUN_MARKER.encode())
    monkeypatch.setitem(sys.modules, "pd_tensor_analyze", None)
    assert collector.main([*argv(tmp_path, ("p1", "p2", "d1", "d2")), "--analyze-pd"]) == 1
    assert not read_manifest(tmp_path)["complete"]
    assert not read_manifest(tmp_path)["has_data"]


@pytest.mark.parametrize("mode", ["missing_repo", "missing_root", "missing_run", "not_directory"])
def test_real_remote_probe_distinguishes_missing_run_from_invalid_repository(tmp_path, mode):
    repo = tmp_path / "repository"
    if mode != "missing_repo":
        repo.mkdir()
    if mode in ("missing_run", "not_directory"):
        (repo / "pd-tensor-dump").mkdir()
    if mode == "not_directory":
        (repo / "pd-tensor-dump/run").write_text("invalid")
    result = subprocess.run([sys.executable, "-c", collector.REMOTE_COLLECT, str(repo), "run"], capture_output=True)
    if mode in ("missing_root", "missing_run"):
        assert result.returncode == collector.MISSING_RUN_EXIT
        assert result.stderr.decode().strip() == collector.MISSING_RUN_MARKER
        assert not result.stdout
    else:
        assert result.returncode != 0
        assert collector.MISSING_RUN_MARKER.encode() not in result.stderr


def test_real_remote_probe_sends_gzip_into_safe_extractor_and_cleans_temporary_archive(tmp_path):
    if collector.shutil.which("tar") is None:
        pytest.skip("tar unavailable")
    repo = tmp_path / "repo with spaces"
    run = repo / "pd-tensor-dump/run/P/request/worker"
    run.mkdir(parents=True)
    (run / "tensor.pt").write_bytes(b"inert payload")
    result = subprocess.run([sys.executable, "-c", collector.REMOTE_COLLECT, str(repo), "run"], capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(b"\x1f\x8b")
    metadata = json.loads(result.stderr.decode().split(collector.REMOTE_ARCHIVE_MARKER)[1])
    assert metadata["archive_bytes"] == len(result.stdout)
    assert metadata["compression"] in ("pigz", "python-gzip")
    assert metadata["pack_seconds"] >= 0
    assert not list(repo.glob(".pd-tensor-collect-*"))
    assert (run / "tensor.pt").read_bytes() == b"inert payload"
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(result.stdout)
    destination = tmp_path / "extract"
    destination.mkdir()
    counts = collector.safe_extract(archive, destination)
    assert counts == {"files": 1, "bytes": len(b"inert payload")}
    assert (destination / "P/request/worker/tensor.pt").read_bytes() == b"inert payload"


@pytest.mark.parametrize("damage", ["trailer_missing", "crc", "length", "truncated_payload"])
def test_compressed_archive_damage_is_rejected_before_publication(tmp_path, monkeypatch, damage):
    compressed = bytearray(gzip.compress(tar_bytes([("tensor.pt", b"data" * 20000)])))
    if damage == "trailer_missing":
        compressed = compressed[:-8]
    elif damage == "crc":
        compressed[-8] ^= 1
    elif damage == "length":
        compressed[-1] ^= 1
    else:
        compressed = compressed[: len(compressed) // 2]
    stub_ssh(monkeypatch, compressed)
    assert collector.main(argv(tmp_path)) == 1
    host = read_manifest(tmp_path)["hosts"][0]
    assert host["status"] == "failed"
    assert not (tmp_path / "collected" / host["directory"]).exists()
    assert (tmp_path / "collected" / host["attempts"][0]["archive_path"]).read_bytes() == compressed


def test_remote_archive_size_and_timings_are_recorded(tmp_path, monkeypatch, capsys):
    compressed = gzip.compress(tar_bytes([("tensor.pt", b"data" * 20000)]), compresslevel=1)
    metadata = {"compression": "pigz", "archive_bytes": len(compressed), "pack_seconds": 0.1}
    stub_ssh(monkeypatch, compressed, error=(collector.REMOTE_ARCHIVE_MARKER + json.dumps(metadata)).encode())
    assert collector.main(argv(tmp_path)) == 0
    attempt = read_manifest(tmp_path)["hosts"][0]["attempts"][0]
    assert attempt["remote_archive"] == metadata
    assert attempt["pack_download_seconds"] >= 0
    assert attempt["extract_seconds"] >= 0
    assert "compressor=pigz pack_s=0.1" in capsys.readouterr().out


def test_remote_archive_size_mismatch_is_not_published(tmp_path, monkeypatch):
    compressed = gzip.compress(tar_bytes([("tensor.pt", b"data")]))
    metadata = {"compression": "pigz", "archive_bytes": len(compressed) + 1, "pack_seconds": 0.1}
    stub_ssh(monkeypatch, compressed, error=(collector.REMOTE_ARCHIVE_MARKER + json.dumps(metadata)).encode())
    assert collector.main(argv(tmp_path)) == 1
    host = read_manifest(tmp_path)["hosts"][0]
    assert "size differs" in host["error"]
    assert not (tmp_path / "collected" / host["directory"]).exists()


@pytest.mark.parametrize("failure", ["missing_tar", "tar_failed"])
def test_remote_pack_failure_sends_nothing_and_cleans_temporary_archive(tmp_path, failure):
    repo = tmp_path / "repository"
    run = repo / "pd-tensor-dump/run"
    run.mkdir(parents=True)
    (run / "tensor.pt").write_bytes(b"keep")
    # Windows also searches System32 even without PATH. Inject a failed tar
    # startup/exit into the child process so both failure cases are portable.
    prelude = f"""
import subprocess, sys
real_popen = subprocess.Popen
def failed_tar(*args, **kwargs):
    if {failure!r} == 'missing_tar':
        raise FileNotFoundError('tar unavailable')
    return real_popen([sys.executable, '-c', 'import sys; sys.exit(7)'], **kwargs)
subprocess.Popen = failed_tar
"""
    result = subprocess.run(
        [sys.executable, "-c", prelude + collector.REMOTE_COLLECT, str(repo), "run"],
        capture_output=True,
        cwd=tmp_path,
        timeout=30,
    )
    assert result.returncode != 0
    assert not result.stdout
    assert not list(repo.glob(".pd-tensor-collect-*"))
    assert (run / "tensor.pt").read_bytes() == b"keep"


@pytest.mark.parametrize("use_pigz", [False, True])
@pytest.mark.parametrize("broken_transfer", [False, True])
def test_remote_finishes_archive_before_transfer_and_cleans_after_disconnect(
    tmp_path, monkeypatch, use_pigz, broken_transfer
):
    if collector.shutil.which("tar") is None:
        pytest.skip("tar unavailable")
    repo = tmp_path / "repository"
    run = repo / "pd-tensor-dump/run"
    run.mkdir(parents=True)
    payload = b"tensor bytes" * 10000
    (run / "tensor.pt").write_bytes(payload)

    class Receiver(io.BytesIO):
        def write(self, data):
            # Inspect the remote temporary file at the first transmitted byte.
            archives = list(repo.glob(".pd-tensor-collect-*/run.tar.gz"))
            assert len(archives) == 1
            assert payload in gzip.decompress(archives[0].read_bytes())
            if broken_transfer:
                raise BrokenPipeError("SSH disconnected")
            return super().write(data)

    def pigz(command, *, stdin, stdout, check):
        assert command == ["pigz", "-1", "-p", str(collector.COMPRESSION_THREADS)]
        assert check is True
        stdout.write(gzip.compress(stdin.read(), compresslevel=1))
        return subprocess.CompletedProcess(command, 0)

    receiver = Receiver()
    monkeypatch.setattr(sys, "argv", ["remote", str(repo), "run"])
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=receiver))
    monkeypatch.setattr(collector.shutil, "which", lambda _: "pigz" if use_pigz else None)
    monkeypatch.setattr(collector.subprocess, "run", pigz)
    if broken_transfer:
        with pytest.raises(BrokenPipeError):
            exec(collector.REMOTE_COLLECT, {})
    else:
        exec(collector.REMOTE_COLLECT, {})
        assert payload in gzip.decompress(receiver.getvalue())
    assert not list(repo.glob(".pd-tensor-collect-*"))
    assert (run / "tensor.pt").read_bytes() == payload


@pytest.mark.parametrize("value", [None, ""])
def test_password_env_missing_or_empty_prevents_collection(tmp_path, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TEST_PD_PASSWORD", raising=False)
    else:
        monkeypatch.setenv("TEST_PD_PASSWORD", value)
    assert collector.main([*argv(tmp_path), "--password-env", "TEST_PD_PASSWORD"]) == 1
    assert not (tmp_path / "collected").exists()


@pytest.mark.parametrize("mode", ["prompt", "environment", "per_host"])
@pytest.mark.parametrize("skip_host_key_check", [False, True])
def test_password_auth_not_in_command_output_or_manifest(tmp_path, monkeypatch, capsys, mode, skip_host_key_check):
    secret = "test-only-secret-with-$-and-quotes'"
    # No sshpass lookup, import or package installation may be required.
    monkeypatch.setattr(collector.shutil, "which", lambda _: pytest.fail("unexpected dependency lookup"))
    prompts = []

    def prompt(message):
        prompts.append(message)
        return secret

    monkeypatch.setattr(collector.getpass, "getpass", prompt)
    monkeypatch.setenv("TEST_PD_PASSWORD", secret)
    flags = {
        "prompt": ["--password"],
        "environment": ["--password-env", "TEST_PD_PASSWORD"],
        "per_host": ["--password-per-host"],
    }[mode]
    if skip_host_key_check:
        flags.append("--skip-host-key-check")
    commands = []
    helpers = []
    parent_env = dict(os.environ)

    def run(command, *, stdout, stderr, check, env, stdin, **kwargs):
        commands.append(command)
        assert command[0] == "ssh"
        assert "BatchMode=no" in command
        assert [item for item in command if item.startswith("StrictHostKeyChecking=")] == [
            "StrictHostKeyChecking=no" if skip_host_key_check else "StrictHostKeyChecking=yes"
        ]
        if skip_host_key_check:
            assert f"UserKnownHostsFile={os.devnull}" in command
            assert f"GlobalKnownHostsFile={os.devnull}" in command
        assert "PreferredAuthentications=password,keyboard-interactive" in command
        assert secret not in " ".join(command)
        assert env[collector.ASKPASS_PASSWORD_ENV] == secret
        assert env["SSH_ASKPASS_REQUIRE"] == "force"
        assert env["DISPLAY"]
        assert stdin == subprocess.DEVNULL
        assert kwargs == (
            {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        )
        helper = Path(env["SSH_ASKPASS"])
        helpers.append(helper)
        assert helper.is_file()
        assert secret not in helper.read_text(encoding="utf-8")
        if os.name != "nt":
            assert helper.stat().st_mode & 0o777 == 0o700
            assert helper.parent.stat().st_mode & 0o777 == 0o700
        stdout.write(tar_bytes([("x.pt", b"x")]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(collector.subprocess, "run", run)
    assert collector.main([*argv(tmp_path, ("one", "two")), *flags]) == 0
    assert len(commands) == 2
    assert len(prompts) == {"prompt": 1, "environment": 0, "per_host": 2}[mode]
    assert secret not in json.dumps(read_manifest(tmp_path))
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert dict(os.environ) == parent_env
    assert not any(path.exists() or path.parent.exists() for path in helpers)


@pytest.mark.parametrize("password", ["line\nline", "line\rline", "nul\0byte"])
def test_password_line_protocol_rejects_invalid_values_before_transport(tmp_path, monkeypatch, password):
    monkeypatch.setattr(collector.getpass, "getpass", lambda _: password)
    assert collector.main([*argv(tmp_path), "--password"]) == 1
    assert not (tmp_path / "collected").exists()


@pytest.mark.parametrize("error", [RuntimeError("test failure"), KeyboardInterrupt()])
def test_password_state_is_cleared_after_error(tmp_path, monkeypatch, error):
    monkeypatch.setenv("TEST_PD_PASSWORD", "fake-secret")
    args = collector.parser().parse_args([*argv(tmp_path), "--password-env", "TEST_PD_PASSWORD"])
    with pytest.raises(type(error)), collector.password_auth(args):
        assert args._passwords
        helper = Path(args._askpass)
        assert helper.is_file()
        raise error
    assert args._passwords == {}
    assert args._askpass is None
    assert not helper.parent.exists()


def test_each_host_receives_only_its_own_password(tmp_path, monkeypatch):
    secrets = iter(["password-one", "password-two"])
    monkeypatch.setattr(collector.getpass, "getpass", lambda _: next(secrets))
    received = []

    def run(command, *, env, **kwargs):
        received.append((command[-2], env[collector.ASKPASS_PASSWORD_ENV]))
        kwargs["stdout"].write(tar_bytes([("x", b"x")]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(collector.subprocess, "run", run)
    assert collector.main([*argv(tmp_path, ("one", "two")), "--password-per-host"]) == 0
    assert received == [("one", "password-one"), ("two", "password-two")]


def test_password_free_mode_never_creates_helper(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "write_askpass", lambda *a, **kw: pytest.fail("unexpected askpass helper"))
    stub_ssh(monkeypatch, tar_bytes([("x", b"x")]))
    assert collector.main(argv(tmp_path)) == 0


@pytest.mark.parametrize("correct_password", [True, False])
def test_real_openssh_reads_password_via_native_callback(tmp_path, monkeypatch, correct_password):
    """Exercise the actual OpenSSH callback without a server or user credentials."""
    keygen = collector.shutil.which("ssh-keygen")
    if keygen is None:
        pytest.skip("OpenSSH ssh-keygen unavailable")
    secret = "test-only-'\"$%&-Unicode-密码"
    key = tmp_path / "throwaway-key"
    result = subprocess.run(
        [keygen, "-q", "-t", "ed25519", "-N", secret, "-f", str(key)],
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    monkeypatch.setenv("TEST_PD_PASSWORD", secret if correct_password else "incorrect-test-password")
    args = collector.parser().parse_args([*argv(tmp_path), "--password-env", "TEST_PD_PASSWORD"])
    with collector.password_auth(args):
        result = collector.run_ssh(
            args,
            args.hosts[0],
            [keygen, "-y", "-f", str(key)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        helper = Path(args._askpass)
        assert secret not in helper.read_text(encoding="utf-8")
    if correct_password:
        assert result.returncode == 0, result.stderr
        assert result.stdout.split()[:2] == key.with_suffix(".pub").read_bytes().split()[:2]
    else:
        assert result.returncode != 0
        assert result.stdout == b""
    assert secret.encode("utf-8") not in result.stdout + result.stderr
    assert not helper.parent.exists()


def test_posix_ssh_detaches_tty_and_overrides_only_child_askpass_environment(monkeypatch):
    parent_env = {"DISPLAY": "", "SSH_ASKPASS": "/old-helper", "SSH_ASKPASS_REQUIRE": "never"}
    monkeypatch.setattr(collector, "os", SimpleNamespace(name="posix", environ=parent_env))
    args = SimpleNamespace(_passwords={"host": "test-password"}, _askpass="/temporary-helper")

    def run(command, *, check, stdin, env, start_new_session):
        assert command == ["ssh", "host"]
        assert start_new_session is True
        assert stdin == subprocess.DEVNULL
        assert env["DISPLAY"] == "pd-tensor:0"
        assert env["SSH_ASKPASS"] == "/temporary-helper"
        assert env["SSH_ASKPASS_REQUIRE"] == "force"
        assert env[collector.ASKPASS_PASSWORD_ENV] == "test-password"
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(collector.subprocess, "run", run)
    assert collector.run_ssh(args, "host", ["ssh", "host"]).returncode == 0
    assert parent_env == {"DISPLAY": "", "SSH_ASKPASS": "/old-helper", "SSH_ASKPASS_REQUIRE": "never"}


@pytest.mark.parametrize("hint", [None, "confirm", "none"])
def test_posix_callback_returns_literal_password_and_rejects_nonpassword_prompts(tmp_path, hint):
    shell = collector.shutil.which("sh")
    if shell is None and os.name == "nt":
        git = collector.shutil.which("git")
        candidate = Path(git).resolve().parent.parent / "bin/bash.exe" if git else None
        shell = str(candidate) if candidate and candidate.is_file() else None
    if shell is None:
        pytest.skip("POSIX shell unavailable")
    helper = collector.write_askpass(tmp_path, windows=False)
    secret = "space ' quote \" $() `command` %PATH% & back\\slash 密码"
    env = dict(os.environ, **{collector.ASKPASS_PASSWORD_ENV: secret})
    env.pop("SSH_ASKPASS_PROMPT", None)
    if hint is not None:
        env["SSH_ASKPASS_PROMPT"] = hint
    result = subprocess.run([shell, str(helper), "password:"], env=env, capture_output=True, timeout=15)
    if hint is None:
        assert result.returncode == 0, result.stderr
        assert result.stdout == secret.encode("utf-8") + b"\n"
    else:
        assert result.returncode == 1
        assert result.stdout == b""


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
@pytest.mark.parametrize("workers", [1, 4, 16])
def test_analyze_pd_runs_after_complete_collection_with_same_roots(
    tmp_path, monkeypatch, capsys, report_status, expected_exit, workers
):
    stub_ssh(monkeypatch, tar_bytes([("P/request/worker/value.pt", b"complete")]))
    calls = []

    def analyze(reference, candidate, *, mode, output, workers):
        manifest = read_manifest(tmp_path)
        assert manifest["complete"] is True
        calls.append((reference, candidate, mode, output, workers))
        output.mkdir()
        report = {"status": report_status, "compared_tensors": 17, "counts": {"different": 1}, "issues": 0}
        (output / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setitem(sys.modules, "pd_tensor_analyze", SimpleNamespace(analyze=analyze))
    flags = [] if workers == 16 else ["--analysis-workers", str(workers)]
    assert collector.main([*argv(tmp_path), "--analyze-pd", *flags]) == expected_exit
    root = (tmp_path / "collected").resolve()
    assert calls == [([root], [root], "pd-kv", root / "report-pd", workers)]
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


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_analysis_workers_is_rejected_before_ssh(tmp_path, monkeypatch, workers):
    commands = []
    stub_ssh(monkeypatch, b"", commands=commands)
    assert collector.main([*argv(tmp_path), "--analyze-pd", "--analysis-workers", str(workers)]) == 1
    assert not commands
    assert not (tmp_path / "collected").exists()


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
