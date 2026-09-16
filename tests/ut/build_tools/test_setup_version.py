# SPDX-License-Identifier: Apache-2.0
"""CPU tests for setup.py's SCM version configuration, without NPU build imports."""

import ast
import subprocess
from pathlib import Path

import pytest
from packaging.version import Version
from setuptools_scm import get_version


def _setup_version(repo):
    source = Path(__file__).resolve().parents[3] / "setup.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    call = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "VERSION" for target in node.targets)
        and isinstance(node.value, ast.Call)
    )
    options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
    return get_version(root=str(repo), **options)


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT).strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    # Do not let a developer's version override hide the regression.
    for name in (
        "SETUPTOOLS_SCM_PRETEND_VERSION",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_ASCEND",
        "VCS_VERSIONING_PRETEND_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "Version Test")
    _git(tmp_path, "config", "user.email", "version-test@example.invalid")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    _git(tmp_path, "config", "tag.gpgsign", "false")
    (tmp_path / "vllm_ascend").mkdir()
    (tmp_path / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "initial")
    return tmp_path


@pytest.mark.parametrize("tag", ["v0.13.0", "0.13.0", "v0.13.0rc1", "0.13.0.post1", "v0.13.0.dev1"])
def test_release_tags_preserve_version(repo, tag):
    _git(repo, "tag", tag)
    assert Version(_setup_version(repo)) == Version(tag)
    assert (repo / "vllm_ascend" / "_version.py").is_file()


def test_performance_tag_does_not_mask_release(repo):
    _git(repo, "tag", "v0.13.0")
    _git(repo, "commit", "--allow-empty", "-m", "after release")
    before = _setup_version(repo)
    _git(repo, "tag", "pd-tpot-96.8ms-baseline-20260910")
    _git(repo, "tag", "remote-fill-checkpoint-20260909")
    assert _setup_version(repo) == before
    assert Version(before).is_devrelease


@pytest.mark.parametrize("tag", ["pd-tpot-96.8ms-baseline-20260910", "remote-fill-checkpoint-20260909"])
def test_only_non_release_tags_behave_like_untagged_checkout(repo, tag):
    before = _setup_version(repo)
    _git(repo, "tag", tag)
    assert _setup_version(repo) == before


def test_no_tags_still_produces_valid_version(repo):
    assert Version(_setup_version(repo)).is_devrelease


def test_dirty_checkout_still_has_dirty_version(repo):
    _git(repo, "tag", "v0.13.0")
    clean = _setup_version(repo)
    (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    dirty = _setup_version(repo)
    assert Version(dirty).is_devrelease
    assert dirty != clean


def test_explicit_version_override_still_works(repo, monkeypatch):
    _git(repo, "tag", "pd-tpot-96.8ms-baseline-20260910")
    monkeypatch.setenv("SETUPTOOLS_SCM_PRETEND_VERSION", "0.13.0+test")
    assert _setup_version(repo) == "0.13.0+test"
