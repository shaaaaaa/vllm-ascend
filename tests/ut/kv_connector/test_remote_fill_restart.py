# SPDX-License-Identifier: Apache-2.0
"""Tests for the deployment-owned RemoteFill paired-restart contract."""

# Standard
import importlib.util
from pathlib import Path

# Third Party
import pytest


def _restart_affected_pair():
    path = Path(__file__).parents[3] / "vllm_ascend" / "distributed" / "kv_transfer" / "remote_fill_restart.py"
    spec = importlib.util.spec_from_file_location("remote_fill_restart", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.restart_affected_pair


class _Adapter:
    def __init__(self, *, stale: str | None = None, all_stopped: bool = True):
        self.calls: list[str] = []
        self.old = {
            "source_incarnation_id": "p-old",
            "destination_incarnation_id": "d-old",
            "source_engine_epoch": 1,
            "destination_engine_epoch": 2,
            "source_session": "p:1",
            "destination_session": "d:1",
            "global_te_registration_generation": 2,
            "shared_cache_generation": 3,
            "shared_cache_incarnation_id": "cache-old",
        }
        self.new = {
            "source_incarnation_id": "p-new",
            "destination_incarnation_id": "d-new",
            "source_engine_epoch": 11,
            "destination_engine_epoch": 12,
            "source_session": "p:2",
            "destination_session": "d:2",
            "global_te_registration_generation": 12,
            "shared_cache_generation": 13,
            "shared_cache_incarnation_id": "cache-new",
        }
        if stale is not None:
            self.new[stale] = self.old[stale]
        self.all_stopped = all_stopped

    def current_identity(self, pair_id):
        self.calls.append("current")
        return self.old

    def invalidate_proxy_placement(self, pair_id):
        self.calls.append("invalidate")

    def stop_admission(self, pair_id):
        self.calls.append("stop_admission")

    def terminate_engine_groups(self, pair_id):
        self.calls.append("terminate")

    def wait_engine_groups_stopped(self, pair_id, timeout_seconds):
        self.calls.append("wait_stopped")
        return {
            "source_expected": 16,
            "source_stopped": 16 if self.all_stopped else 15,
            "destination_expected": 16,
            "destination_stopped": 16 if self.all_stopped else 15,
        }

    def start_engine_groups(self, pair_id):
        self.calls.append("start")

    def discover_identity(self, pair_id, timeout_seconds):
        self.calls.append("discover")
        return self.new

    def publish_proxy_placement(self, pair_id, identity):
        self.calls.append("publish")

    def restore_admission(self, pair_id):
        self.calls.append("restore")


def test_paired_restart_orders_full_stop_before_fresh_publication() -> None:
    adapter = _Adapter()

    record = _restart_affected_pair()(adapter, "p0-d1", timeout_seconds=30)

    assert adapter.calls == [
        "current",
        "invalidate",
        "stop_admission",
        "terminate",
        "wait_stopped",
        "start",
        "discover",
        "publish",
        "restore",
    ]
    assert record["new_identity"] == adapter.new


@pytest.mark.parametrize(
    "stale",
    (
        "source_engine_epoch",
        "destination_engine_epoch",
        "source_incarnation_id",
        "destination_incarnation_id",
        "global_te_registration_generation",
        "shared_cache_incarnation_id",
    ),
)
def test_paired_restart_never_restores_stale_identity(stale) -> None:
    adapter = _Adapter(stale=stale)

    with pytest.raises(RuntimeError, match="reused identity"):
        _restart_affected_pair()(adapter, "p0-d1", timeout_seconds=30)

    assert "publish" not in adapter.calls
    assert "restore" not in adapter.calls
    assert adapter.calls[-2:] == ["terminate", "wait_stopped"]


def test_paired_restart_never_starts_until_both_groups_stop() -> None:
    adapter = _Adapter(all_stopped=False)

    with pytest.raises(RuntimeError, match="every P and D process"):
        _restart_affected_pair()(adapter, "p0-d1", timeout_seconds=30)

    assert "start" not in adapter.calls
    assert "restore" not in adapter.calls


def test_paired_restart_accepts_zero_initial_shared_cache_generation() -> None:
    adapter = _Adapter()
    adapter.old["shared_cache_generation"] = 0

    record = _restart_affected_pair()(adapter, "p0-d1", timeout_seconds=30)

    assert record["old_identity"]["shared_cache_generation"] == 0


def test_paired_restart_allows_stable_routable_sessions() -> None:
    adapter = _Adapter()
    adapter.new["source_session"] = adapter.old["source_session"]
    adapter.new["destination_session"] = adapter.old["destination_session"]

    record = _restart_affected_pair()(adapter, "p0-d1", timeout_seconds=30)

    assert record["new_identity"]["source_session"] == "p:1"
