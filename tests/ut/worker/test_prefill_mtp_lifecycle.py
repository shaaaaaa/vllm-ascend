# SPDX-License-Identifier: Apache-2.0
"""CPU execution of production connector lifecycle around the MTP forward."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def runner_api(connector):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    names = {"maybe_get_kv_connector_output", "finalize_kv_connector"}
    cls.body = [n for n in cls.body if getattr(n, "name", None) in names]
    cls.bases = []
    ns = dict(
        contextmanager=contextmanager,
        has_kv_transfer_group=lambda: True,
        get_kv_transfer_group=lambda: connector,
        get_forward_context=lambda: None,
        KVConnectorBase=type(connector),
        KVConnectorOutput=NS,
    )
    unit = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), ns)
    return ns["NPUModelRunner"]


class Connector:
    def __init__(self):
        self.bound = False
        self.draft_finished = False
        self.waits = 0

    def bind_connector_metadata(self, metadata):
        self.bound = True

    def clear_connector_metadata(self):
        self.bound = False

    def start_load_kv(self, context):
        assert self.bound

    def wait_for_save(self):
        assert self.bound and self.draft_finished, "handoff before MTP KV finished"
        self.waits += 1

    def get_finished(self, ids):
        return set(), set()

    def get_block_ids_with_load_errors(self):
        return set()

    def get_kv_connector_stats(self):
        return None

    def get_kv_connector_kv_cache_events(self):
        return None

    def build_connector_worker_meta(self):
        assert self.draft_finished
        return "target-and-draft-persisted"


def test_production_lifecycle_keeps_bank_generators_alive_until_mtp_finished():
    connector = Connector()
    runner = runner_api(connector)
    scheduler = NS(kv_connector_metadata=object(), finished_req_ids=set())
    with runner.maybe_get_kv_connector_output(scheduler, defer_finalize=True):
        assert connector.bound  # target layers run here
    assert connector.bound and connector.waits == 0
    connector.draft_finished = True  # draft layer writes both KV groups
    output = runner.finalize_kv_connector(set())
    assert output.kv_connector_worker_meta == "target-and-draft-persisted"
    assert connector.waits == 1 and not connector.bound


def test_failed_mtp_cannot_publish_successful_handoff():
    connector = Connector()
    runner = runner_api(connector)
    connector.bound = True
    with pytest.raises(AssertionError, match="before MTP"):
        runner.finalize_kv_connector(set())
    assert not connector.bound and connector.waits == 0
