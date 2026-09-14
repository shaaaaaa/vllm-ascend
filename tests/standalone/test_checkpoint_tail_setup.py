# SPDX-License-Identifier: Apache-2.0
"""Checkpoint tail alignment is negotiated once at scheduler construction."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def scheduler(enabled, alignment):
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/core/recompute_scheduler.py"
    cls = next(
        n
        for n in ast.parse(path.read_text(encoding="utf8")).body
        if isinstance(n, ast.ClassDef) and n.name == "RecomputeScheduler"
    )
    cls.body = [n for n in cls.body if getattr(n, "name", None) == "__init__"]

    class Latent:
        block_size = 128
        checkpoint_tail_alignment = 0

    class Connector:
        supports_preemption_checkpoint = enabled

        @property
        def preemption_checkpoint_chunk_size(self):
            assert enabled, "disabled scheduler accessed checkpoint config"
            return alignment

    class Base:
        def __init__(self):
            self.connector = Connector()
            self.scheduler_config = NS(mc2_recovery_token_budget=None)
            self.max_num_scheduled_tokens = 32
            self.kv_cache_manager = NS(coordinator=NS(single_type_managers=[Latent(), NS()]))
            self.vllm_config = NS(
                speculative_config=None, kv_transfer_config=None, model_config=NS(hf_text_config=NS(model_type="glm"))
            )

    ns = dict(Scheduler=Base, DSALatentManager=Latent, register_ascend_mla_spec_in_manager=lambda: None)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), ns)
    return ns[cls.name]()


@pytest.mark.parametrize("enabled", [False, True])
def test_only_checkpoint_latent_manager_retains_the_aligned_tail(enabled):
    obj = scheduler(enabled, 1024)
    latent, indexer = obj.kv_cache_manager.coordinator.single_type_managers
    assert latent.checkpoint_tail_alignment == (1024 if enabled else 0)
    assert not hasattr(indexer, "checkpoint_tail_alignment")


@pytest.mark.parametrize("alignment", [0, -128, 1000, None])
def test_incompatible_chunk_size_fails_before_scheduling(alignment):
    with pytest.raises(ValueError, match="align to latent blocks"):
        scheduler(True, alignment)


@pytest.mark.parametrize("sizes", [(1024,), (1024, 1024), (1024, 2048)])
def test_multi_connector_negotiates_only_checkpoint_children(sizes):
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py"
    node = next(
        n
        for n in ast.walk(ast.parse(path.read_text(encoding="utf8")))
        if isinstance(n, ast.FunctionDef) and n.name == "preemption_checkpoint_chunk_size"
    )
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    obj = NS(
        _connectors=[NS(supports_preemption_checkpoint=True, preemption_checkpoint_chunk_size=size) for size in sizes]
    )
    obj._connectors.append(NS(supports_preemption_checkpoint=False))
    getter = ns[node.name].fget
    if len(set(sizes)) == 1:
        assert getter(obj) == sizes[0]
    else:
        with pytest.raises(ValueError, match="agree on chunk size"):
            getter(obj)
