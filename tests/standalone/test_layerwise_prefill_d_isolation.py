# SPDX-License-Identifier: Apache-2.0
"""Exercise production P gates without loading the Ascend runtime."""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2] / "vllm_ascend"


def method(path, cls, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)


def execute(nodes, namespace):
    tree = ast.parse("from __future__ import annotations")
    tree.body.extend(deepcopy(nodes))
    exec(compile(ast.fix_missing_locations(tree), "<production-D-isolation>", "exec"), namespace)


def assigns(node, name):
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    return any(isinstance(target, ast.Name) and target.id == name for target in targets)


def calls(node, name):
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == name for n in ast.walk(node)
    )


@pytest.mark.parametrize("p_node", [False, True])
def test_mlapo_only_p_enters_the_added_bank_wait_wrapper(p_node):
    forward = method("attention/sfa_v1.py", "AscendSFAImpl", "forward")
    outer = next(n for n in forward.body if isinstance(n, ast.If) and "self.enable_mlapo" in ast.unparse(n.test))
    dispatch = outer.body[0]
    seen = []

    def preprocess(**kwargs):
        seen.append(("legacy", kwargs))
        return 1, 2, 3, 4

    obj = NS(_layerwise_prefill_p_node=p_node, _sfa_preprocess_with_mlapo=preprocess)
    if p_node:
        obj._sfa_preprocess_with_mlapo_after_layerwise_wait = lambda **kwargs: (
            seen.append(("bank_wait", kwargs)) or (1, 2, 3, 4)
        )
    ns = dict(
        self=obj,
        hidden_states=object(),
        kv_cache=object(),
        cos=object(),
        sin=object(),
        slot_mapping=object(),
        num_input_tokens=1,
        layer_name="layer",
        attn_metadata=object(),
    )
    execute([dispatch], ns)
    assert [entry[0] for entry in seen] == (["bank_wait"] if p_node else ["legacy"])
    assert ns["ql_nope"] == 2
    assert ("layer_name" in seen[0][1]) is p_node


@pytest.mark.parametrize("p_node", [False, True])
def test_input_preparation_uses_single_legacy_table_on_d(p_node):
    prepare = method("worker/model_runner_v1.py", "NPUModelRunner", "_prepare_inputs")
    nodes = [
        node
        for node in prepare.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.layerwise_prefill_p_node"
        and (calls(node, "commit_block_table") or calls(node, "compute_slot_mapping"))
    ]
    assert len(nodes) == 2
    seen = []

    def table(name):
        return NS(
            commit_block_table=lambda count: seen.append((name, "blocks", count)),
            compute_slot_mapping=lambda requests, positions: seen.append((name, "slots", requests, positions)),
            commit_slot_mapping=lambda count: seen.append((name, "commit", count)),
        )

    primary, shadow = table("primary"), table("shadow")
    obj = NS(layerwise_prefill_p_node=p_node, input_batch=NS(block_table=primary))
    if p_node:
        obj._refresh_layerwise_prefill_block_tables = lambda: (primary, shadow)
    execute(nodes, dict(self=obj, num_reqs=1, req_indices="req", positions_np="pos", total_num_scheduled_tokens=3))
    assert seen == (
        [
            ("primary", "blocks", 1),
            ("shadow", "blocks", 1),
            ("primary", "slots", "req", "pos"),
            ("primary", "commit", 3),
            ("shadow", "slots", "req", "pos"),
            ("shadow", "commit", 3),
        ]
        if p_node
        else [("primary", "blocks", 1), ("primary", "slots", "req", "pos"), ("primary", "commit", 3)]
    )


@pytest.mark.parametrize("p_node", [False, True])
def test_shared_pool_only_p_reuses_layer_tensor_objects(p_node):
    reshape = method("worker/model_runner_v1.py", "NPUModelRunner", "_reshape_kv_cache_tensors")
    gate = next(
        n
        for n in ast.walk(reshape)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "self.dsa_shared_pool and len(raws) == 1"
    )
    function = ast.parse("def reshape(self, layers):\n    for layer_name in layers:\n        pass\n").body[0]
    function.body[0].body = [gate]
    records = []

    def views(*args, is_indexer):
        result = (object(),)
        records.append((is_indexer, result))
        return result

    caches = {}
    ns = dict(
        reshape_dsa_shared_pool_raw=views,
        raws=[object()],
        spec=NS(dtype="dtype"),
        bs=128,
        nh=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        kv_caches=caches,
        dsa_shared_views={},
    )
    execute([function], ns)
    ns["reshape"](
        NS(dsa_shared_pool=True, layerwise_prefill_p_node=p_node), ["layer.0.attn", "layer.1.attn", "layer.1.indexer"]
    )
    assert [index for index, _ in records] == ([False, True] if p_node else [False, False, True])
    assert (caches["layer.0.attn"] is caches["layer.1.attn"]) is p_node


@pytest.mark.parametrize(
    "path,cls,v2",
    [
        ("core/recompute_scheduler.py", "RecomputeScheduler", False),
        ("core/recompute_scheduler.py", "RecomputeScheduler", True),
        ("core/scheduler_dynamic_batch.py", "SchedulerDynamicBatch", False),
        ("patch/platform/patch_balance_schedule.py", "BalanceScheduler", False),
        ("patch/platform/patch_balance_schedule.py", "BalanceScheduler", True),
    ],
)
@pytest.mark.parametrize("p_node", [False, True])
def test_scheduler_d_constructs_baseline_payload_directly(path, cls, v2, p_node):
    schedule = method(path, cls, "schedule")
    start = next(i for i, n in enumerate(schedule.body) if assigns(n, "layerwise_prefill"))
    nodes = schedule.body[start : start + 2]
    seen = []
    req = NS(request_id="r", _all_token_ids=[10, 11])
    blocks = NS(get_block_ids=lambda: ("physical",))
    obj = NS(use_v2_model_runner=v2, kv_cache_manager=NS(coordinator=NS(layerwise_prefill_p_node=p_node)))
    if p_node:
        obj._make_new_request_data = lambda *args: seen.append(("banked", args)) or args
    ns = dict(
        self=obj,
        NewRequestData=NS(from_request=lambda *args: seen.append(("legacy", args)) or args),
        scheduled_new_reqs=[req],
        scheduled_resumed_reqs=[],
        req_to_new_blocks={"r": blocks},
    )
    execute(nodes, ns)
    assert seen == [
        (
            "banked" if p_node else "legacy",
            (req, blocks if p_node else ("physical",), *([req._all_token_ids] if v2 else [])),
        )
    ]
    assert len(ns["new_reqs_data"]) == 1


@pytest.mark.parametrize("p_node,expected", [(False, True), (True, False)])
def test_d_cold_resume_classification_keeps_baseline_without_shrink(p_node, expected):
    route = method("worker/model_runner_v1.py", "NPUModelRunner", "_staged_sfa_local_route")
    start = next(i for i, n in enumerate(route.body) if assigns(n, "possible_cold_resume"))
    ns = dict(
        self=NS(layerwise_prefill_p_node=p_node, dsa_shrink_latent=0),
        np=np,
        is_decode_state=False,
        graph_configured=False,
        num_computed_tokens=[4],
        prompt_lens=[5],
        num_reqs=1,
    )
    execute(route.body[start : start + 2], ns)
    assert ns["possible_cold_resume"] is expected


def test_d_full_graph_callback_check_ignores_p_capability():
    validate = method("worker/model_runner_v1.py", "NPUModelRunner", "_validate_sfa_layerwise_connector_cudagraph_mode")
    node = next(n for n in validate.body if assigns(n, "uses_layerwise_callbacks"))

    class Connector:
        uses_layerwise_model_callbacks = False

        @property
        def supports_layerwise_prefill_transfer_window(self):
            raise AssertionError("P capability was evaluated on D")

    ns = dict(connector=Connector(), layerwise_prefill_p_node=False)
    execute([node], ns)
    assert ns["uses_layerwise_callbacks"] is False
