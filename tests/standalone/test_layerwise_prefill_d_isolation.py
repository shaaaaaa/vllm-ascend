# SPDX-License-Identifier: Apache-2.0
"""Exercise production P gates without loading the Ascend runtime."""

import ast
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

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


def initialized_shrink(p_node, configured, *, builder=False, unbundle=True):
    owner = "AscendSFAMetadataBuilder" if builder else "AscendSFAImpl"
    init = method("attention/sfa_v1.py", owner, "__init__")
    assignment = next(
        node
        for node in init.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "dsa_shrink_latent" for target in node.targets)
    )
    obj = NS(_layerwise_prefill_p_node=p_node, dsa_offload_unbundle=unbundle)
    execute(
        [assignment],
        dict(
            self=obj,
            envs=NS(
                VLLM_ASCEND_DSA_UNBUNDLE=unbundle,
                VLLM_ASCEND_DSA_SHRINK_LATENT=configured,
            ),
        ),
    )
    return obj


@pytest.mark.parametrize("builder", [False, True])
@pytest.mark.parametrize("p_node", [False, True])
@pytest.mark.parametrize("unbundle", [False, True])
@pytest.mark.parametrize("configured", [0, 1, 2, 3])
def test_compact_scratch_policy_excludes_only_banked_p(builder, p_node, unbundle, configured):
    obj = initialized_shrink(p_node, configured, builder=builder, unbundle=unbundle)
    assert obj.dsa_shrink_latent == (configured if unbundle and not p_node else 0)


@pytest.mark.parametrize("p_node", [False, True])
@pytest.mark.parametrize("mlapo", [False, True])
@pytest.mark.parametrize("has_indexer", [False, True])
@pytest.mark.parametrize("decode_rows", [0, 1, 2, 4])
def test_bank_waits_follow_native_qkv_and_protect_cache_writes(p_node, mlapo, has_indexer, decode_rows):
    forward = method("attention/sfa_v1.py", "AscendSFAImpl", "forward")
    dispatch = deepcopy(next(
        node for node in forward.body
        if isinstance(node, ast.If) and "self.enable_mlapo" in ast.unparse(node.test)
    ))
    first_cache_write = next(i for i, node in enumerate(dispatch.orelse) if calls(node, "exec_kv"))
    dispatch.orelse = dispatch.orelse[:first_cache_write + 1]
    index_write = next(
        node for node in forward.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "kv_cache is not None and self.has_indexer"
    )
    obj = initialized_shrink(p_node, 2)
    obj.enable_mlapo = mlapo
    obj.has_indexer = has_indexer
    obj.q_lora_rank = obj.kv_lora_rank = obj.qk_rope_head_dim = 1
    obj.q_a_layernorm = lambda tensor: tensor
    obj.enable_dsa_cp = True
    obj.use_sparse_c8_indexer = False
    obj.is_kv_producer = False
    operations = []
    value = torch.zeros(4, 1)

    def wait(layer_name):
        assert layer_name not in operations, "Duplicate wait advances a retriever twice"
        operations.append(layer_name)

    def write_latent(*args, **kwargs):
        if p_node:
            assert "latent" in operations
        operations.append("write_latent")
        return value, value

    def project(tensor):
        assert "latent" not in operations and "indexer" not in operations
        operations.append("native_qkv")
        return (NS(split=lambda *args, **kwargs: (value, value)),)

    project.weight = object()
    obj.fused_qkv_a_proj = project
    obj.exec_kv = write_latent

    def preprocess(**kwargs):
        if p_node:
            assert operations == ["latent"]
        operations.append("mlapo_qkv")
        write_latent()
        return value, value, value, value

    obj._sfa_preprocess_with_mlapo = preprocess

    def indexer_preprocess(**kwargs):
        assert "indexer" not in operations
        operations.append("indexer_qkv")
        return value, None

    def write_index(*args):
        assert "indexer" in operations
        operations.append("write_index")

    obj.indexer_select_pre_process = indexer_preprocess
    namespace = dict(
        self=obj,
        dcp=None,
        attn_metadata=NS(num_decode_tokens=decode_rows, num_actual_tokens=4),
        layer_name="latent",
        index_layer_name="indexer",
        index_lmcache_enabled=has_indexer,
        hidden_states=value,
        kv_cache=(value, value, value),
        cos=value,
        sin=value,
        slot_mapping=torch.arange(4),
        idx_slot_mapping=torch.arange(4),
        slot_mapping_cp=torch.arange(4),
        num_input_tokens=4,
        MLAPO_MAX_SUPPORTED_TOKENS=16,
        _dsa_prof=NS(section=lambda name: nullcontext()),
        get_weight_prefetch_method=lambda: NS(maybe_prefetch_mla_or_sla_weight_in_current_stream=lambda **kwargs: None),
        wait_for_kv_layer_from_connector=wait,
        torch_npu=NS(npu_scatter_nd_update_=write_index),
    )
    execute([dispatch, index_write], namespace)
    if mlapo:
        expected = (["latent"] if p_node else []) + ["mlapo_qkv", "write_latent"]
        if has_indexer:
            expected.append("indexer_qkv")
    else:
        expected = ["native_qkv"]
        if has_indexer:
            expected.append("indexer_qkv")
        if p_node or decode_rows == 0:
            expected.append("latent")
        expected.append("write_latent")
    if has_indexer:
        expected.extend(["indexer", "write_index"])
    assert operations == expected


@pytest.mark.parametrize("p_node,remaps", [(False, True), (True, False)])
def test_p_does_not_remap_full_bank_topk_to_decode_scratch(p_node, remaps):
    obj = initialized_shrink(p_node, 2)
    forward = method("attention/sfa_v1.py", "AscendSFAImpl", "forward")
    compact = next(
        node
        for node in forward.body
        if isinstance(node, ast.If) and "attn_metadata.need_sparse_lmcache_payload" in ast.unparse(node.test)
    )
    # Execute the production dispatch condition; no fake NPU remap implementation.
    branch = deepcopy(compact)
    branch.body = ast.parse("selected.append('decode_scratch')").body
    selected = []
    execute(
        [branch],
        dict(
            self=obj,
            selected=selected,
            dcp=None,
            attn_metadata=NS(
                split_boundary=object(),
                num_decode_tokens=2,
                need_sparse_lmcache_payload=True,
            ),
        ),
    )
    assert bool(selected) is remaps


@pytest.mark.parametrize("p_node", [False, True])
def test_mlapo_waits_only_for_its_latent_bank_before_cache_write(p_node):
    forward = method("attention/sfa_v1.py", "AscendSFAImpl", "forward")
    outer = next(n for n in forward.body if isinstance(n, ast.If) and "self.enable_mlapo" in ast.unparse(n.test))
    dispatch = outer.body[:2]
    seen = []

    def preprocess(**kwargs):
        seen.append(("legacy", kwargs))
        return 1, 2, 3, 4

    obj = NS(_layerwise_prefill_p_node=p_node, _sfa_preprocess_with_mlapo=preprocess)
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
    ns["wait_for_kv_layer_from_connector"] = lambda name: seen.append(("wait", name))
    execute(dispatch, ns)
    assert [entry[0] for entry in seen] == (["wait"] if p_node else []) + ["legacy"]
    assert ns["ql_nope"] == 2
    assert "layer_name" not in seen[-1][1]


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
    obj = NS(layerwise_prefill_p_node=p_node, dsa_decoder_dcp=False, input_batch=NS(block_table=primary))
    if p_node:
        obj._refresh_layerwise_prefill_block_tables = lambda: (primary, shadow)
    execute(nodes, dict(self=obj, num_reqs=1, req_indices="req", positions_np="pos",
                        total_num_scheduled_tokens=3, scheduler_output=NS(dsa_dcp_compact_req_ids=None)))
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


@pytest.mark.parametrize("dcp_size", [4, 8])
@pytest.mark.parametrize("rank", [0, 1])
def test_dcp_slot_mapping_keeps_indexer_replicated_and_shifts_only_compact_rows(dcp_size, rank):
    def buffer(*shape, **kwargs):
        return NS(np=np.zeros(shape, dtype=np.int32), copy_to_gpu=lambda count: None)

    scope = dict(
        np=np, torch=torch, CpuGpuBuffer=buffer, cdiv=lambda a, b: (a + b - 1) // b,
        get_dcp_group=lambda: NS(world_size=dcp_size, rank_in_group=rank),
        get_pcp_group=lambda: NS(world_size=1, rank_in_group=0),
    )
    tree = ast.parse((ROOT / "worker/block_table.py").read_text(encoding="utf-8"))
    execute([n for n in tree.body if isinstance(n, ast.ClassDef)], scope)
    tables = scope["MultiGroupBlockTable"](
        2, 2048, 4, False, torch.device("cpu"), [128, 128],
        num_speculative_tokens=1, max_num_blocks=[40, 16], kernel_sizes=[[128], [128]],
        cp_kv_cache_interleave_size=128, dsa_dcp_sizes=[dcp_size, 1],
    )
    tables.add_row(([7], list(range(20, 36))), 1)
    tables.add_row((list(range(1, 33)) + [0, 51], list(range(40, 56))), 0)
    prepare = method("worker/model_runner_v1.py", "NPUModelRunner", "_prepare_inputs")
    mapping = next(n for n in prepare.body if isinstance(n, ast.If)
                   and ast.unparse(n.test) == "self.layerwise_prefill_p_node"
                   and calls(n, "compute_slot_mapping"))
    positions = np.array([127, 128, 128 * dcp_size, 128 * (dcp_size + 1)])
    runner = NS(
        layerwise_prefill_p_node=False, dsa_decoder_dcp=True,
        input_batch=NS(block_table=tables, req_ids=["compact", "dense"]),
        _dsa_dcp_scratch_origin=4096,
        _dsa_dcp_live_origins=np.zeros(2, dtype=np.int32),
    )
    inputs = dict(self=runner, num_reqs=2, req_indices=np.array([1, 1, 0, 0]),
                  positions_np=positions, total_num_scheduled_tokens=4,
                  scheduler_output=NS(dsa_dcp_compact_req_ids=None))
    with pytest.raises(RuntimeError, match="missing the compact table layout"):
        execute([mapping], inputs)
    inputs["scheduler_output"].dsa_dcp_compact_req_ids = {"compact"}
    execute([mapping], inputs)
    assert tables[0].block_table.np.shape == (2, 40)
    assert tables[0].slot_mapping.np[:4].tolist() == (
        [7 * 128 + 127, -1, 51 * 128, -1] if rank == 0 else [-1, 7 * 128, -1, 51 * 128]
    )
    assert tables[1].slot_mapping.np[:4].tolist() == [
        20 * 128 + 127, 21 * 128, (40 + dcp_size) * 128, (41 + dcp_size) * 128,
    ]


@pytest.mark.parametrize("p_node", [False, True])
def test_shared_pool_only_p_reuses_layer_tensor_objects(p_node):
    reshape = method("worker/model_runner_v1.py", "NPUModelRunner", "_reshape_kv_cache_tensors")
    gate = next(
        n
        for n in ast.walk(reshape)
        if isinstance(n, ast.If) and ast.unparse(n.test).startswith("self.dsa_shared_pool and")
    )
    function = ast.parse("def reshape(self, layers):\n    for layer_name in layers:\n        pass\n").body[0]
    function.body[0].body = [gate]
    records = []

    def views(*args, is_indexer, **kwargs):
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
        layer_c8=False,
        dsa_shared_views={},
    )
    execute([function], ns)
    ns["reshape"](
        NS(
            dsa_shared_pool=True,
            layerwise_prefill_p_node=p_node,
            use_sparse_c8_indexer=False,
            _mixed_indexer_c8_names=None,
            kv_cache_config=NS(),
        ),
        ["layer.0.attn", "layer.1.attn", "layer.1.indexer"],
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
