# SPDX-License-Identifier: Apache-2.0
"""Execute the real runner/SFA transfer control flow without torch or an NPU."""

import ast
from copy import deepcopy
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2] / "vllm_ascend"


def source_class(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def assigns(node, name):
    if isinstance(node, ast.Assign):
        return any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    return isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name


def execute(nodes, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *deepcopy(nodes)], type_ignores=[]))
    exec(compile(module, "<real-layerwise-prefill-save-policy>", "exec"), namespace)


@pytest.fixture
def api():
    state = source_class("attention/attention_v1.py", "AscendAttentionState")
    runner = source_class("worker/model_runner_v1.py", "NPUModelRunner")
    classify = next(n for n in runner.body if getattr(n, "name", None) == "_build_attn_state")
    prepare = next(n for n in runner.body if getattr(n, "name", None) == "_prepare_inputs")
    # Keep the scheduler's actual valid-token calculation. A sampled token is
    # not num_valid_tokens: without drafts, valid == scheduled, even on P.
    prepare_start = next(
        i
        for i, n in enumerate(prepare.body)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "not scheduler_output.scheduled_spec_decode_tokens"
    )
    prepare_end = next(i for i, n in enumerate(prepare.body) if assigns(n, "attn_state"))
    runner_class = ast.ClassDef(name="Runner", bases=[], keywords=[], body=[classify], decorator_list=[])

    sfa = source_class("attention/sfa_v1.py", "AscendSFAImpl")
    forward = next(n for n in sfa.body if getattr(n, "name", None) == "forward")
    entry_start = next(i for i, n in enumerate(forward.body) if assigns(n, "transfer_context"))
    entry_end = next(i for i, n in enumerate(forward.body) if assigns(n, "pending_transfer_names"))
    pure_decode = next(n for n in forward.body if assigns(n, "_is_pure_decode"))
    save_start = next(i for i, n in enumerate(forward.body) if assigns(n, "save_operations"))
    final_flush = next(
        i
        for i, n in enumerate(forward.body)
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "is_last_transfer_layer"
    )
    tail = next(
        n
        for n in forward.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "use_layerwise_transfer_window"
    )
    profile_return = next(
        n for n in forward.body if isinstance(n, ast.If) and ast.unparse(n.test) == "attn_metadata is None"
    )
    # Run production transfer entry, policy, final-layer flush, and projection
    # tail together. Device attention/projection math is outside this contract.
    control = ast.parse(
        "def control(self, attn_metadata, layer_name, kv_cache, index_layer_name, "
        "index_lmcache_enabled, attn_output, output):\n    pass\n"
    ).body[0]
    control.body = [
        profile_return,
        *forward.body[entry_start : entry_end + 1],
        pure_decode,
        *forward.body[save_start : final_flush + 1],
        tail,
    ]
    helper_names = {
        "_submit_sfa_save_operations",
        "_submit_sfa_transfer_window_save_operations",
        "_submit_sfa_post_transfer_window_save_operations",
        "_submit_sfa_layerwise_transfer_window",
        "_finish_sfa_layerwise_transfer_window",
    }
    helpers = [n for n in sfa.body if getattr(n, "name", None) in helper_names]
    assert len(helpers) == len(helper_names)
    impl_class = ast.ClassDef(name="Impl", bases=[], keywords=[], body=[*helpers, control], decorator_list=[])
    context = NS(additional_kwargs={})
    events = []

    def record(kind):
        def callback(name, *args):
            events.append((kind, name))
            return True

        return callback

    namespace = dict(
        np=np,
        Enum=Enum,
        get_forward_context=lambda: context,
        maybe_save_kv_layer_to_connector=record("legacy_save"),
        maybe_save_kv_layer_in_layerwise_prefill_transfer_window=record("save"),
        maybe_save_kv_layer_outside_layerwise_prefill_transfer_window=record("post_save"),
        maybe_submit_layerwise_prefill_load=record("load"),
        maybe_finish_layerwise_prefill_save=record("finish"),
    )
    execute([state, runner_class, impl_class], namespace)
    return NS(
        namespace=namespace,
        states=namespace["AscendAttentionState"],
        Runner=namespace["Runner"],
        Impl=namespace["Impl"],
        prepare_nodes=prepare.body[prepare_start : prepare_end + 1],
        context=context,
        events=events,
    )


def classify(api, *, computed, scheduled, drafts=0, mtp=True):
    runner = api.Runner()
    runner.input_batch = NS(req_ids=["r"], num_computed_tokens_cpu=np.array([computed]))
    runner.speculative_config = NS(method="mtp") if mtp else None
    runner.scheduler_config = NS(enable_chunked_prefill=True)
    namespace = dict(
        api.namespace,
        self=runner,
        num_reqs=1,
        num_scheduled_tokens=np.array([scheduled]),
        scheduler_output=NS(
            num_scheduled_tokens={"r": scheduled},
            scheduled_spec_decode_tokens={"r": list(range(drafts))} if drafts else {},
        ),
    )
    execute(api.prepare_nodes, namespace)
    assert namespace["num_valid_tokens"].tolist() == [scheduled - drafts]
    return namespace["attn_state"]


def impl(api, *, p_node=True, window=0, shrink=2):
    obj = api.Impl()
    obj._layerwise_prefill_p_node = p_node
    obj._first_layerwise_prefill_layer_index = 0
    obj._last_layerwise_prefill_layer_index = 77
    obj.layer_name = None
    obj.dsa_shrink_latent = shrink
    obj.dsa_offload_unbundle = True
    obj.enable_dsa_cp_with_o_proj_tp = False
    obj.enable_dsa_cp_with_layer_shard = False
    obj.o_proj = lambda value: (value,)
    api.namespace["_decode_window_save_window_size"] = lambda: window
    api.namespace["layerwise_prefill_transfer_window_supported"] = lambda: p_node
    return obj


def step(api, obj, state, *, layer=0, has_indexer=True, decode_rows=0):
    name = f"model.layers.{layer}.self_attn.attn"
    indexer = f"model.layers.{layer}.self_attn.indexer.k_cache" if has_indexer else None
    caches = [object(), object(), object()]
    obj.control(
        NS(attn_state=state, num_decode_tokens=decode_rows),
        name,
        caches,
        indexer,
        has_indexer,
        np.ones(1),
        np.zeros(1),
    )
    return [name, indexer] if has_indexer else [name]


@pytest.mark.parametrize(
    "computed,scheduled,mtp,expected",
    [
        (0, 8, True, "PrefillNoCache"),
        (4096, 8, True, "ChunkedPrefill"),
        (4096, 2, True, "ChunkedPrefill"),
        (4096, 1, True, "SpecDecoding"),
        (4096, 1, False, "DecodeOnly"),
    ],
)
@pytest.mark.parametrize("has_indexer", [True, False])
def test_p_prefill_keeps_saves_for_real_runner_state(api, computed, scheduled, mtp, expected, has_indexer):
    state = classify(api, computed=computed, scheduled=scheduled, mtp=mtp)
    assert state == getattr(api.states, expected)
    obj = impl(api)
    # These are prompt rows, including the one-token remainder. SFA's builder
    # therefore reports zero decode rows despite the runner's decode enum.
    names = step(api, obj, state, has_indexer=has_indexer, decode_rows=0)
    pending = api.context.additional_kwargs["sfa_layerwise_prefill_pending"]
    assert [name for name, _ in pending] == names
    step(api, obj, state, layer=1, has_indexer=has_indexer, decode_rows=0)
    assert [name for kind, name in api.events if kind == "save"] == names
    assert [name for kind, name in api.events if kind == "load"] == [-1, *names]
    assert [name for kind, name in api.events if kind == "finish"] == names


@pytest.mark.parametrize("has_indexer", [True, False])
def test_one_token_p_tail_flushes_all_target_layers_before_mtp(api, has_indexer):
    state = classify(api, computed=4096, scheduled=1)
    assert state == api.states.SpecDecoding
    obj = impl(api)
    expected = []
    for layer in range(78):
        expected.extend(step(api, obj, state, layer=layer, has_indexer=has_indexer))
    assert "sfa_layerwise_prefill_pending" not in api.context.additional_kwargs
    for kind in ("save", "finish", "post_save"):
        assert [name for event, name in api.events if event == kind] == expected
    assert [name for kind, name in api.events if kind == "load"] == [-1, *expected]
    assert not any(kind == "legacy_save" for kind, _ in api.events)

    # The MTP proposer creates a fresh forward context. Its layer 78 entry
    # must not inherit target pending saves, and its own P save stays eligible.
    mtp_context = NS(additional_kwargs={})
    api.namespace["get_forward_context"] = lambda: mtp_context
    events_before_mtp = list(api.events)
    mtp_names = step(api, obj, state, layer=78, has_indexer=has_indexer, decode_rows=0)
    assert api.events == events_before_mtp
    pending = mtp_context.additional_kwargs["sfa_layerwise_prefill_pending"]
    assert [name for name, _ in pending] == mtp_names


@pytest.mark.parametrize("scheduled,drafts,mtp", [(1, 0, False), (1, 0, True), (2, 1, True)])
@pytest.mark.parametrize("window", [0, 2048])
def test_d_decode_retains_existing_window_save_policy(api, scheduled, drafts, mtp, window):
    state = classify(api, computed=4096, scheduled=scheduled, drafts=drafts, mtp=mtp)
    assert state == (api.states.SpecDecoding if mtp else api.states.DecodeOnly)
    obj = impl(api, p_node=False, window=window)
    names = step(api, obj, state, decode_rows=scheduled)
    expected = [("legacy_save", name) for name in names] if window else []
    assert api.events == expected
    assert not api.context.additional_kwargs


def test_d_without_shrink_still_saves(api):
    state = classify(api, computed=4096, scheduled=1)
    obj = impl(api, p_node=False, shrink=0)
    names = step(api, obj, state, decode_rows=1)
    assert api.events == [("legacy_save", name) for name in names]


def test_d_does_not_consult_p_transfer_capability_or_move_save_policy_before_projection(api):
    obj = impl(api, p_node=False, window=2048)
    del obj._first_layerwise_prefill_layer_index
    del obj._last_layerwise_prefill_layer_index
    order = []
    obj.o_proj = lambda value: (order.append("projection") or value,)
    api.namespace["_decode_window_save_window_size"] = lambda: order.append("save_policy") or 2048

    def unexpected():
        raise AssertionError("D must not consult the P transfer-window capability")

    api.namespace["layerwise_prefill_transfer_window_supported"] = unexpected
    names = step(api, obj, api.states.SpecDecoding, decode_rows=1)
    assert order == ["projection", "save_policy"]
    assert api.events == [("legacy_save", name) for name in names]


def test_no_metadata_profile_forward_never_starts_transfers(api):
    obj = impl(api)
    filled = []
    output = NS(fill_=lambda value: filled.append(value))
    obj.control(None, "model.layers.0.self_attn.attn", [], None, False, None, output)
    assert filled == [0]
    assert api.events == []
    assert not api.context.additional_kwargs
