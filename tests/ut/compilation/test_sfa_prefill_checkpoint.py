# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checkpoint/dispatch regressions. Not an NPU/model correctness claim."""

import ast
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_sfa_parity import checkpoint as checkpoint
from test_sfa_parity import parity as parity
from test_sfa_parity import worker as worker


def bindings(checkpoint, *, moved=False):
    groups = {}
    for name, blocks, widths in (("latent", [1, 3, 1], (4, 2)), ("indexer", [2, 4, 2], (3,))):
        offset = 2 if moved else 0
        groups[name] = checkpoint.CacheBinding(
            tuple(torch.zeros(8, 2, 1, width, dtype=torch.bfloat16) for width in widths),
            torch.tensor([[block + offset for block in blocks]], dtype=torch.int32),
        )
    return groups


def payload(checkpoint, source):
    return {
        "caches": checkpoint.capture_caches(source),
        "callbacks": [("wait", "latent"), ("wait", "indexer"), ("save", "latent"), ("save", "indexer")],
    }


def test_checkpoint_restores_independent_groups_and_keeps_destination_addresses(checkpoint):
    source, destination = bindings(checkpoint), bindings(checkpoint, moved=True)
    for group in source.values():
        for cache in group.caches:
            cache.copy_(torch.arange(cache.numel()).reshape(cache.shape))
            cache[1, 0, 0, 0] = float("nan")  # Unused storage bits must survive unchanged.
    snapshot = payload(checkpoint, source)
    plan = checkpoint.plan_cache_restore(snapshot["caches"], destination)
    pointers = [cache.data_ptr() for group in destination.values() for cache in group.caches]
    calls = []
    checkpoint.restore_prefill(
        snapshot,
        destination,
        plan,
        wait=lambda name: calls.append(("wait", name)),
        save=lambda name, caches: calls.append(("save", name)),
    )
    checkpoint.verify_cache_restore(snapshot["caches"], destination, plan)
    assert calls == snapshot["callbacks"]
    assert pointers == [cache.data_ptr() for group in destination.values() for cache in group.caches]
    for group in destination.values():
        assert not group.caches[0][0].any()  # Unrelated physical blocks were not overwritten.
    destination["latent"].caches[0].zero_()
    # Reference bytes are independent of the second decode branch's mutations.
    assert snapshot["caches"]["latent"]["values"][0].isnan().any()
    with pytest.raises(checkpoint.ParityError, match="restored_KV"):
        checkpoint.verify_cache_restore(snapshot["caches"], destination, plan)


@pytest.mark.parametrize("bad", ["missing_group", "alias", "shape", "dtype", "oob", "bad_mask", "bad_blocks"])
def test_checkpoint_validates_all_layouts_before_any_write(checkpoint, bad):
    source, destination = bindings(checkpoint), bindings(checkpoint, moved=True)
    snapshot = payload(checkpoint, source)
    if bad == "missing_group":
        destination.pop("indexer")
    elif bad == "alias":
        destination["indexer"].block_table[0, 2] = 5
    elif bad == "shape":
        destination["indexer"].caches = (torch.zeros(8, 2, 1, 4, dtype=torch.bfloat16),)
    elif bad == "dtype":
        destination["indexer"].caches = (destination["indexer"].caches[0].float(),)
    elif bad == "oob":
        destination["indexer"].block_table[0, 1] = 99
    elif bad == "bad_mask":
        destination["indexer"].block_table[0, 1] = -1
    else:
        snapshot["caches"]["indexer"]["blocks"][0] = 0
    with pytest.raises(checkpoint.ParityError):
        checkpoint.plan_cache_restore(snapshot["caches"], destination)
    assert all(not cache.any() for group in destination.values() for cache in group.caches)


@pytest.mark.parametrize("bad", ["missing", "duplicate", "unknown", "sparse_wait"])
def test_checkpoint_refuses_to_silently_omit_connector_state(checkpoint, bad):
    groups = bindings(checkpoint)
    calls = payload(checkpoint, groups)["callbacks"]
    if bad == "missing":
        calls.pop()
    elif bad == "duplicate":
        calls.append(calls[-1])
    elif bad == "unknown":
        calls.append(("save", "foreign-request"))
    else:
        calls[0] = ("unsupported_wait", "latent")
    with pytest.raises(checkpoint.ParityError):
        checkpoint.validate_callbacks(calls, groups)


@pytest.mark.parametrize("rank", range(8))
def test_actual_worker_exports_once_then_imports_without_calling_model(worker, checkpoint, monkeypatch, tmp_path, rank):
    current = SimpleNamespace(moe_layer_index=0)
    monkeypatch.setattr(worker, "get_forward_context", lambda: current)
    rng = torch.Generator().manual_seed(0).get_state()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(synchronize=lambda: None, get_rng_state=lambda: rng, set_rng_state=lambda value: None),
        raising=False,
    )
    module = sys.modules["vllm_ascend.attention.sfa_v1"]
    callbacks = []
    module.wait_for_kv_layer_from_connector = lambda name: callbacks.append(("wait", name))
    module.maybe_save_kv_layer_to_connector = lambda name, caches: callbacks.append(("save", name))
    source, dest = bindings(checkpoint), bindings(checkpoint, moved=True)
    source_state, dest_state = {"counts": torch.tensor([3])}, {"counts": torch.tensor([0])}

    def subject(graph):
        instance = worker.SFAParityWorker()
        instance.parity_rank, instance.parity_tp_size, instance.parity_is_graph = rank, 8, graph
        instance.parity_directory = tmp_path / f"rank-{rank}"
        instance.parity_directory.mkdir(exist_ok=True)
        instance.device = torch.device("cpu")
        instance._check = lambda check, phase: check()
        instance._prefill_bindings = lambda ctx, names: (
            (dest if graph else source),
            (dest_state if graph else source_state),
            {"sequence_end": 4},
        )
        return instance

    inputs = {"input_ids": torch.tensor([100, 101]), "positions": torch.tensor([0, 1])}
    computations = []

    def prefill():
        computations.append("prefill")
        for name in ("latent", "indexer"):
            module.wait_for_kv_layer_from_connector(name)
        for name, group in source.items():
            for cache in group.caches:
                cache.fill_(rank + 1)
            module.maybe_save_kv_layer_to_connector(name, list(group.caches))
        current.moe_layer_index = 8
        return (torch.ones(2, 4) * rank, [torch.tensor([1.0])])

    eager = subject(False)
    output = eager._prefill_pass("target", 0, [], inputs, prefill)
    assert computations == ["prefill"]
    with pytest.raises(worker.ParityError, match="same prefill twice"):
        eager._prefill_pass("target", 0, [], inputs, prefill)
    assert computations == ["prefill"]
    expected_calls = list(callbacks)
    for group in source.values():  # Original eager decode later mutates its own state.
        for cache in group.caches:
            cache.fill_(99)
    callbacks.clear()

    def forbidden():
        raise AssertionError("The graph engine must never compute prefill")

    graph = subject(True)
    restored = graph._prefill_pass("target", 0, [], inputs, forbidden)
    checkpoint.assert_same_tree(output, restored, "output")
    assert computations == ["prefill"]
    assert callbacks == expected_calls
    assert dest_state["counts"].item() == 3
    assert current.moe_layer_index == 8
    assert torch.equal(dest["latent"].caches[0][3], torch.full((2, 1, 4), rank + 1))
    wrong = {**inputs, "input_ids": torch.tensor([101, 102])}
    callbacks.clear()
    with pytest.raises(worker.ParityError, match="inputs"):
        graph._prefill_pass("target", 0, [], wrong, forbidden)
    assert not callbacks
    with pytest.raises(FileNotFoundError):
        graph._prefill_pass("target", 1, [], inputs, forbidden)


@pytest.mark.parametrize("graph", [False, True])
def test_dispatch_calls_original_decode_but_graph_prefill_only_imports(worker, monkeypatch, graph):
    subject = worker.SFAParityWorker()
    subject.parity_is_graph, subject.parity_rank, subject.parity_tp_size = graph, 1, 8
    for field in (
        "step",
        "prefill_steps",
        "prefill_tokens",
        "prefill_model_calls",
        "prefill_imports",
        "decode_steps",
        "q2_steps",
    ):
        setattr(subject, "parity_" + field, 0)
    subject.model_runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=1), _sfa_full_graph=SimpleNamespace(replay_count=0)
    )
    context = SimpleNamespace(staged_sfa_graph_dummy_run=False, skip_compiled=False, cudagraph_runtime_mode="NONE")
    monkeypatch.setattr(worker, "get_forward_context", lambda: context)
    subject._check = lambda check, phase: check()
    subject.parity_attention_names = []
    phase = {"decode": False}
    subject._prepare_step = lambda inputs, ctx: {"decode": phase["decode"], "rows": 2}
    observations, calls = [], []
    subject._observe_step = lambda state, result, replays: observations.append((state["decode"], replays))

    def checkpoint_pass(kind, step, names, inputs, run):
        calls.append("import" if graph else "export")
        return torch.ones(2, 4) if graph else run()

    subject._prefill_pass = checkpoint_pass

    def original(input_ids):
        calls.append("decode" if phase["decode"] else "prefill_compute")
        if graph and phase["decode"]:
            subject.model_runner._sfa_full_graph.replay_count += 1
        return torch.ones(2, 4)

    signature = inspect.signature(original)
    subject._parity_forward(original, signature, torch.ones(2))
    assert calls == (["import"] if graph else ["export", "prefill_compute"])
    assert not observations  # Imported prefill is not falsely reported as numerical parity.
    assert subject.parity_prefill_model_calls == int(not graph)
    phase["decode"] = True
    context.cudagraph_runtime_mode = "PIECEWISE" if graph else "NONE"
    subject._parity_forward(original, signature, torch.ones(2))
    assert calls[-1] == "decode"
    assert observations == [(True, int(graph))]
    assert context.skip_compiled is False
    assert subject.parity_decode_steps == subject.parity_q2_steps == 1
    context.staged_sfa_graph_dummy_run = True
    subject._parity_forward(original, signature, torch.ones(2))
    assert calls[-1] == "decode"  # Startup capture/warmup is never imported.
    assert len(observations) == 1


@pytest.mark.parametrize("graph", [False, True])
def test_mtp_initial_state_import_is_separate_from_real_decode_draft(worker, graph):
    subject = worker.SFAParityWorker()
    subject.model_runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=1), drafter=SimpleNamespace(attn_layer_names=["draft-attention"])
    )
    subject.parity_last_target_decode, subject.parity_last_target_step = False, 8
    subject.parity_is_graph = graph
    subject.parity_draft_prefill_model_calls = subject.parity_draft_prefill_imports = 0
    calls = []

    def original(inputs, **kwargs):
        calls.append("model")
        return torch.ones(1)

    def checkpoint_pass(kind, step, names, inputs, run):
        assert kind == "draft" and step == 8 and names == ["draft-attention"]
        calls.append("import" if graph else "export")
        return torch.ones(1) if graph else run()

    subject._prefill_pass = checkpoint_pass
    subject._parity_draft_forward(original, {}, draft_step=0, runtime_inputs={})
    assert calls == (["import"] if graph else ["export", "model"])
    subject.parity_last_target_decode = True
    subject._parity_draft_forward(original, {}, draft_step=0, runtime_inputs={})
    assert calls[-1] == "model"
    assert subject.parity_draft_prefill_model_calls == int(not graph)
    assert subject.parity_draft_prefill_imports == int(graph)


def test_actual_sfa_cache_binding_preserves_groups_and_excludes_capture_dummy_state(worker, monkeypatch):
    # Execute the real SFA cache registry method, not a fake successful binding.
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSFAImpl")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_cross_layer_kv_cache"
    )
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_dsa_indexer_layer_name"
    )
    context = SimpleNamespace(virtual_engine=0, no_compile_layers={}, attn_metadata={})
    namespace = {"torch": torch, "get_forward_context": lambda: context, "_dsa_index_lmcache_enabled": lambda: True}
    exec(compile(ast.Module(body=[helper, method], type_ignores=[]), str(path), "exec"), namespace)
    monkeypatch.setattr(
        worker.AscendSFAImpl, "_cross_layer_kv_cache", namespace["_cross_layer_kv_cache"], raising=False
    )
    name = "model.layers.0.self_attn.attn"
    index_name = "model.layers.0.self_attn.indexer.k_cache"
    impl = worker.AscendSFAImpl()
    impl.dsa_offload_unbundle = True
    impl._sorted_resident_state = SimpleNamespace(
        dummy_state_base=1,
        tokens=torch.zeros(2, 3, dtype=torch.int32),
        slots=torch.zeros(2, 3, dtype=torch.int16),
        counts=torch.zeros(2, 2, dtype=torch.int32),
        generations=torch.full((2, 2), -1, dtype=torch.int64),
    )
    latent = (torch.ones(5, 2, 1, 4), torch.ones(5, 2, 1, 2))
    index = torch.ones(7, 2, 1, 3)
    context.no_compile_layers[name] = SimpleNamespace(impl=impl, kv_cache=[latent])
    context.no_compile_layers[index_name] = SimpleNamespace(kv_cache=[index])
    context.attn_metadata[name] = SimpleNamespace(
        block_table=torch.tensor([[1, 2]], dtype=torch.int32),
        indexer_block_table=torch.tensor([[3, 4]], dtype=torch.int32),
        num_actual_tokens=3,
        num_decode_tokens=0,
        seq_lens=torch.tensor([3]),
        cum_query_lens=torch.tensor([3]),
    )
    groups, persistent, metadata = worker.SFAParityWorker()._prefill_bindings(context, [name])
    assert groups[name].caches == latent and groups[index_name].caches[0] is index
    assert groups[name].block_table.tolist() == [[1, 2]]
    assert groups[index_name].block_table.tolist() == [[3, 4]]
    for field in ("tokens", "slots", "counts", "generations"):
        original = getattr(impl._sorted_resident_state, field)
        view = persistent[f"{name}.{field}"]
        assert view.shape[0] == 1 and view.data_ptr() == original.data_ptr()
        view.fill_(5)
        assert not torch.all(original[1] == 5)  # Capture dummy row was not restored.
    context.attn_metadata[name].seq_lens.fill_(999)
    assert metadata[name]["seq_lens"].tolist() == [3]


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("compare_output", [False, True])
def test_final_gate_requires_zero_graph_prefill_compute_and_real_decode(worker, tmp_path, graph, compare_output):
    subject = worker.SFAParityWorker()
    subject.parity_directory = tmp_path
    subject.parity_is_graph = graph
    subject.parity_rank, subject.parity_tp_size = 0, 8
    subject.parity_step, subject.parity_prefill_steps, subject.parity_prefill_tokens = 3, 1, 4351
    subject.parity_decode_steps = subject.parity_q2_steps = subject.parity_draft_calls = 2
    subject.parity_options = {"compare_output": compare_output}
    subject.parity_decode_observations = 2
    subject.parity_transfers = [10] * 8
    subject.parity_prefill_model_calls = subject.parity_draft_prefill_model_calls = int(not graph)
    subject.parity_prefill_imports = subject.parity_draft_prefill_imports = int(graph)
    names = ["target-prefill-000000.pt", "draft-prefill-000000.pt"]
    if not compare_output:
        names += ["step-000001.pt", "step-000002.pt"]
    for name in names:
        (tmp_path / name).touch()
    summary = subject._local_summary()
    assert summary["prefill_model_calls"] == int(not graph)
    assert summary["compare_output"] == compare_output
    if compare_output:
        subject.parity_decode_observations -= 1
        with pytest.raises(worker.ParityError, match="observation coverage"):
            subject._local_summary()
        subject.parity_decode_observations += 1
    subject.parity_prefill_model_calls += 1
    with pytest.raises(worker.ParityError, match="compute once"):
        subject._local_summary()
    subject.parity_prefill_model_calls -= 1
    subject.parity_decode_steps = 0
    with pytest.raises(worker.ParityError, match="phase coverage"):
        subject._local_summary()
