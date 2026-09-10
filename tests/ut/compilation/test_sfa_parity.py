# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of comparison/probe logic, not evidence of real NPU/model parity."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def parity(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/sfa_parity.py"
    spec = importlib.util.spec_from_file_location("tested_sfa_parity", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def worker(parity, monkeypatch):
    env = SimpleNamespace(VLLM_ASCEND_SFA_FULL_GRAPH=True, VLLM_ASCEND_SFA_STAGED_GRAPH=True)
    for name, attributes in {
        "vllm_ascend": {"envs": env},
        "vllm.forward_context": {"get_forward_context": lambda: None},
        "vllm.distributed": {"get_tp_group": lambda: None},
        "vllm.model_executor.model_loader.dummy_loader": {"DummyModelLoader": type("DummyLoader", (), {})},
        "vllm_ascend.attention.sfa_v1": {"AscendSFAImpl": type("FakeSFAImpl", (), {})},
        "vllm_ascend.worker.worker": {"NPUWorker": object},
    }.items():
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.sfa_parity", parity)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/sfa_parity_worker.py"
    spec = importlib.util.spec_from_file_location("tested_sfa_parity_worker", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_snapshot_is_a_copy_not_an_alias(parity):
    probe = parity.DeviceSnapshot((2, 3), dtype=torch.float32, device="cpu")
    source = torch.arange(6.0).reshape(2, 3)
    probe.write(source)
    source.zero_()
    assert torch.equal(probe.read(2, label="layer=0"), torch.arange(6.0).reshape(2, 3))


@pytest.mark.parametrize("writes", [0, 2])
def test_missing_stale_or_duplicate_probe_fails(parity, writes):
    probe = parity.DeviceSnapshot((2, 3), dtype=torch.float32, device="cpu")
    probe.write(torch.ones(2, 3))
    probe.reset()
    for _ in range(writes):
        probe.write(torch.zeros(2, 3))
    with pytest.raises(parity.ParityError, match="one fresh"):
        probe.read(2, label="layer=3")


def test_q1_does_not_compare_unwritten_padding(parity):
    probe = parity.DeviceSnapshot((2, 3), dtype=torch.float32, device="cpu")
    probe.value.fill_(float("nan"))
    probe.write(torch.ones(1, 3))
    assert torch.isfinite(probe.read(1, label="q1")).all()
    with pytest.raises(parity.ParityError, match="capacity"):
        probe.read(3, label="oversize")


def test_short_planner_prefix_clears_unused_tail(parity):
    probe = parity.DeviceSnapshot((1, 4), dtype=torch.int32, device="cpu")
    probe.write_padded(torch.tensor([[7, 8, 9, 10]]))
    probe.reset()
    probe.write_padded(torch.tensor([[11, 12]]))
    assert probe.read(1, label="selected").tolist() == [[11, 12, -1, -1]]


def test_sparse_kv_uses_request_block_table_and_query_boundaries(parity):
    cache = torch.arange(16.0).reshape(4, 2, 1, 2)
    topk = torch.tensor([[[0, 3, -1]], [[1, 2, -1]], [[0, 1, -1]]])
    tables = torch.tensor([[2, 0], [3, 1]])
    values, valid, invalid, slots = parity.gather_sparse_kv(cache, topk, tables, torch.tensor([2, 3]))
    assert slots.tolist() == [[4, 1, -1], [5, 0, -1], [6, 7, -1]]
    assert values[0, 0].tolist() == [8, 9]
    assert values[2, 1].tolist() == [14, 15]
    assert torch.equal(values[:, -1], torch.zeros(3, 2))
    assert valid.sum() == 6
    assert not invalid.any()


@pytest.mark.parametrize("table,topk", [([[99]], [[[0]]]), ([[0]], [[[2]]]), ([[-1]], [[[0]]])])
def test_invalid_mapping_is_reported_not_hidden_by_clamping(parity, table, topk):
    values, valid, invalid, _ = parity.gather_sparse_kv(
        torch.ones(1, 2, 1, 3), torch.tensor(topk), torch.tensor(table), torch.tensor([1])
    )
    assert invalid.all() and not valid.any()
    assert not values.any()


def test_different_physical_allocations_compare_same_logical_kv(parity):
    first = torch.arange(16.0).reshape(4, 2, 1, 2)
    second = first.flip(0)
    topk = torch.tensor([[[0, 1, 2, 3]]])
    ref, _, _, slots_a = parity.gather_sparse_kv(first, topk, torch.tensor([[0, 2]]), torch.tensor([1]))
    val, _, _, slots_b = parity.gather_sparse_kv(second, topk, torch.tensor([[3, 1]]), torch.tensor([1]))
    assert not torch.equal(slots_a, slots_b)
    parity.compare_tensor(ref, val, label="kv", atol=0, rtol=0)
    second[3, 0, 0, 1] += 1
    corrupted, _, _, _ = parity.gather_sparse_kv(second, topk, torch.tensor([[3, 1]]), torch.tensor([1]))
    with pytest.raises(parity.ParityError, match="first_index=.*max_abs=1"):
        parity.compare_tensor(ref, corrupted, label="layer=2 kv")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_even_identical_nonfinite_reference_cannot_pass(parity, value):
    with pytest.raises(parity.ParityError, match="NaN/Inf"):
        parity.compare_tensor(torch.tensor([value]), torch.tensor([value]), label="hidden")


def test_float_tolerance_and_small_signal_error(parity):
    parity.compare_tensor(torch.tensor([1.0]), torch.tensor([1.001]), label="roundoff")
    with pytest.raises(parity.ParityError, match="mismatches=1/1"):
        parity.compare_tensor(torch.tensor([1e-5]), torch.tensor([0.0]), label="lost signal")


def test_integer_difference_is_exact_despite_float_tolerance(parity):
    with pytest.raises(parity.ParityError, match="first_index"):
        parity.compare_tensor(torch.tensor([2048]), torch.tensor([2049]), label="topk", atol=100, rtol=1)


@pytest.mark.parametrize("actual", [torch.ones(2), torch.ones(1, dtype=torch.float64)])
def test_shape_or_dtype_mismatch_fails(parity, actual):
    with pytest.raises(parity.ParityError, match="shape/dtype"):
        parity.compare_tensor(torch.ones(1), actual, label="input")


def make_step():
    return {
        "rank": 0,
        "tp_size": 8,
        "step": 2,
        "decode": True,
        "rows": 2,
        "layers": 8,
        "input_ids": torch.tensor([100, 100]),
        "positions": torch.tensor([4351, 4352]),
        "seq_lens": torch.tensor([4353]),
        "query_ends": torch.tensor([2]),
        "tensors": {"layer=0 input": torch.ones(2, 3), "layer=1 input": torch.ones(2, 3)},
    }


def test_compare_checks_step_alignment_before_layer_differences(parity):
    ref, val = make_step(), make_step()
    val["input_ids"][0] = 101
    val["tensors"]["layer=0 input"].zero_()
    with pytest.raises(parity.ParityError, match="step=2 input_ids"):
        parity.compare_step(ref, val, atol=0, rtol=0)


def test_compare_reports_first_divergent_layer(parity):
    ref, val = make_step(), make_step()
    val["tensors"]["layer=1 input"][0, 1] += 2
    with pytest.raises(parity.ParityError, match="step=2 layer=1 input"):
        parity.compare_step(ref, val, atol=0, rtol=0)


def test_empty_coverage_or_missing_layer_cannot_pass(parity):
    ref, val = make_step(), make_step()
    del val["tensors"]["layer=1 input"]
    with pytest.raises(parity.ParityError, match="coverage"):
        parity.compare_step(ref, val, atol=0, rtol=0)
    ref["tensors"].clear()
    val["tensors"].clear()
    with pytest.raises(parity.ParityError, match="coverage"):
        parity.compare_step(ref, val, atol=0, rtol=0)


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_tolerance_cannot_hide_differences(parity, tolerance):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        parity.compare_tensor(torch.ones(1), torch.zeros(1), label="hidden", atol=tolerance)


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(4)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states.clone()
        else:
            residual = residual + hidden_states
        return hidden_states * 2, residual


def fake_impl():
    return SimpleNamespace(
        index_topk=2,
        local_num_heads=1,
        kv_lora_rank=3,
        qk_rope_head_dim=2,
        indexer_select_post_process=lambda: torch.tensor([[[0, 1]], [[0, 1]]], dtype=torch.int32),
        _prepare_decode_sparse_indices=lambda *a, **k: (
            a[0],
            torch.tensor([[0, 1]], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([[0, 1]]),
        ),
        _execute_sparse_flash_attention_process=lambda q, *a, **k: q,
    )


def test_decoder_hooks_are_traceable_tensor_operations(worker):
    layer = TinyDecoder()
    probes = worker.LayerSnapshots(layer, fake_impl(), 0, 4, torch.device("cpu"))
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    for rows in (2, 1, 2):
        probes.reset()
        x = torch.arange(rows * 4.0).reshape(rows, 4)
        compiled(torch.arange(rows), x, None)
        tensors, _ = probes.read(rows, decode=False)
        assert torch.equal(tensors["layer=0 input.hidden"], x)
        assert torch.equal(tensors["layer=0 output.hidden"], x * 2)
        assert not tensors["layer=0 input.residual"].any()


def test_attention_probes_capture_current_kv_and_mask_planner_tail(worker):
    layer, impl = TinyDecoder(), fake_impl()
    probes = worker.LayerSnapshots(layer, impl, 0, 4, torch.device("cpu"))
    layer(torch.arange(2), torch.ones(2, 4), None)
    indices = impl.indexer_select_post_process()
    impl._prepare_decode_sparse_indices(indices, torch.tensor([256, 256]))
    impl._execute_sparse_flash_attention_process(
        torch.ones(2, 1, 3),
        torch.ones(2, 1, 2),
        (torch.arange(6.0).reshape(1, 2, 1, 3), torch.arange(4.0).reshape(1, 2, 1, 2)),
        indices,
        SimpleNamespace(block_table=torch.tensor([[0]])),
        torch.tensor([2]),
        torch.tensor([2]),
    )
    tensors, addresses = probes.read(2, decode=True)
    assert addresses["layer=0 miss_tokens"].tolist() == [[0, 1, -1, -1]]
    assert tensors["layer=0 kv_nope"][0].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert "layer=0 physical_slots" in addresses
    assert "layer=0 physical_slots" not in tensors


def test_dummy_integer_weights_and_fingerprint_are_deterministic(worker):
    a, b = torch.nn.Module(), torch.nn.Module()
    for module, fill in ((a, 1), (b, 7)):
        module.register_parameter(
            "packed", torch.nn.Parameter(torch.full((5, 3), fill, dtype=torch.int32), requires_grad=False)
        )
        module.register_buffer("flags", torch.ones(2, dtype=torch.bool))
        module.register_parameter("weight", torch.nn.Parameter(torch.ones(2)))
        worker.deterministic_dummy_load(lambda *a: None, None, module, None)
    assert torch.equal(a.packed, b.packed)
    assert a.flags.all()  # Integer/bool buffers are metadata, not dummy weights.
    assert worker.weight_fingerprint(a) == worker.weight_fingerprint(b)
    b.packed[3, 1] += 1
    assert worker.weight_fingerprint(a) != worker.weight_fingerprint(b)


@pytest.mark.parametrize(
    "field,value",
    [("parity_decode_steps", 0), ("parity_q2_steps", 0), ("parity_draft_calls", 0), ("parity_transfers", [0] * 8)],
)
def test_completion_cannot_pass_without_live_coverage(worker, tmp_path, field, value):
    subject = worker.SFAParityWorker()
    subject.parity_group = SimpleNamespace(world_size=1, rank_in_group=0)
    subject.parity_decode_steps = subject.parity_q2_steps = subject.parity_draft_calls = 3
    subject.parity_transfers = [4] * 8
    subject.parity_directory = tmp_path
    setattr(subject, field, value)
    with pytest.raises(worker.ParityError, match="coverage|historical KV"):
        subject.parity_summary()


def test_wrong_tp_rank_or_world_size_is_not_a_valid_reference(parity):
    ref, val = make_step(), make_step()
    val["rank"] = 7
    with pytest.raises(parity.ParityError, match="incomparable rank"):
        parity.compare_step(ref, val, atol=0, rtol=0)
    val["rank"] = 0
    val["tp_size"] = 1
    with pytest.raises(parity.ParityError, match="incomparable tp_size"):
        parity.compare_step(ref, val, atol=0, rtol=0)


def test_one_rank_failure_is_seen_by_every_rank(parity, monkeypatch):
    statuses = [("after target", None)] * 8
    statuses[5] = ("after target", "rank=5 ParityError: layer=3 kv mismatch")

    def gather(output, value, *, group):
        output[:] = statuses

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    for rank in range(8):
        group = SimpleNamespace(world_size=8, rank_in_group=rank, cpu_group="gloo")
        with pytest.raises(parity.ParityError, match="rank=5.*layer=3"):
            parity.coordinated_check(lambda: None, group=group, phase="after target")


def test_rank_success_returns_its_own_local_result(parity, monkeypatch):
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value, **kwargs: output.__setitem__(slice(None), [("prepare", None)] * 8),
    )
    for rank in range(8):
        group = SimpleNamespace(world_size=8, rank_in_group=rank, cpu_group="gloo")
        assert parity.coordinated_check(lambda rank=rank: rank, group=group, phase="prepare") == rank


def test_diagnostic_phase_mismatch_fails_closed(parity, monkeypatch):
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value, **kwargs: output.__setitem__(slice(None), [("prepare", None), ("observe", None)]),
    )
    group = SimpleNamespace(world_size=2, rank_in_group=0, cpu_group="gloo")
    with pytest.raises(parity.ParityError, match="phases diverged"):
        parity.coordinated_check(lambda: None, group=group, phase="prepare")


def test_eight_rank_weight_manifests_are_isolated(worker, tmp_path):
    def subject(rank, graph):
        instance = worker.SFAParityWorker()
        instance.parity_rank, instance.parity_tp_size = rank, 8
        instance.parity_directory = tmp_path / f"rank-{rank}"
        instance.parity_is_graph = graph
        instance.device = torch.device("cpu")
        instance.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=4))
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([TinyDecoder() for _ in range(8)])
        for index, layer in enumerate(model.layers):
            layer.attn = torch.nn.Module()
            layer.attn.layer_name = f"layers.{index}.attn"
            layer.attn.impl = worker.AscendSFAImpl()
            layer.attn.impl.__dict__.update(fake_impl().__dict__)
            # Different shards need not have identical weights across ranks.
            with torch.no_grad():
                layer.input_layernorm.weight.fill_(rank + 1)
        instance.model_runner = SimpleNamespace(
            get_model=lambda: model,
            drafter=SimpleNamespace(model=TinyDecoder()),
            speculative_config=SimpleNamespace(num_speculative_tokens=1),
        )
        return instance

    for rank in range(8):
        eager = subject(rank, False)
        eager._prepare_probes()
        assert eager.model_runner.model_memory_usage > 0
    assert len(list(tmp_path.glob("rank-*/weights.json"))) == 8
    for rank in range(8):
        subject(rank, True)._prepare_probes()
    wrong_rank = subject(7, True)
    wrong_rank.parity_directory = tmp_path / "rank-0"
    with pytest.raises(worker.ParityError, match="weights differ"):
        wrong_rank._prepare_probes()


def test_eight_rank_forward_snapshots_compare_only_matching_shards(worker, monkeypatch, tmp_path):
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: None)), raising=False
    )

    def subject(rank, graph, *, corrupt=False):
        instance = worker.SFAParityWorker()
        instance.parity_rank, instance.parity_tp_size = rank, 8
        instance.parity_step = 2
        instance.parity_directory = tmp_path / f"rank-{rank}"
        instance.parity_directory.mkdir(exist_ok=True)
        instance.parity_is_graph = graph
        instance.parity_options = {"atol": 0, "rtol": 0}
        instance.parity_transfers = [0] * 8
        instance.parity_layers = []
        for layer in range(8):

            def read(rows, decode, layer=layer):
                value = rank + 1 + int(corrupt and layer == 3)
                return (
                    {f"layer={layer} input.hidden": torch.full((rows, 4), float(value))},
                    {f"layer={layer} miss_count": torch.tensor([2])},
                )

            instance.parity_layers.append(SimpleNamespace(index=layer, read=read))
        return instance

    def state(rank):
        result = make_step()
        result["rank"] = rank
        return result

    for rank in range(8):
        subject(rank, False)._observe_step(state(rank), torch.full((2, 4), float(rank + 1)), 0)
    assert len(list(tmp_path.glob("rank-*/step-000002.pt"))) == 8
    for rank in range(8):
        subject(rank, True)._observe_step(state(rank), torch.full((2, 4), float(rank + 1)), 1)
    with pytest.raises(worker.ParityError, match="rank=7 step=2 layer=3"):
        subject(7, True, corrupt=True)._observe_step(state(7), torch.full((2, 4), 8.0), 1)
    with pytest.raises(worker.ParityError, match="expected 1 root replay, got 0"):
        subject(7, True)._observe_step(state(7), torch.full((2, 4), 8.0), 0)
