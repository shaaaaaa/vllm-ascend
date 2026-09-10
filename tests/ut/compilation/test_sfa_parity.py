# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of comparison/probe logic, not evidence of real NPU/model parity."""

import hashlib
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
        "vllm_ascend.worker.worker": {"NPUWorker": type("NPUWorker", (), {})},
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


@pytest.fixture
def modelslim(worker, monkeypatch):
    # Import the real parser/metadata/lookup code; only its hardware/model
    # dependencies are stubbed. No fake successful model inference here.
    dependencies = {
        "vllm.config": {"get_current_vllm_config": lambda: None},
        "vllm.logger": {"logger": SimpleNamespace(info=lambda *args: None)},
        "vllm.model_executor.layers.attention_layer_base": {"AttentionLayerBase": type("Attention", (), {})},
        "vllm.model_executor.layers.fused_moe": {"FusedMoE": type("MoE", (), {})},
        "vllm.model_executor.layers.linear": {"LinearBase": type("Linear", (), {})},
        "vllm.model_executor.layers.quantization": {"register_quantization_config": lambda name: lambda cls: cls},
        "vllm.model_executor.layers.quantization.base_config": {
            "QuantizationConfig": object,
            "QuantizeMethodBase": object,
        },
        "vllm.model_executor.layers.vocab_parallel_embedding": {
            "UnquantizedEmbeddingMethod": object,
            "VocabParallelEmbedding": type("Embedding", (), {}),
        },
        "vllm.model_executor.models.utils": {"WeightsMapper": object},
        "vllm_ascend.utils": {"ASCEND_QUANTIZATION_METHOD": "ascend", "calc_split_factor": lambda *args: 1},
        "vllm_ascend.quantization.methods": {"get_scheme_class": lambda *args: None},
    }
    for name, attributes in dependencies.items():
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/quantization/modelslim_config.py"
    spec = importlib.util.spec_from_file_location("vllm_ascend.quantization.modelslim_config", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("source", [8, 78, 80])
@pytest.mark.parametrize("head_quant", ["FLOAT", "W8A8"])
def test_truncated_mtp_uses_original_quantization_not_decoder_layer_eight(worker, modelslim, source, head_quant):
    description = {f"model.layers.{i}.self_attn.q_proj.weight": "W8A8" for i in range(9)}
    description.update(
        {
            "fa_quant_type": "C8",
            "indexer_quant_type": "INT8",
            "is_rot_used": True,
            "model.layers.8.indexer.quant_type": "INT8",  # stale dropped target metadata
            "model.layers.8.decoder_only.weight": "FLOAT",
            f"model.layers.{source}.shared_head.head.weight": head_quant,
            f"model.layers.{source}.self_attn.q_proj.weight": "W8A8_DYNAMIC",
            f"model.layers.{source}.mlp.experts.0.gate_proj.weight_packed": "W4A8_DYNAMIC",
            f"model.layers.{source}.mlp.experts.0.up_proj.weight_packed": "W4A8_DYNAMIC",
            f"model.layers.{source}.fa_k.scale": "C8",
        }
    )
    before = description.copy()
    if source != 8:
        with pytest.raises(KeyError, match=r"model.layers.8.head.weight"):
            modelslim.get_linear_quant_type(description, "model.layers.8.head", {})
    remapped = worker.remap_mtp_quant_description(description, source, 1)
    config = modelslim.AscendModelSlimConfig(remapped)

    def get_type(prefix, packed=None):
        return modelslim.get_linear_quant_type(config.quant_description, prefix, packed or {})

    assert get_type("model.layers.8.head") == head_quant
    assert config.is_layer_skipped_ascend("model.layers.8.head") == (head_quant == "FLOAT")
    assert get_type("model.layers.8.self_attn.q_proj") == "W8A8_DYNAMIC"
    assert (
        get_type("model.layers.8.mlp.experts.0.gate_up_proj", {"gate_up_proj": ["gate_proj", "up_proj"]})
        == "W4A8_DYNAMIC"
    )
    assert config.is_fa_quant_layer("model.layers.8.self_attn")
    assert config.is_indexer_quant_layer("model.layers.8.self_attn") == (source == 8)
    if source != 8:
        assert "model.layers.8.decoder_only.weight" not in config.quant_description
    for i in range(8):
        assert get_type(f"model.layers.{i}.self_attn.q_proj") == "W8A8"
    assert config.quant_description["is_rot_used"] is True
    assert description == before  # no mutation of the checkpoint description


def test_mtp_remap_copies_all_draft_layers_without_prefix_collisions(worker):
    description = {
        "model.layers.78.head.weight": "FLOAT",
        "model.layers.79.head.weight": "W8A8",
        "model.layers.780.head.weight": "DO_NOT_COPY",
        "model.layers.8.stale.weight": "FLOAT",
        "model.layers.9.stale.weight": "FLOAT",
    }
    remapped = worker.remap_mtp_quant_description(description, 78, 2)
    assert remapped["model.layers.8.head.weight"] == "FLOAT"
    assert remapped["model.layers.9.head.weight"] == "W8A8"
    assert remapped["model.layers.780.head.weight"] == "DO_NOT_COPY"
    assert "model.layers.8.stale.weight" not in remapped
    assert "model.layers.9.stale.weight" not in remapped


@pytest.mark.parametrize("source,count", [(7, 1), (True, 1), (78.0, 1), (78, 0), (78, True), (78, 1)])
def test_bad_or_missing_original_mtp_quantization_fails_closed(worker, source, count):
    # Even an existing destination head must not hide absent original MTP data.
    description = {"model.layers.8.head.weight": "FLOAT"}
    with pytest.raises(ValueError):
        worker.remap_mtp_quant_description(description, source, count)
    assert description == {"model.layers.8.head.weight": "FLOAT"}


@pytest.mark.parametrize("graph", [False, True])
def test_worker_remaps_before_model_construction_in_both_modes(worker, modelslim, monkeypatch, tmp_path, graph):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    checkpoint = '{"num_hidden_layers": 80}'
    (model_dir / "config.json").write_text(checkpoint)
    original = modelslim.AscendModelSlimConfig({"model.layers.80.shared_head.head.weight": "FLOAT"})
    target = SimpleNamespace(
        model=str(model_dir), hf_config=SimpleNamespace(num_hidden_layers=8), enforce_eager=not graph
    )
    subject = worker.SFAParityWorker()
    subject.model_config = target
    subject.vllm_config = SimpleNamespace(
        model_config=target,
        load_config=SimpleNamespace(load_format="dummy"),
        quant_config=original,
        speculative_config=SimpleNamespace(
            num_speculative_tokens=1,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="deepseek_mtp", num_nextn_predict_layers=1)
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8, data_parallel_size=1, pipeline_parallel_size=1, enable_expert_parallel=False
        ),
        additional_config={"sfa_parity": {"mode": "graph" if graph else "eager", "reference": str(tmp_path)}},
    )
    monkeypatch.setattr(worker.envs, "VLLM_ASCEND_SFA_FULL_GRAPH", graph)
    monkeypatch.setattr(worker.envs, "VLLM_ASCEND_SFA_STAGED_GRAPH", graph)
    monkeypatch.setattr(worker, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=8))
    subject._check = lambda check, phase: check()  # coordination is covered by the real Gloo tests
    original_load = lambda *args: None
    monkeypatch.setattr(worker.DummyModelLoader, "load_weights", original_load, raising=False)

    class StopBeforeHardware(RuntimeError):
        pass

    def start_model(self):
        config = self.vllm_config.quant_config
        assert config is not original
        assert modelslim.get_linear_quant_type(config.quant_description, "model.layers.8.head", {}) == "FLOAT"
        assert self.model_config.hf_config.num_hidden_layers == 8
        assert self.vllm_config.speculative_config.num_speculative_tokens == 1
        raise StopBeforeHardware

    monkeypatch.setattr(worker.NPUWorker, "load_model", start_model, raising=False)
    with pytest.raises(StopBeforeHardware):
        subject.load_model()
    assert worker.DummyModelLoader.load_weights is original_load
    assert "model.layers.8.head.weight" not in original.quant_description
    assert (model_dir / "config.json").read_text() == checkpoint


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int8, torch.int32, torch.int64])
def test_fingerprint_internal_storage_never_uses_tensor_or_storage_cpu(parity, monkeypatch, dtype):
    payload = torch.arange(48).to(dtype).reshape(3, 16)

    class InternalTensor:
        shape, device = payload.shape, SimpleNamespace(type="npu")

        def __init__(self):
            self.dtype = dtype

        def detach(self):
            return self

        def is_contiguous(self):
            return True

        def storage_offset(self):
            return 0

        def untyped_storage(self):
            return payload.untyped_storage()

        def numel(self):
            return payload.numel()

        def element_size(self):
            return payload.element_size()

        def forbidden(self, *args, **kwargs):
            raise AssertionError("No slicing, reshaping, conversion or Tensor.cpu on internal-format weights")

        __getitem__ = cpu = split = contiguous = reshape = view = forbidden

    def forbidden_storage_cpu(*args):
        raise AssertionError("Storage.cpu would reintroduce the NPU format conversion")

    monkeypatch.setattr(torch.UntypedStorage, "cpu", forbidden_storage_cpu)
    stub = ModuleType("torch_npu")
    stub.get_npu_format = lambda tensor: 29
    monkeypatch.setitem(sys.modules, "torch_npu", stub)
    monkeypatch.setattr(parity, "WEIGHT_HASH_CHUNK_BYTES", 7)
    model = SimpleNamespace(state_dict=lambda: {"packed": InternalTensor()})
    expected = hashlib.sha256(f"packed:{dtype}:{tuple(payload.shape)}:format=29".encode())
    expected.update(payload.reshape(-1).view(torch.uint8).numpy().tobytes())
    before = parity.weight_fingerprint(model)
    assert before == expected.hexdigest()
    payload.reshape(-1)[-1] += 1  # Last byte/chunk cannot be omitted.
    assert parity.weight_fingerprint(model) != before


@pytest.mark.parametrize("kind", ["padding", "offset", "transpose"])
def test_internal_storage_hash_rejects_padding_and_partial_views(parity, kind):
    payload = torch.arange(64, dtype=torch.int32).reshape(8, 8)
    value = {"padding": payload[:7], "offset": payload[1:], "transpose": payload.T}[kind]
    with pytest.raises(parity.ParityError, match="complete unpadded storage"):
        parity._weight_bytes_on_cpu(value, 29)


@pytest.mark.parametrize("shape", [(), (0,), (0, 3), (2, 0), (3, 5)])
def test_fingerprint_handles_scalar_empty_and_noncontiguous_cpu_values(parity, shape):
    payload = torch.ones(shape, dtype=torch.bfloat16)
    if len(shape) == 2:
        payload = payload.T
    model = torch.nn.Module()
    model.register_buffer("value", payload)
    same = torch.nn.Module()
    same.register_buffer("value", payload.contiguous())
    assert parity.weight_fingerprint(model) == parity.weight_fingerprint(same)


def test_fingerprint_error_keeps_tensor_name_and_metadata(parity, monkeypatch):
    model = torch.nn.Module()
    model.register_buffer("broken_weight", torch.ones((2, 4), dtype=torch.int32))

    def fail(*args):
        raise RuntimeError("Identity/TransData failed")

    monkeypatch.setattr(parity, "_weight_bytes_on_cpu", fail)
    with pytest.raises(parity.ParityError, match=r"name=broken_weight shape=\(2, 4\) dtype=torch.int32.*TransData"):
        parity.weight_fingerprint(model)


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
