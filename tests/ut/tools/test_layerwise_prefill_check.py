# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the opt-in three-pass diagnostic, without vLLM/NPU imports."""

import ast
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[3] / "tools"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECK = load_tool("layerwise_prefill_check")
PROBE = load_tool("layerwise_prefill_probe")


def args():
    return CHECK.parser().parse_args([])


def test_full_model_three_separate_roles():
    for stage, role in zip(CHECK.STAGES, ("kv_both", "kv_producer", "kv_consumer"), strict=True):
        options = CHECK.engine_options(args(), 15000, stage)
        assert options["load_format"] == "safetensors"
        assert "hf_overrides" not in options
        assert "speculative_config" not in options
        assert options["tensor_parallel_size"] == 8
        assert options["gpu_memory_utilization"] == 0.96
        assert options["max_model_len"] == 16384
        assert options["kv_transfer_config"]["kv_role"] == role
        assert options["enforce_eager"]


@pytest.mark.parametrize("stage", CHECK.STAGES)
def test_fixed_model_length_boundary(stage):
    options = CHECK.engine_options(args(), 16384 - args().output_tokens, stage)
    assert options["max_model_len"] == 16384
    with pytest.raises(ValueError, match="exceeds max_model_len=16384"):
        CHECK.engine_options(args(), 16384 - args().output_tokens + 1, stage)


def test_stages_only_read_prefill_archive_and_isolate_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", "/old/mooncake.yaml")
    monkeypatch.setenv("VLLM_ASCEND_SFA_FULL_GRAPH", "1")
    for stage in CHECK.STAGES:
        env = CHECK.stage_environment(args(), tmp_path, stage)
        assert "LMCACHE_CONFIG_FILE" not in env
        assert env["VLLM_ASCEND_SFA_FULL_GRAPH"] == "0"
        assert env["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] == str(stage == "prefill").lower()
        extra = json.loads(env["LMCACHE_EXTRA_CONFIG"])
        source = "prefill" if stage == "decode" else stage
        assert Path(extra["validation_archive"]) == tmp_path / source / "archive"
        assert extra["validation_read_only"] == (stage == "decode")


def test_statistics_are_population_variance_and_abs_diff():
    stats = CHECK.tensor_statistics(torch.tensor([1.0, -3.0]), torch.tensor([2.0, -1.0]))
    assert stats["baseline"]["mean"] == -1
    assert stats["baseline"]["variance"] == 4
    assert stats["candidate_abs"]["mean"] == 1.5
    assert stats["diff"]["mean"] == 1.5
    assert stats["abs_diff"]["mean"] == 1.5
    assert stats["abs_diff"]["variance"] == 0.25


def test_tiny_difference_reported_not_rejected():
    base = torch.ones(10, dtype=torch.float64)
    stats = CHECK.tensor_statistics(base, base + 1e-9)
    assert 0 < stats["abs_diff"]["mean"] < 1e-7


def test_bfloat_subtraction_widens_before_stats():
    base = torch.tensor([0.5, 1], dtype=torch.bfloat16)
    other = torch.tensor([0.50390625, 1.0078125], dtype=torch.bfloat16)
    assert CHECK.tensor_statistics(base, other)["abs_diff"]["mean"] == 0.005859375


def test_nonfinite_count_preserved_and_json_safe():
    stats = CHECK.tensor_statistics(torch.tensor([1.0, float("nan")]), torch.tensor([float("inf"), 2.0]))
    assert stats["baseline"]["nonfinite"] == 1
    assert stats["abs_diff"]["nonfinite"] == 2
    assert stats["abs_diff"]["mean"] is None
    json.dumps(stats, allow_nan=False)


def test_streaming_moments_match_whole_tensor():
    values = torch.tensor([1000.0, 1000.125, 1000.25, -4.0], dtype=torch.float64)
    moments = CHECK.Moments()
    moments.add(values[:1])
    moments.add(values[1:])
    assert moments.result()["mean"] == pytest.approx(float(values.mean()))
    assert moments.result()["variance"] == pytest.approx(float(values.var(correction=0)))


def test_token_divergence_and_length():
    assert CHECK.first_difference([1, 2], [1, 2]) is None
    assert CHECK.first_difference([1, 2], [1, 3]) == 1
    assert CHECK.first_difference([1], [1, 2]) == 1


def save_rows(path, positions, values):
    torch.save({"positions": torch.tensor(positions), "values": torch.tensor(values)}, path)
    return path


def test_compare_matches_logical_positions_not_record_order(tmp_path):
    a = save_rows(tmp_path / "a.pt", [0, 1, 2], [[1.0], [2.0], [3.0]])
    b = save_rows(tmp_path / "b.pt", [2, 0], [[3.1], [1.0]])
    report = CHECK.compare_rows([a], [b], 0, 3)
    assert report["matched_rows"] == 2
    assert report["baseline_only_rows"] == 1
    assert report["stats"]["abs_diff"]["mean"] == pytest.approx(0.05)


def test_no_decode_comparison_after_input_divergence(tmp_path):
    a = save_rows(tmp_path / "a.pt", [10, 11], [[1.0], [200.0]])
    b = save_rows(tmp_path / "b.pt", [10, 11], [[1.0], [-500.0]])
    report = CHECK.compare_rows([a], [b], 10, 11)
    assert report["matched_rows"] == 1
    assert report["stats"]["abs_diff"]["max"] == 0


def test_duplicate_writes_are_not_silently_overwritten(tmp_path):
    file = save_rows(tmp_path / "a.pt", [0, 0], [[1.0], [2.0]])
    with pytest.raises(RuntimeError, match="Duplicate"):
        CHECK.load_rows([file], 0, 2)


def test_probe_uses_physical_slots_and_independent_tables():
    cache = torch.arange(24).reshape(3, 2, 1, 4)
    slots = PROBE.slots_for_positions(cache, torch.tensor([[2, 0, 1]]), torch.tensor([0, 1, 2]))
    assert slots.tolist() == [4, 5, 0]
    assert PROBE.rows_from_slots(cache, slots).flatten().tolist() == list(range(16, 24)) + list(range(4))


def test_probe_rejects_invalid_live_slot():
    with pytest.raises(RuntimeError, match="invalid live slots"):
        PROBE.rows_from_slots(torch.zeros(2, 2, 1, 4), torch.tensor([-1]))


def test_loaded_rows_dedup_but_current_writes_retained(tmp_path):
    rec = PROBE.KVRecorder(tmp_path, 0, 3)
    cache = torch.arange(16).reshape(2, 2, 1, 4)
    for _ in range(2):
        rec.save("layer0", "nope", "loaded", torch.tensor([0, 1, 3]), cache, torch.tensor([0, 1, 3]))
    rec.save("layer0", "nope", "current", torch.tensor([3]), cache, torch.tensor([3]))
    index = CHECK.read_index(tmp_path)
    loaded = index[("rank0", "layer0", "nope", "loaded")]
    assert len(loaded) == 1
    assert CHECK.load_rows(loaded, 0, 4)[0].tolist() == [0, 1]
    assert len(index[("rank0", "layer0", "nope", "current")]) == 1


def test_reload_cannot_pass_by_recomputing_prompt(tmp_path):
    decode = tmp_path / "decode"
    decode.mkdir()
    CHECK.write_json(decode / "output.json", {"num_cached_tokens": 0})
    with pytest.raises(RuntimeError, match="BOTH"):
        CHECK.validate_reload(tmp_path, 1024)
    with (decode / "archive_reads_1.jsonl").open("w") as file:
        for group in (0, 1):
            file.write(json.dumps({"key": str(group), "kv_group": group}) + "\n")
    with pytest.raises(RuntimeError, match="too short"):
        CHECK.validate_reload(tmp_path, 1024)
    CHECK.write_json(decode / "output.json", {"num_cached_tokens": 1023})
    assert CHECK.validate_reload(tmp_path, 1024)["cached_tokens"] == 1023


@pytest.fixture
def connector_module(monkeypatch):
    """Load real connector with CPU allocator/interface substitutes only."""
    for name in ("lmcache.v1.memory_management", "lmcache.v1.storage_backend.connector.base_connector"):
        module = ModuleType(name)
        if name.endswith("memory_management"):
            module.MemoryFormat = lambda value: value
        else:
            module.RemoteConnector = object
        monkeypatch.setitem(sys.modules, name, module)
    return load_tool("layerwise_prefill_store")


class Memory:
    def __init__(self, raw):
        self.raw_data = raw
        self.meta = SimpleNamespace(valid_tokens=2, cached_positions=torch.tensor([3, 4]))
        self.released = 0

    @property
    def byte_array(self):
        return memoryview(self.raw_data.numpy())

    def get_shapes(self):
        return [torch.Size([self.raw_data.numel()])]

    def get_dtypes(self):
        return [torch.uint8]

    def get_memory_format(self):
        return SimpleNamespace(value=1)

    def ref_count_down(self):
        self.released += 1


def make_connector(module, tmp_path, read_only=False, allocation_delta=0):
    extra = {
        "validation_archive": str(tmp_path / "archive"),
        "validation_stage_dir": str(tmp_path / "stage"),
        "validation_read_only": read_only,
    }
    allocations = []

    def allocate(shapes, dtypes, fmt, busy_loop):
        obj = Memory(torch.empty(shapes[0].numel() + allocation_delta, dtype=torch.uint8))
        allocations.append(obj)
        return obj

    config = SimpleNamespace(get_extra_config_value=lambda k, default=None: extra.get(k, default))
    connector = module.ValidationFileConnector(None, SimpleNamespace(allocate=allocate), config)
    key = SimpleNamespace(to_string=lambda: "model@8@0@abcdef@bf16@1@7", kv_group=1, layer_id=7, worker_id=0)
    return connector, key, allocations


def test_archive_roundtrip_preserves_flat_valid_tokens_and_bytes(connector_module, tmp_path):
    conn, key, allocated = make_connector(connector_module, tmp_path)
    source = Memory(torch.arange(12, dtype=torch.uint8))
    asyncio.run(conn.put(key, source))
    assert source.released == 0  # InstrumentedRemoteConnector owns the release.
    result = asyncio.run(conn.get(key))
    assert torch.equal(result.raw_data, source.raw_data)
    assert result.meta.valid_tokens == 2
    assert result.meta.cached_positions.tolist() == [3, 4]
    assert result is allocated[0]
    assert conn.exists_sync(key)
    assert conn.requires_put_completion()


def test_archive_range_positions_are_weights_only_loadable(connector_module, tmp_path):
    conn, key, _ = make_connector(connector_module, tmp_path)
    source = Memory(torch.arange(12, dtype=torch.uint8))
    source.meta.cached_positions = range(10, 12)
    asyncio.run(conn.put(key, source))
    restored = asyncio.run(conn.get(key))
    assert restored.meta.cached_positions.tolist() == [10, 11]


def test_readonly_archive_cannot_be_overwritten(connector_module, tmp_path):
    conn, key, _ = make_connector(connector_module, tmp_path, read_only=True)
    source = Memory(torch.zeros(12, dtype=torch.uint8))
    with pytest.raises(RuntimeError, match="must not modify"):
        asyncio.run(conn.put(key, source))
    assert source.released == 0  # Including on the error path.
    assert not list((tmp_path / "archive").glob("*.pt"))


def test_corrupted_archive_is_rejected(connector_module, tmp_path):
    conn, key, _ = make_connector(connector_module, tmp_path)
    asyncio.run(conn.put(key, Memory(torch.arange(12, dtype=torch.uint8))))
    path = conn.path_for(key)
    payload = torch.load(path, weights_only=True)
    payload["raw"][0] = 255
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="checksum"):
        asyncio.run(conn.get(key))


def test_reload_allocation_failure_releases_memory(connector_module, tmp_path):
    conn, key, allocated = make_connector(connector_module, tmp_path, allocation_delta=1)
    asyncio.run(conn.put(key, Memory(torch.arange(12, dtype=torch.uint8))))
    with pytest.raises(RuntimeError, match="allocation size"):
        asyncio.run(conn.get(key))
    assert allocated[0].released == 1


@pytest.mark.parametrize("read_only", [False, True])
def test_real_instrumented_put_releases_exactly_once(connector_module, tmp_path, read_only):
    source_path = TOOLS.parents[1] / "LMCache/lmcache/v1/storage_backend/connector/instrumented_connector.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "put")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = {"time": SimpleNamespace(perf_counter=lambda: 0), "logger": Mock()}
    exec(compile(ast.fix_missing_locations(module), "<actual-instrumented-put>", "exec"), namespace)
    conn, key, _ = make_connector(connector_module, tmp_path, read_only)
    source = Memory(torch.arange(8, dtype=torch.uint8))
    source.get_size = lambda: 8
    wrapper = SimpleNamespace(_connector=conn, _stats_monitor=Mock(), name="test")
    if read_only:
        with pytest.raises(RuntimeError):
            asyncio.run(namespace["put"](wrapper, key, source))
    else:
        asyncio.run(namespace["put"](wrapper, key, source))
    assert source.released == 1


def test_real_worker_hooks_observe_writes_and_loads(monkeypatch, tmp_path):
    sfa = ModuleType("vllm_ascend.attention.sfa_v1")
    sfa._dsa_indexer_layer_name = lambda name: name + ".index"
    sfa.wait_for_kv_layer_from_connector = lambda name, **kwargs: None
    index = torch.arange(4, dtype=torch.float32).reshape(2, 2, 1, 1) + 200
    parts = tuple(torch.arange(4, dtype=torch.float32).reshape(2, 2, 1, 1) + i * 100 for i in range(2))

    class Impl:
        has_indexer = True
        dsa_shrink_latent = 0

        def forward(self, layer, hidden, caches, meta, **kwargs):
            sfa.wait_for_kv_layer_from_connector(layer)
            sfa.wait_for_kv_layer_from_connector(layer + ".index")
            caches[0].view(-1)[3] = 90
            caches[1].view(-1)[3] = 91
            index.view(-1)[2] = 92
            return "output"

    sfa.AscendSFAImpl = Impl
    attention = ModuleType("vllm_ascend.attention")
    attention.sfa_v1 = sfa
    context = ModuleType("vllm.forward_context")
    context.get_forward_context = lambda: SimpleNamespace(
        virtual_engine=0, no_compile_layers={"layer.index": SimpleNamespace(kv_cache=[index])}
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention", attention)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", context)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    worker = PROBE.PrefillValidationWorker()
    worker.rank = 0
    worker.install_prefill_validation(str(tmp_path), 3)
    meta = SimpleNamespace(
        req_ids=["r"],
        num_actual_tokens=1,
        query_start_loc_cpu=torch.tensor([0, 1]),
        seq_lens_cpu=torch.tensor([3]),
        slot_mapping=torch.tensor([3]),
        indexer_slot_mapping=torch.tensor([2]),
        block_table=torch.tensor([[0, 1]]),
        indexer_block_table=torch.tensor([[1, 0]]),
    )
    assert Impl().forward("layer", None, parts, meta) == "output"
    manifest = CHECK.read_index(tmp_path)
    for part, expected in (("nope", 90), ("pe", 91), ("index", 92)):
        positions, values = CHECK.load_rows(manifest[("rank0", "layer", part, "current")], 0, 3)
        assert positions.tolist() == [2]
        assert values.reshape(-1).tolist() == [expected]
    positions, values = CHECK.load_rows(manifest[("rank0", "layer", "index", "loaded")], 0, 3)
    assert positions.tolist() == [0, 1]
    assert values.reshape(-1).tolist() == [202, 203]


def test_three_pass_report_keeps_small_differences_and_real_output_divergence(connector_module, tmp_path):
    for stage, tokens in (("baseline", [7, 8]), ("prefill", [7]), ("decode", [7, 9])):
        directory = tmp_path / stage
        directory.mkdir()
        CHECK.write_json(
            directory / "output.json",
            {"token_ids": tokens, "prompt_sha256": "same", "num_hidden_layers": 1, "num_cached_tokens": 3},
        )
        rec = PROBE.KVRecorder(directory, 0, 4)
        for part in ("nope", "pe", "index"):
            cache = torch.ones(3, 2, 1, 1, dtype=torch.float64)
            if stage == "prefill":
                cache += 1e-9
            positions = torch.arange(4) if stage != "decode" else torch.tensor([3])
            rec.save("layer", part, "current", positions, cache, positions)
            if stage == "decode":
                rec.save("layer", part, "loaded", torch.arange(3), cache, torch.arange(3))
            if stage != "prefill":
                rec.save("layer", part, "current", torch.tensor([4]), cache, torch.tensor([4]))
    CHECK.write_json(tmp_path / "prompt.json", {"length": 4, "sha256": "same"})
    for stage in ("baseline", "prefill"):
        conn, key, _ = make_connector(connector_module, tmp_path / stage)
        for group in (0, 1):
            key.kv_group = group
            key.to_string = lambda group=group: f"group{group}"
            asyncio.run(conn.put(key, Memory(torch.arange(12, dtype=torch.uint8))))
    CHECK.seal_archive(tmp_path)
    sealed = json.loads((tmp_path / "prefill/archive_manifest.json").read_text())
    with (tmp_path / "decode/archive_reads_1.jsonl").open("w") as file:
        for group in (0, 1):
            file.write(
                json.dumps({"key": f"group{group}", "sha256": sealed[f"group{group}"], "kv_group": group}) + "\n"
            )
    result = CHECK.analyse(tmp_path, 1)
    assert result["tokens_equal"] is False
    assert result["first_different_output_token_index"] == 1
    assert result["decode_compare_position_exclusive"] == 5
    assert result["structural_errors"] == []
    assert (tmp_path / "kv_statistics.csv").is_file()
    written = [row for row in result["kv"] if row["comparison"] == "prefill_written"]
    assert all(0 < row["stats"]["abs_diff"]["mean"] < 1e-7 for row in written)
