# SPDX-License-Identifier: Apache-2.0
"""CPU checks of full tensor archives, real-consumer hooks and coverage."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    layout = NS(
        install_local_merged_layout=lambda: None,
        validate_local_merged_engine=lambda engine: {
            "layout": "merged",
            "remote_url": None,
            "native_mooncake": False,
            "production_transfer_unchanged": True,
        },
    )
    monkeypatch.setitem(sys.modules, "layerwise_prefill_correctness_layout", layout)
    path = ROOT / "tools/layerwise_prefill_correctness_worker.py"
    spec = importlib.util.spec_from_file_location("correctness_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_off_archives_every_element_and_on_loads_without_second_archive(worker, tmp_path):
    values = torch.arange(6000, dtype=torch.float32).reshape(1000, 6)
    off = worker.TensorArchive(tmp_path / "off", 3)
    kwargs = dict(step=0, layer=0, kind="decoder", name="input", span=(0, 1000))
    off.record(values, **kwargs)
    off.close()
    manifest = json.loads(off.index_path.read_text())
    assert "sha256" not in manifest
    assert off.file_count == 1 and off.archived_bytes > values.numel() * values.element_size()
    saved = torch.load(tmp_path / "off" / manifest["path"], weights_only=True)
    assert torch.equal(saved, values)
    values[-1, -1] = -999  # A changed last element must reach the online comparator.
    seen = []

    def compare(reference, actual):
        seen.append((reference[-1, -1].item(), actual[-1, -1].item(), actual.numel()))
        return {"comparable": True, "rmse": 1.0}

    on = worker.TensorArchive(tmp_path / "on", 3, compare=compare)
    on.record(values, **kwargs)
    on.close()
    assert seen == [(5999, -999, 6000)]
    assert on.file_count == on.archived_bytes == 0
    record = json.loads(on.index_path.read_text())
    assert record["path"] is None and record["comparison"]["comparable"]


def test_archive_counts_nonfinite_without_rejecting_valid_negative_infinity(worker, tmp_path):
    archive = worker.TensorArchive(tmp_path / "off", 0)
    archive.record(
        torch.tensor([float("-inf"), float("inf"), float("nan"), 1.0]),
        step=0,
        layer=0,
        kind="indexer",
        name="weights",
        span=(0, 4),
    )
    archive.close()
    assert next(iter(archive.records.values()))["nonfinite"] == 3
    assert archive.errors == []


def test_on_worker_reads_external_off_during_inference_and_keeps_archive_unchanged(worker, tmp_path):
    old = tmp_path / "old"
    identity = dict(step=0, layer=0, kind="decoder", name="input", span=(0, 2))
    off = worker.TensorArchive(old / "off", 1)
    off.record(torch.tensor([1.0, 2.0]), **identity)
    off.close()
    (off.root / "result.json").write_text(json.dumps({"case": "off", "completed": True}))
    (off.root / "environment.json").write_text(
        json.dumps({"VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": "false", "LMCACHE_STORE_ASYNC": "false"})
    )
    renamed = old / "renamed_baseline"
    off.root.rename(renamed)
    before = {str(path.relative_to(old)): path.read_bytes() for path in old.rglob("*") if path.is_file()}
    current = tmp_path / "current"
    current.mkdir()
    (current / "off_reference.json").write_text(json.dumps({"schema": 1, "off_dir": str(renamed.resolve())}))
    on = worker.TensorArchive(current / "on", 1)
    on.record(torch.tensor([1.0, 2.5]), **identity)
    record = next(iter(on.records.values()))
    assert record["comparison"]["abs_diff"]["max"] == 0.5
    assert record["path"] is None and not on.errors
    on.close()
    assert not (current / "off").exists()
    assert before == {str(path.relative_to(old)): path.read_bytes() for path in old.rglob("*") if path.is_file()}


def test_archive_rejects_duplicates_and_surfaces_disk_failure(worker, tmp_path, monkeypatch):
    archive = worker.TensorArchive(tmp_path / "off", 0)
    kwargs = dict(step=0, layer=0, kind="sfa", name="input", span=(0, 1))
    archive.record(torch.ones(1), **kwargs)
    with pytest.raises(RuntimeError, match="Duplicate"):
        archive.record(torch.ones(1), **kwargs)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        archive.record(torch.ones(1), **dict(kwargs, name="output"))
    assert archive.errors and archive.file_count == 1
    archive.close()


def test_physical_bank_addresses_normalize_to_identical_logical_rows(worker):
    logical = torch.arange(16).reshape(8, 1, 2)
    for mapping in ([2, 0], [1, 3]):
        cache = torch.zeros((4, 4, 1, 2), dtype=torch.long)
        table = torch.tensor([mapping])
        slots = worker.slots_for_positions(cache, table, 0, 8)
        cache.view(-1, 1, 2)[slots] = logical
        assert torch.equal(worker.rows_from_slots(cache, slots), logical)
    with pytest.raises(RuntimeError, match="invalid slots"):
        worker.rows_from_slots(cache, torch.tensor([-1]))


@pytest.mark.parametrize("valid_rows", [0, 1])
def test_real_online_statistics_exclude_padding_but_off_saves_full_tensor(worker, tmp_path, valid_rows):
    kwargs = dict(step=0, layer=0, kind="sfa", name="output", span=(0, 5), valid_rows=valid_rows)
    off = worker.TensorArchive(tmp_path / "off", 0)
    reference = torch.tensor([[1.0, 2.0], [900.0, 900.0]])
    off.record(reference, **kwargs)
    off.close()
    off_record = next(iter(off.records.values()))
    assert torch.equal(torch.load(tmp_path / "off" / off_record["path"], weights_only=True), reference)
    # Default constructor imports the actual public comparison implementation.
    on = worker.TensorArchive(tmp_path / "on", 0, save_on_tensors=True)
    on.record(torch.tensor([[1.0, 2.0], [float("nan"), float("-inf")]]), **kwargs)
    on.close()
    record = next(iter(on.records.values()))
    assert record["shape"] == [2, 2] and record["numel"] == 4
    assert record["comparison_shape"] == [valid_rows, 2]
    assert record["comparison_numel"] == valid_rows * 2
    assert record["nonfinite"] == 2 and record["comparison_nonfinite"] == 0
    assert record["comparison"]["comparable"]
    assert record["comparison"]["new_nonfinite"] == record["comparison"]["mismatched"] == 0
    assert on.file_count == 1 and on.errors == []


def test_valid_rows_follow_cp_metadata_and_distinguish_global_positions(worker):
    meta = NS(
        num_actual_tokens=5,
        num_input_tokens=5,
        dsa_cp_context=NS(num_tokens_pad=8, local_start=4, local_end=5, local_end_with_pad=6),
    )
    assert worker.valid_tensor_rows(meta, torch.empty(2, 8)) == 1
    assert worker.valid_tensor_rows(meta, torch.empty(5, 8)) == 5
    assert worker.valid_tensor_rows(meta, torch.empty(8), global_only=True) == 5
    meta.dsa_cp_context.local_start = 6
    meta.dsa_cp_context.local_end_with_pad = 8
    assert worker.valid_tensor_rows(meta, torch.empty(2, 8)) == 0
    with pytest.raises(RuntimeError, match="neither global"):
        worker.valid_tensor_rows(meta, torch.empty(3, 8))


def test_ambiguous_cp_shape_and_decode_classified_prefill_fail_closed(worker):
    meta = NS(
        num_actual_tokens=1,
        num_input_tokens=1,
        seq_lens_cpu=[8],
        query_start_loc_cpu=[0, 1],
        num_decode_tokens=1,
        dsa_cp_context=NS(num_tokens_pad=8, local_start=7, local_end=1, local_end_with_pad=8),
    )
    with pytest.raises(RuntimeError, match="Ambiguous"):
        worker.valid_tensor_rows(meta, torch.empty(1, 8))
    assert worker.valid_tensor_rows(meta, torch.empty(1, 8), prefer_local=True) == 0
    assert worker.valid_tensor_rows(meta, torch.empty(1, 8), prefer_local=False) == 1
    with pytest.raises(RuntimeError, match="Decode-classified"):
        worker.single_request_span(meta, 8)
    assert worker.single_request_span(meta, 7) is None


def _runtime(worker, monkeypatch, case_dir, *, map_ids=(2, 0), compare=None, h2d_source="merged"):
    shared = {}
    operations = NS(
        npu_lightning_indexer=lambda **kw: torch.zeros((kw["query"].shape[0], 1, 2), dtype=torch.long),
        npu_sparse_flash_attention=lambda **kw: kw["query"].clone(),
    )
    monkeypatch.setattr(torch.ops, "_C_ascend", operations)
    context = NS(attn_metadata={})

    class SFAImpl:
        def __init__(self, index):
            self.has_indexer = index == 0
            self.skip_topk = index != 0
            self.use_sparse_c8_indexer = False

        def indexer_select_post_process(self, x, q_c, kv_cache, attn_metadata):
            return operations.npu_lightning_indexer(
                query=q_c.unsqueeze(1),
                weights=torch.ones((len(x), 1)),
                key=kv_cache[2],
                block_table=attn_metadata.indexer_block_table,
            )

        def forward(self, layer_name, hidden_states, kv_cache, attn_metadata, output=None):
            if self.has_indexer:
                shared["topk"] = self.indexer_select_post_process(
                    x=hidden_states, q_c=hidden_states, kv_cache=kv_cache, attn_metadata=attn_metadata
                )
            result = operations.npu_sparse_flash_attention(
                query=hidden_states.unsqueeze(1),
                query_rope=hidden_states.unsqueeze(1),
                key=kv_cache[0],
                key_rope=kv_cache[1],
                value=kv_cache[0],
                sparse_indices=shared["topk"],
                block_table=attn_metadata.block_table,
            )
            return result.squeeze(1)

    class Attention(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.impl = SFAImpl(index)
            self.layer_name = f"model.layers.{index}.self_attn.attn"
            self.cache = tuple(torch.zeros((4, 4, 1, 2)) for _ in range(3))

        def forward(self, hidden):
            return self.impl.forward(self.layer_name, hidden, self.cache, context.attn_metadata[self.layer_name])

    class TestDecoderLayer(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.attention = Attention(index)

        def forward(self, positions, hidden_states, residual):
            return self.attention(hidden_states), hidden_states.clone()

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([TestDecoderLayer(0), TestDecoderLayer(1)])
    config = NS(num_hidden_layers=2, indexer_types=["full", "shared"])
    layers, implementations = worker.inventory(model, config, SFAImpl)
    archive = worker.TensorArchive(case_dir, 0, compare=compare)
    probe = worker.CorrectnessProbe(archive, 8, layers, implementations, lambda: context, config)
    connector = NS(_layer_source_memory_objs=lambda source, layer: source)

    class MergedPage:
        pass

    probe.install(NS(AscendSFAImpl=SFAImpl), NS(), connector, MergedPage)

    def batched_to_gpu():
        source = MergedPage() if h2d_source == "merged" else object()
        return connector._layer_source_memory_objs([source], 0)

    def run(start, end):
        for layer, module in enumerate(model.layers):
            attention = module.attention
            table = torch.tensor([map_ids])
            all_slots = worker.slots_for_positions(attention.cache[0], table, 0, 8)
            for part, cache in enumerate(attention.cache):
                cache.view(-1, 1, 2)[all_slots] = (
                    torch.arange(16, dtype=torch.float32).reshape(8, 1, 2) + layer * 100 + part * 10
                )
            context.attn_metadata[attention.layer_name] = NS(
                seq_lens_cpu=[end],
                query_start_loc_cpu=[0, end - start],
                num_actual_tokens=end - start,
                block_table=table,
                indexer_block_table=table,
                slot_mapping=all_slots[start:end],
                indexer_slot_mapping=all_slots[start:end],
            )
        if h2d_source is not None:
            batched_to_gpu()
        hidden = torch.arange((end - start) * 2, dtype=torch.float32).reshape(-1, 2)
        for module in model.layers:
            hidden, _ = module(torch.arange(start, end), hidden, None)

    return probe, run, operations, SFAImpl


def test_real_consumer_hooks_cover_shared_indexer_and_bank_changes(worker, tmp_path, monkeypatch):
    probe, run, _, _ = _runtime(worker, monkeypatch, tmp_path / "off")
    run(0, 4)
    run(4, 8)
    off = probe.finish(NS())
    assert off["complete"], off["errors"]
    assert off["indexer_layers"] == off["indexer_cache_layers"] == [0]
    assert off["decoder_layers"] == off["sfa_layers"] == [0, 1]
    assert off["file_count"] == off["records"]
    for record in probe.archive.records.values():
        if record["kind"] == "kv_loaded":
            assert record["positions"] == [0, 4] and record["shape"][0] == 4
    compared = []

    def compare(reference, actual):
        assert torch.equal(reference, actual)
        compared.append(actual.numel())
        return {"comparable": True, "rmse": 0.0}

    probe, run, _, _ = _runtime(worker, monkeypatch, tmp_path / "on", map_ids=(1, 3), compare=compare)
    run(0, 4)
    run(4, 8)
    on = probe.finish(NS())
    assert on["complete"], on["errors"]
    assert len(compared) == on["records"] == off["records"]
    assert on["file_count"] == 0


def test_full_worker_off_on_artifacts_pass_real_comparator(worker, tmp_path, monkeypatch):
    from layerwise_prefill_correctness_compare import compare_runs

    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    write(
        tmp_path / "model_info.json",
        {
            "num_hidden_layers": 2,
            "indexer_types": ["full", "shared"],
            "index_topk_pattern": ["F", "S"],
        },
    )
    for case, mapping in (("off", (2, 0)), ("on", (1, 3))):
        probe, run, _, _ = _runtime(
            worker, monkeypatch, tmp_path / case, map_ids=mapping, h2d_source="merged" if case == "on" else None
        )
        run(0, 4)
        run(4, 8)
        coverage = probe.finish(NS())
        assert coverage["complete"], coverage["errors"]
        assert coverage["merged_load_sources"] == (2 if case == "on" else 0)
        directory = tmp_path / case
        write(directory / "coverage.json", [coverage])
        write(directory / "engine_options.json", {"tensor_parallel_size": 1})
        write(
            directory / "environment.json",
            {
                "PYTHONHASHSEED": "0",
                "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": "true" if case == "on" else "false",
                "LMCACHE_STORE_ASYNC": "true" if case == "on" else "false",
            },
        )
        write(
            directory / "result.json",
            {
                "completed": True,
                "prompt_token_ids": list(range(8)),
                "prompt_length": 8,
                "token_ids": [91],
                "output_length": 1,
                "text": "word",
                "num_cached_tokens": 0,
            },
        )
    report = compare_runs(tmp_path)
    assert report["complete"] and report["passed"], report
    assert report["counts"]["compared"] == coverage["records"]
    assert report["status"] == "equal"
    assert (tmp_path / "diff.jsonl").read_text() == ""


def test_history_h2d_evidence_required_only_for_on(worker, tmp_path, monkeypatch):
    for case in ("off", "on"):
        probe, run, _, _ = _runtime(worker, monkeypatch, tmp_path / case, h2d_source=None)
        run(0, 4)
        run(4, 8)
        result = probe.finish(NS())
        assert result["merged_load_sources"] == result["legacy_load_sources"] == 0
        history = [record for record in probe.archive.records.values() if record["kind"] == "kv_loaded"]
        assert len(history) == 5
        assert all(record["positions"] == [0, 4] for record in history)
        assert result["complete"] is (case == "off")
        assert result["errors"] == ([] if case == "off" else ["No actual merged-page H2D source was observed"])


def test_off_without_h2d_still_requires_consumer_history(worker, tmp_path, monkeypatch):
    probe, run, _, _ = _runtime(worker, monkeypatch, tmp_path / "off", h2d_source=None)
    run(0, 4)
    run(4, 8)
    identity = (0, 1, 0, "kv_loaded", "nope")
    del probe.archive.records[identity]
    result = probe.finish(NS())
    assert not result["complete"]
    assert result["errors"] == [f"Missing required tensor {identity}"]


@pytest.mark.parametrize("case", ["off", "on"])
def test_legacy_h2d_evidence_still_rejected_in_both_cases(worker, tmp_path, monkeypatch, case):
    if case == "on":
        baseline, run, _, _ = _runtime(worker, monkeypatch, tmp_path / "off", h2d_source=None)
        run(0, 4)
        run(4, 8)
        assert baseline.finish(NS())["complete"]
    probe, run, _, _ = _runtime(worker, monkeypatch, tmp_path / case, h2d_source="legacy")
    run(0, 4)
    run(4, 8)
    result = probe.finish(NS())
    assert not result["complete"]
    assert result["legacy_load_sources"] == 2
    assert "Legacy nonmerged H2D source objects were observed" in result["errors"]


def test_incomplete_coverage_is_explicit_and_hooks_restore(worker, tmp_path, monkeypatch):
    probe, run, operations, cls = _runtime(worker, monkeypatch, tmp_path / "off")
    wrapped_forward = cls.forward
    wrapped_kernel = operations.npu_sparse_flash_attention
    run(0, 4)
    result = probe.finish(NS())
    assert not result["complete"]
    assert any("complete prompt" in error for error in result["errors"])
    assert cls.forward is not wrapped_forward
    assert operations.npu_sparse_flash_attention is not wrapped_kernel
    assert not probe.handles and not probe.patches


def test_static_inventory_rejects_missing_tail_and_wrong_shared_producer(worker, tmp_path, monkeypatch):
    probe, _, _, cls = _runtime(worker, monkeypatch, tmp_path / "off")
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(list(probe.layers.values()))
    with pytest.raises(RuntimeError, match="num_hidden_layers"):
        worker.inventory(model, NS(num_hidden_layers=3), cls)
    with pytest.raises(RuntimeError, match="indexer presence"):
        worker.inventory(model, NS(num_hidden_layers=2, indexer_types=["full", "full"]), cls)
    probe.restore()
    probe.archive.close()


def test_archive_progress_tracks_real_off_on_phases_and_completed_work(worker, tmp_path, monkeypatch):
    kwargs = dict(step=0, layer=1, kind="decoder", name="input", span=(0, 2))
    observed = {}
    for case in ("off", "on"):
        progress = worker.ProgressMonitor(tmp_path / case / "tensors" / "rank1", 1, log=lambda _: None)
        phases = []
        original = progress.phase

        def phase(phase_name, original=original, phases=phases, **fields):
            phases.append(phase_name)
            original(phase_name, **fields)

        monkeypatch.setattr(progress, "phase", phase)
        archive = worker.TensorArchive(tmp_path / case, 1, progress=progress)
        archive.record(torch.ones(2, 3), **kwargs)
        state = progress.snapshot()
        assert state["phase"] == "compute" and state["records"] == 1
        assert state["files"] == (1 if case == "off" else 0)
        assert state["bytes"] == archive.archived_bytes
        assert state["layer"] == 1 and state["name"] == "input"
        observed[case] = phases
        archive.close()
        assert progress.snapshot()["status"] == "stopped"
    assert observed["off"] == ["copy_cpu", "stats", "save", "manifest", "compute"]
    assert observed["on"] == ["copy_cpu", "stats", "load_off", "compare", "manifest", "compute"]


def test_archive_failure_retains_exact_phase_and_stops_monitor(worker, tmp_path, monkeypatch):
    progress = worker.ProgressMonitor(tmp_path / "off" / "tensors" / "rank1", 1, log=lambda _: None)
    progress.start()
    archive = worker.TensorArchive(tmp_path / "off", 1, progress=progress)
    kwargs = dict(step=0, layer=0, kind="decoder", name="input", span=(0, 2))
    archive.record(torch.ones(2), **kwargs)

    def fail(*args, **kwargs):
        assert progress.snapshot()["phase"] == "save"
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        archive.record(torch.ones(2), **dict(kwargs, name="output"))
    state = json.loads((progress.directory / "progress.json").read_text())
    assert state["status"] == "failed" and state["phase"] == "save" and state["name"] == "output"
    assert state["records"] == state["files"] == 1 and state["bytes"] > 0
    assert not progress._thread.is_alive()
    archive.close()


def test_no_slice_nonfinite_statistics_scan_full_tensor_once(worker, tmp_path, monkeypatch):
    original = torch.isfinite
    sizes = []

    def isfinite(tensor):
        sizes.append(tensor.numel())
        return original(tensor)

    monkeypatch.setattr(torch, "isfinite", isfinite)
    archive = worker.TensorArchive(tmp_path / "off", 1)
    values = torch.tensor([float("nan"), 1.0, 2.0])
    archive.record(values, step=0, layer=0, kind="decoder", name="input", span=(0, 3))
    assert sizes == [3]
    record = next(iter(archive.records.values()))
    assert record["nonfinite"] == record["comparison_nonfinite"] == 1
    archive.record(values, step=0, layer=0, kind="decoder", name="output", span=(0, 3), valid_rows=2)
    assert sizes == [3, 3, 2]
    archive.close()


def test_runner_progress_preserves_results_restores_methods_and_finishes(worker, tmp_path, monkeypatch):
    probe, run, _, _ = _runtime(worker, monkeypatch, tmp_path / "off")
    progress = worker.ProgressMonitor(probe.archive.directory, 0, log=lambda _: None)
    probe.archive.progress = progress
    runner = NS(execute_model=run, sample_tokens=lambda: "original tokens")
    execute, sample = runner.execute_model, runner.sample_tokens
    probe.install_runner_progress(runner)
    assert runner.execute_model(0, 4) is None
    assert progress.snapshot()["operation"] == "execute_model_done"
    assert progress.snapshot()["phase"] == "await_sample"
    assert runner.sample_tokens() == "original tokens"
    assert progress.snapshot()["operation"] == "sample_tokens_done"
    runner.execute_model(4, 8)
    runner.sample_tokens()
    summary = probe.finish(NS())
    assert summary["complete"]
    assert progress.snapshot()["status"] == "finished"
    assert progress.snapshot()["records"] == summary["records"]
    assert runner.execute_model is execute and runner.sample_tokens is sample


def test_runner_error_preserves_inner_archive_failure(worker, tmp_path, monkeypatch):
    probe, run, _, _ = _runtime(worker, monkeypatch, tmp_path / "off")
    progress = worker.ProgressMonitor(probe.archive.directory, 1, log=lambda _: None)
    probe.archive.progress = progress
    runner = NS(execute_model=run, sample_tokens=lambda: None)
    probe.install_runner_progress(runner)

    def fail(*args, **kwargs):
        raise OSError("archive failed")

    monkeypatch.setattr(torch, "save", fail)
    try:
        with pytest.raises(OSError, match="archive failed"):
            runner.execute_model(0, 4)
        state = progress.snapshot()
        assert state["status"] == "failed" and state["phase"] == "save"
        assert state["operation"] == "execute_model" and state["kind"] == "decoder"
    finally:
        probe.restore()
        probe.archive.close()
