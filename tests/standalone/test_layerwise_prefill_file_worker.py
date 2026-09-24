# SPDX-License-Identifier: Apache-2.0
"""CPU contract checks for the explicitly selected full-tensor file-PD probe."""

import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def worker(monkeypatch):
    installed = []
    monkeypatch.setitem(sys.modules, "layerwise_prefill_file_store", NS(install=lambda: installed.append(True)))
    path = ROOT / "tools" / "layerwise_prefill_file_worker.py"
    spec = importlib.util.spec_from_file_location("file_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert installed == [True]
    return module


def call(**updates):
    result = dict(
        model="main",
        call=0,
        phase="prefill",
        positions=[0, 1],
        token_ids=[10, 11],
        context_token_ids=[10, 11],
        expected=[],
    )
    result.update(updates)
    return result


def test_full_tensor_saved_with_padding_and_explicit_valid_mapping(worker, tmp_path):
    archive = worker.FileTensorArchive(tmp_path / "baseline", 2)
    values = torch.arange(6000, dtype=torch.bfloat16).reshape(1000, 6)
    context = call()
    record = archive.record(values, context, 3, "decoder", "input", positions=[0, 1], token_ids=[10, 11])
    archive.close()
    stored = torch.load(archive.root / record["path"], weights_only=True)
    assert torch.equal(stored, values)
    assert record["valid_rows"] == 2 and record["shape"] == [1000, 6]
    assert record["row_axis"] == 0 and record["call_positions"] == [0, 1]
    assert record["token_ids"] == [10, 11]
    assert context["expected"][0]["mapping_only"] is False
    assert "hash" not in json.dumps(record)


def test_archive_fails_duplicates_bad_rows_and_partial_write(worker, tmp_path, monkeypatch):
    archive = worker.FileTensorArchive(tmp_path, 0)
    context = call()
    archive.record(torch.ones(2), context, 0, "decoder", "input", positions=[0, 1], token_ids=[10, 11])
    with pytest.raises(RuntimeError, match="Duplicate"):
        archive.record(torch.ones(2), context, 0, "decoder", "input")
    with pytest.raises(RuntimeError, match="mapping"):
        archive.record(torch.ones(1), context, 0, "decoder", "output", positions=[0, 1], token_ids=[10, 11])
    monkeypatch.setattr(torch, "save", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        archive.record(torch.ones(2), context, 0, "decoder", "output")
    archive.close()
    assert len((archive.directory / "index.jsonl").read_text().splitlines()) == 1
    assert archive.count == 1


def test_tp_row_windows_do_not_assign_global_tokens_to_padding(worker):
    cp = NS(local_start=4, local_end=5, local_end_with_pad=8)
    assert worker.row_window(4, 5, cp, prefer_local=True) == (4, 5)
    assert worker.row_window(8, 5, cp) == (0, 5)
    empty = NS(local_start=8, local_end=8, local_end_with_pad=12)
    assert worker.row_window(4, 5, empty, prefer_local=True) == (8, 8)
    ambiguous = NS(local_start=1, local_end=1, local_end_with_pad=2)
    assert worker.row_window(1, 1, ambiguous, prefer_local=True) == (1, 1)
    assert worker.row_window(1, 1, ambiguous, prefer_local=False) == (0, 1)
    with pytest.raises(RuntimeError, match="Cannot map"):
        worker.row_window(3, 10)


def test_query_context_accepts_decode_and_checks_sharded_real_positions(worker):
    meta = NS(num_actual_tokens=2, seq_lens_cpu=[7], query_start_loc_cpu=[0, 2])
    assert worker.query_context(meta, [5, 6], [21, 22]) == ([5, 6], [21, 22])
    meta.dsa_cp_context = NS(local_start=1, local_end=2, local_end_with_pad=2)
    assert worker.query_context(meta, [6], [21, 22], source_positions=[5, 6]) == ([5, 6], [21, 22])
    with pytest.raises(RuntimeError, match="disagree"):
        worker.query_context(meta, [100], [21, 22], source_positions=[5, 6])
    with pytest.raises(RuntimeError, match="require the observed"):
        worker.query_context(meta, [6], [21, 22])
    with pytest.raises(RuntimeError, match="exactly one"):
        worker.query_context(
            NS(num_actual_tokens=2, seq_lens_cpu=[2, 4], query_start_loc_cpu=[0, 1, 2]), [1, 3], [9, 10]
        )


def test_mtp_one_token_padding_is_not_a_global_position(worker):
    meta = NS(
        num_actual_tokens=1,
        seq_lens_cpu=[101],
        query_start_loc_cpu=[0, 1],
        dsa_cp_context=NS(local_start=1, local_end=1, local_end_with_pad=2),
    )
    assert worker.query_context(meta, [0], [999], positions_local=True, source_positions=[100]) == ([100], [999])
    meta.dsa_cp_context = NS(local_start=0, local_end=1, local_end_with_pad=1)
    assert worker.query_context(meta, [100], [999], positions_local=True, source_positions=[100]) == ([100], [999])
    with pytest.raises(RuntimeError, match="disagree"):
        worker.query_context(meta, [0], [999], positions_local=True, source_positions=[100])


def test_mtp_shifted_history_matches_full_prefill_and_fresh_decode(worker, tmp_path):
    class Model:
        def forward(self, input_ids, positions):
            return None

    model = Model()
    contexts = []
    for stage, positions, inputs in (("baseline", [0, 1, 2, 3], [11, 12, 13, 99]), ("decode", [3], [99])):
        archive = worker.FileTensorArchive(tmp_path / stage, 0)
        probe = worker.FileProbe(NS(), archive, [10, 11, 12, 13], lambda: NS(flash_comm_v1_enabled=False))
        query_count = len(positions)
        probe.metadata = lambda name, count=query_count: NS(
            num_actual_tokens=count, seq_lens_cpu=[4], query_start_loc_cpu=[0, count]
        )
        probe.emit = lambda *args, **kwargs: None
        probe.mtp_sample_indices = [len(positions) - 1]
        probe.model_pre("mtp", model, (torch.tensor(inputs), torch.tensor(positions)), {})
        contexts.append(probe.calls[-1]["context_token_ids"])
        archive.close()
    assert contexts == [[11, 12, 13, 99], [11, 12, 13, 99]]


def test_cross_rank_only_for_proven_full_feature_cp_roles(worker, tmp_path):
    archive = worker.FileTensorArchive(tmp_path, 1)
    probe = worker.FileProbe(NS(), archive, [10, 11], lambda: None)
    probe.inventories["main"] = ({}, {1: (0, NS(impl=NS(enable_dsa_cp=True)))}, {})
    probe.active.append(call())
    cp = NS(local_start=1, local_end=2, local_end_with_pad=2)
    record = probe.emit(torch.ones(1, 3), 0, "attention", "query_nope", meta=NS(dsa_cp_context=cp))
    assert record["cross_rank"] and record["tensor_layout"] == "sequence_sharded"
    mask = probe.emit(torch.ones(1, 3, dtype=torch.bool), 0, "mapping", "attention_valid", meta=NS(dsa_cp_context=cp))
    assert mask["cross_rank"]
    physical = probe.emit(
        torch.ones(1, 3), 0, "mapping", "attention_slots", meta=NS(dsa_cp_context=cp), mapping_only=True
    )
    assert not physical["cross_rank"] and physical["tensor_layout"] == "mapping"
    archive.close()


def test_mtp_logits_keep_padding_but_map_only_real_sample_rows(worker, tmp_path):
    archive = worker.FileTensorArchive(tmp_path, 1)
    probe = worker.FileProbe(NS(), archive, [10, 11], lambda: None)
    context = call(model="mtp", positions=[1], token_ids=[99], context_token_ids=[11, 99], logits_indices=[0])
    probe.last["mtp"] = context
    wrapped = probe.logits_factory("mtp")(lambda: torch.tensor([[1.0, 2.0], [999.0, 999.0]]))
    result = wrapped()
    archive.close()
    record = json.loads((archive.directory / "index.jsonl").read_text())
    assert record["shape"] == [2, 2] and record["valid_rows"] == 1
    assert record["positions"] == [1] and record["token_ids"] == [99]
    assert torch.equal(torch.load(archive.root / record["path"], weights_only=True), result)


def test_mtp_model_output_before_allgather_has_no_rows_on_padding_rank(worker, tmp_path):
    archive = worker.FileTensorArchive(tmp_path, 1)
    probe = worker.FileProbe(NS(), archive, [10, 11], lambda: NS(flash_comm_v1_enabled=True))
    meta = NS(dsa_cp_context=NS(local_start=1, local_end=1, local_end_with_pad=2))
    probe.metadata = lambda name: meta
    probe.active.append(call(model="mtp", positions=[1], token_ids=[99], context_token_ids=[11, 99]))
    probe.model_post("mtp", None, (), {}, torch.zeros(1, 4))
    archive.close()
    record = json.loads((archive.directory / "index.jsonl").read_text())
    assert record["valid_rows"] == 0 and record["positions"] == [] and record["shape"] == [1, 4]


@pytest.mark.parametrize("rank", [0, 1, 7])
def test_installed_mtp_probe_uses_proposer_positions_and_keeps_raw_sum(worker, tmp_path, monkeypatch, rank):
    class Model(torch.nn.Module):
        def forward(self, input_ids, positions, hidden_states):
            return hidden_states

        def compute_logits(self, hidden_states):
            return hidden_states

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class Impl:
        def forward(self):
            pass

        def exec_kv(self):
            pass

        def indexer_select_post_process(self):
            pass

    model = Model()
    source = torch.tensor([3])
    meta = NS(
        num_actual_tokens=1,
        seq_lens_cpu=[4],
        query_start_loc_cpu=[0, 1],
        dsa_cp_context=NS(local_start=rank, local_end=min(rank + 1, 1), local_end_with_pad=rank + 1),
    )
    layer_name = "model.layers.78.self_attn.attn"
    inventory = ({}, {0: (78, NS(layer_name=layer_name, impl=NS(enable_dsa_cp=True)))}, dict(required_roles=[]))
    monkeypatch.setattr(worker, "model_inventory", lambda *args: inventory)
    monkeypatch.setattr(torch.ops, "_C_ascend", NS(npu_sparse_flash_attention=lambda: None))
    drafter = NS(
        _get_positions=lambda count: source[:count],
        _run_mtp_draft_layer_with_diagnostics=lambda model_kwargs, **kwargs: model(**model_kwargs),
    )
    runner = NS(drafter=drafter, parallel_config=NS(tensor_parallel_size=8))
    archive = worker.FileTensorArchive(tmp_path / "decode", rank)
    probe = worker.FileProbe(
        runner,
        archive,
        [10, 11, 12, 13],
        lambda: NS(attn_metadata={layer_name: meta}, flash_comm_v1_enabled=True),
    )
    probe.models["mtp"] = model
    original_draft = drafter._run_mtp_draft_layer_with_diagnostics
    probe.install(NS(AscendSFAImpl=Impl), NS(), NS())
    raw_positions = torch.tensor([24 if rank == 0 else 0])
    runtime = dict(num_input_tokens=1, batch_size=1, token_indices_to_sample=torch.tensor([0]))
    drafter._run_mtp_draft_layer_with_diagnostics(
        dict(input_ids=torch.tensor([99]), positions=raw_positions, hidden_states=torch.ones(1, 4)),
        runtime_inputs=runtime,
    )
    assert probe.mtp_positions is None and probe.mtp_sample_indices is None
    # Logits run after the draft wrapper returns and must retain its selection.
    model.compute_logits(torch.ones(1, 4))
    summary = probe.finish()
    assert summary["complete"], summary["errors"]
    assert drafter._run_mtp_draft_layer_with_diagnostics is original_draft
    assert summary["calls"][0]["positions"] == [3]
    assert summary["calls"][0]["context_token_ids"] == [11, 12, 13, 99]
    records = [json.loads(line) for line in (archive.directory / "index.jsonl").read_text().splitlines()]
    record = next(item for item in records if (item["kind"], item["name"]) == ("model_input", "positions"))
    assert record["positions"] == ([3] if rank == 0 else [])
    assert torch.equal(torch.load(archive.root / record["path"], weights_only=True), raw_positions)


def test_sparse_slots_follow_actual_remapped_table_and_exclude_causal_padding(worker):
    selected = torch.tensor([[[0, 1, -1]], [[2, 0, 3]]])
    logical = torch.tensor([[[1, 0, -1]], [[3, 1, 99]]])
    table = torch.tensor([[2, 0]])
    logical, slots, valid = worker.sparse_slots(
        selected, table, torch.tensor([2]), torch.tensor([4]), logical, [1, 3], 2
    )
    assert slots.tolist() == [[4, 5, -1], [0, 4, -1]]
    assert valid.tolist() == [[True, True, False], [True, True, False]]
    cache = torch.zeros(3, 2, 1, 2)
    cache.reshape(6, 1, 2)[4] = 10
    cache.reshape(6, 1, 2)[5] = 20
    cache.reshape(6, 1, 2)[0] = 30
    values, positions, pairs = worker.unique_kv_rows(cache, logical, slots, valid)
    assert positions == [0, 1, 3]
    assert values[:, 0, 0].tolist() == [20, 10, 30]
    assert pairs.tolist() == [[0, 5], [1, 4], [3, 0]]


def test_duplicate_scratch_aliases_checked_not_silently_discarded(worker):
    logical = torch.tensor([[1, 1]])
    slots = torch.tensor([[0, 1]])
    valid = torch.ones_like(slots, dtype=torch.bool)
    cache = torch.tensor([[[[5.0]], [[5.0]]]])
    values, positions, pairs = worker.unique_kv_rows(cache, logical, slots, valid)
    assert positions == [1] and values.numel() == 1 and len(pairs) == 2
    cache[0, 1] = 6
    with pytest.raises(RuntimeError, match="unequal consumed"):
        worker.unique_kv_rows(cache, logical, slots, valid)


def test_sparse_mapping_rejects_mismatched_rows_and_bad_physical_slots(worker):
    with pytest.raises(RuntimeError, match="cannot be aligned"):
        worker.sparse_slots(
            torch.zeros(2, 1, 2),
            torch.zeros(1, 1),
            torch.tensor([2]),
            torch.tensor([2]),
            torch.zeros(1, 1, 2),
            [1, 2],
            2,
        )
    with pytest.raises(RuntimeError, match="invalid physical"):
        worker.unique_kv_rows(
            torch.zeros(1, 2, 1, 1), torch.tensor([[0]]), torch.tensor([[99]]), torch.tensor([[True]])
        )
    values, positions, pairs = worker.unique_kv_rows(
        torch.zeros(1, 2, 1, 1),
        torch.empty(0, 2, dtype=torch.long),
        torch.empty(0, 2, dtype=torch.long),
        torch.empty(0, 2, dtype=torch.bool),
    )
    assert list(values.shape) == [0, 1, 1] and positions == [] and pairs.shape == (0, 2)


@pytest.mark.parametrize("invocation", ["module", "eager_decorator", "explicit_forward"])
def test_probe_captures_decoder_sfa_indexer_kv_and_presampling_logits(worker, tmp_path, monkeypatch, invocation):
    meta = NS(num_actual_tokens=4, seq_lens_cpu=[4], query_start_loc_cpu=[0, 4])
    cache = torch.arange(8, dtype=torch.float32).reshape(2, 2, 1, 2)
    table = torch.tensor([[0, 1]])
    lengths = torch.tensor([4])
    operations = NS(
        npu_lightning_indexer=lambda **kw: torch.arange(4).reshape(1, 1, 4).repeat(4, 1, 1),
        npu_sparse_flash_attention=lambda **kw: kw["query"] + 1,
    )
    monkeypatch.setattr(torch.ops, "_C_ascend", operations)

    class Impl:
        has_indexer = True
        skip_topk = False

        def exec_kv(self, kv_no_split, kv_cache, slots, attn_metadata):
            return kv_cache[1].reshape(4, 1, 2), kv_cache[0].reshape(4, 1, 2)

        def indexer_select_post_process(self, x, q_c):
            return operations.npu_lightning_indexer(
                query=x[:, None],
                key=cache,
                weights=x,
                actual_seq_lengths_query=lengths,
                actual_seq_lengths_key=lengths,
                block_table=table,
            )

        def forward(self, layer_name, hidden_states, kv_cache, attn_metadata):
            self.exec_kv(hidden_states, kv_cache, torch.arange(4), attn_metadata)
            topk = self.indexer_select_post_process(hidden_states, hidden_states)
            return operations.npu_sparse_flash_attention(
                query=hidden_states[:, None],
                query_rope=hidden_states[:, None],
                key=cache,
                key_rope=cache,
                sparse_indices=topk,
                block_table=table,
                actual_seq_lengths_query=lengths,
                actual_seq_lengths_kv=lengths,
            ).reshape(4, 2)

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.impl = Impl()
            self.layer_name = "model.layers.0.self_attn.attn"

        def forward(self, hidden_states):
            return self.impl.forward(self.layer_name, hidden_states, (cache, cache), meta)

    class FakeDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()

        def forward(self, positions, hidden_states, residual=None):
            return self.attn(hidden_states), hidden_states

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([FakeDecoderLayer()])

        def forward(self, input_ids, positions):
            hidden = input_ids.float()[:, None].repeat(1, 2)
            return self.layers[0](positions, hidden)[0]

        def compute_logits(self, hidden_states):
            return hidden_states.repeat(1, 2)

    class EagerDecoratedModel(Model):
        # vLLM support_torch_compile replaces DeepSeekMTP.__call__ with this
        # dispatch when do_not_compile is true; nn.Module hooks are bypassed.
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    model = EagerDecoratedModel() if invocation == "eager_decorator" else Model()
    runner = NS(model=model, logits_indices=torch.tensor([3]))
    archive = worker.FileTensorArchive(tmp_path / "baseline", 0)
    probe = worker.FileProbe(
        runner, archive, [10, 11, 12, 13], lambda: NS(attn_metadata={model.layers[0].attn.layer_name: meta})
    )
    probe.models["main"] = model
    original = Impl.forward
    probe.install(NS(AscendSFAImpl=Impl), NS(), NS())
    invoke = model.forward if invocation == "explicit_forward" else model
    result = invoke(torch.tensor([10, 11, 12, 13]), torch.arange(4))
    logits = model.compute_logits(result[runner.logits_indices])
    summary = probe.finish()
    assert summary["complete"], summary["errors"]
    assert len(summary["calls"]) == 1  # Ordinary nn.Module must not be observed twice.
    assert Impl.forward is original
    records = [json.loads(line) for line in (archive.directory / "index.jsonl").read_text().splitlines()]
    assert len(records) == summary["records"]
    logit_record = next(item for item in records if item["kind"] == "logits")
    assert logit_record["positions"] == [3] and logit_record["token_ids"] == [13]
    assert torch.equal(torch.load(archive.root / logit_record["path"], weights_only=True), logits)
    kv = next(item for item in records if (item["kind"], item["name"]) == ("kv_consumed", "nope"))
    assert kv["positions"] == [0, 1, 2, 3]
    assert torch.equal(torch.load(archive.root / kv["path"], weights_only=True), cache.reshape(4, 1, 2))
    current_kv = next(item for item in records if (item["kind"], item["name"]) == ("kv_current", "nope"))
    assert current_kv["positions"] == [0, 1, 2, 3]
    assert torch.equal(torch.load(archive.root / current_kv["path"], weights_only=True), cache.reshape(4, 1, 2))
    assert next(item for item in records if (item["kind"], item["name"]) == ("attention", "topk"))["mapping_only"]
    assert summary["calls"][0]["context_token_ids"] == [10, 11, 12, 13]
    assert json.loads((archive.directory / "coverage.json").read_text())["complete"]
    (tmp_path / "model_info.json").write_text(json.dumps({"num_hidden_layers": 1}), encoding="utf-8")
    (archive.root / "output.json").write_text(
        json.dumps(
            {
                "completed": True,
                "stage": "baseline",
                "prompt_token_ids": [10, 11, 12, 13],
                "prompt_length": 4,
                "token_ids": [14],
                "output_token_limit": 1,
                "num_cached_tokens": 0,
                "enforce_eager": True,
                "mtp": {"configured_tokens": 0, "metrics": {}},
            }
        ),
        encoding="utf-8",
    )
    (archive.root / "coverage.json").write_text(json.dumps([summary]), encoding="utf-8")
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    from layerwise_prefill_file_compare import validate_baseline

    validation = validate_baseline(archive.root, 1, 1)
    assert validation["valid"], validation["errors"]


def test_finish_requires_every_static_role_and_preserves_mtp_model_identity(worker, tmp_path):
    archive = worker.FileTensorArchive(tmp_path / "decode", 0)
    probe = worker.FileProbe(NS(), archive, [10], lambda: None)
    inventory = dict(
        layers=[9], sfa_layers=[9], indexer_layers=[9], required_roles=[dict(kind="decoder", name="input", layers=[9])]
    )
    probe.inventories = {"mtp": ({}, {}, inventory)}
    probe.calls.append(call(model="mtp"))
    summary = probe.finish()
    assert not summary["complete"] and any("Missing required" in error for error in summary["errors"])
    assert any("pre-sampling logits" in error for error in summary["errors"])
    assert summary["models"]["mtp"]["layers"] == [9]


def test_prefill_can_finish_before_first_mtp_forward(worker, tmp_path):
    archive = worker.FileTensorArchive(tmp_path / "prefill", 0)
    probe = worker.FileProbe(NS(), archive, [10], lambda: None)
    probe.inventories = {"mtp": ({}, {}, dict(layers=[9], required_roles=[]))}
    assert probe.finish()["complete"]


def test_flush_uses_real_completion_and_surfaces_failure(worker):
    events = []
    engine = NS(
        poll_layerwise_prefill_puts=lambda **kw: events.append(("poll", kw)),
        wait_for_pending_sync_stores=lambda: events.append(("sync_puts", {})),
        _store_cv=threading.Condition(),
        _pending_store_reqs={},
        config=NS(blocking_timeout_secs=0.01),
        _direct_store_states={"req": object()},
        wait_for_direct_stores=lambda ids: events.append(("direct", ids)),
    )
    assert worker.flush_engine_stores(engine)
    assert events == [("poll", {"final": True}), ("direct", ("req",)), ("poll", {"final": True}), ("sync_puts", {})]
    with pytest.raises(RuntimeError, match="failed write"):
        worker.flush_engine_stores(engine, ["failed write"])
    engine._pending_store_reqs = {"req": 1}
    with pytest.raises(TimeoutError):
        worker.flush_engine_stores(engine)
