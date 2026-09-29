# SPDX-License-Identifier: Apache-2.0
"""CPU observations of the real PD recorder, without importing NPU packages."""

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def module():
    spec = importlib.util.spec_from_file_location("_pd_dump_test", ROOT / "vllm_ascend/pd_tensor_dump.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    loaded.test_probes = []
    yield loaded
    for probe in loaded.test_probes:
        probe.restore()


def make_probe(module, tmp_path, prompts, *, cp=None, external=None, max_token_rows=64, max_features=8):
    ids = list(prompts)
    requests = {
        key: NS(prompt_token_ids=list(tokens), output_token_ids=[], sampling_params="max_tokens=16")
        for key, tokens in prompts.items()
    }
    meta = NS(num_actual_tokens=0, query_start_loc_cpu=[0], dsa_cp_context=cp)
    runner = NS(
        requests=requests,
        input_batch=NS(req_ids=ids, vocab_size=1000),
        model_config=NS(model="test-model"),
        layerwise_prefill_p_node=False,
    )
    probe = module.PDTensorDump(
        runner,
        tmp_path,
        "D",
        dict(host="worker.test", pid=123, tp_rank=1, tp_size=4, dp_rank=2, dp_size=4),
        lambda: NS(attn_metadata={"layer0": meta}),
        max_token_rows=max_token_rows,
        max_features=max_features,
    )
    probe.layers = {0: object()}
    module.test_probes.append(probe)
    probe.attentions = {1: (0, "layer0")}
    probe.observe_scheduler(
        NS(
            scheduled_new_reqs=[
                NS(req_id=key, external_req_id=(external or {}).get(key, f"external-{key}")) for key in ids
            ],
            finished_req_ids=set(),
            num_scheduled_tokens={key: len(prompts[key]) for key in ids},
        )
    )
    return probe, runner, meta


def batch(meta, rows):
    """Rows are (request-local positions, actual input tokens) per request."""
    positions, tokens, boundaries = [], [], [0]
    for pos, ids in rows:
        positions.extend(pos)
        tokens.extend(ids)
        boundaries.append(len(tokens))
    meta.num_actual_tokens = len(tokens)
    meta.query_start_loc_cpu = boundaries
    return torch.tensor(tokens), torch.tensor(positions)


@pytest.mark.parametrize("padding_rank", [False, True])
def test_mtp_real_wrappers_keep_shifted_context_separate_from_main_sampling(module, tmp_path, padding_rank):
    probe, runner, main_meta = make_probe(module, tmp_path, {"a": [10, 11, 12, 13], "b": [20, 21, 22]})
    inputs, positions = batch(main_meta, [([3, 4], [13, 99]), ([2], [22])])
    with probe.forward(inputs, positions):
        fill_expected(probe, main_meta)
    cp = NS(local_start=2, local_end=2, local_end_with_pad=3) if padding_rank else None
    mtp_meta = NS(num_actual_tokens=2, query_start_loc_cpu=[0, 1, 2], dsa_cp_context=cp)
    context = NS(
        attn_metadata={"layer0": main_meta, "model.layers.1.attn": mtp_meta}, flash_comm_v1_enabled=padding_rank
    )
    probe.get_context = lambda: context

    class Impl:
        pass

    class TestDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = torch.nn.Module()
            self.attention.impl = Impl()
            self.attention.layer_name = "model.layers.1.attn"

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleDict({"1": TestDecoderLayer()})

        def forward(self, input_ids, positions, hidden_states):
            for item in probe.expected("mtp"):
                if item["kind"] in ("model_input", "model_output", "logits", "draft"):
                    continue
                probe.emit(torch.ones(2, 3), item["layer"], item["kind"], item["name"], mtp_meta, prefer_local=False)
            return hidden_states

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)  # Real eager decorator bypasses nn hooks.

        def compute_logits(self, hidden_states):
            return hidden_states

    model = Draft()
    runner.drafter = NS(
        method="mtp",
        model=model,
        _get_positions=lambda n: torch.tensor([3, 2])[:n],
        _run_mtp_draft_layer_with_diagnostics=lambda values, **kwargs: model(**values),
    )
    original = model.forward
    probe.install_mtp(Impl)
    hidden = torch.ones(1 if padding_rank else 2, 3)
    actual = runner.drafter._run_mtp_draft_layer_with_diagnostics(
        dict(
            input_ids=torch.tensor([99, 199]),
            positions=torch.tensor([0]) if padding_rank else torch.tensor([3, 2]),
            hidden_states=hidden,
        ),
        runtime_inputs=dict(num_input_tokens=2, batch_size=2, token_indices_to_sample=torch.tensor([0, 1])),
        draft_step=0,
    )
    assert actual is hidden
    logits = torch.tensor([[0.0, 3.0, 1.0], [9.0, 0.0, 1.0], [999.0, 999.0, 999.0]])
    assert model.compute_logits(logits) is logits
    finish_forward(probe, sampled=[[31], [32]])
    for internal, shifted, token in (("a", [11, 12, 13, 99], 1), ("b", [21, 22, 199], 0)):
        archive = probe.archives[internal]
        call = archive.calls[1]
        assert call["model"] == "mtp" and call["parent_call"] == 0 and call["complete"]
        assert call["context_token_ids"] == shifted
        assert value_for(archive, "draft", "token_ids", call=1).tolist() == [token]
        assert all(row["model"] == "mtp" for row in records(archive) if row["call"] == 1)
        hidden_record = record_for(archive, "model_input", "hidden_states", call=1)
        assert hidden_record["positions"] == ([] if padding_rank else [len(shifted) - 1])
        assert json.loads((archive.root / "sampled.jsonl").read_text())["after_call"] == 0
        assert archive.metadata["complete"]
    probe.restore()
    assert model.forward == original


def records(archive):
    path = archive.root / "index.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_mtp_rejected_padding_is_not_archived_as_valid_query_or_current_kv(module, tmp_path):
    probe, _, main_meta = make_probe(module, tmp_path, {"a": [10, 11], "b": [20, 21]})
    ids, positions = batch(main_meta, [([1, 2], [11, 999]), ([1, 2], [21, 998])])
    with probe.forward(ids, positions):
        fill_expected(probe, main_meta)
    probe.mtp_layers = {1: object()}
    probe.attentions[2] = (1, "mtp")
    meta = NS(
        num_actual_tokens=4,
        query_start_loc_cpu=[0, 2, 4],
        dsa_cp_context=None,
        slot_mapping=torch.tensor([0, -1, 2, -1]),
    )
    probe.get_context = lambda: NS(attn_metadata={"mtp": meta})
    probe.mtp_runtime = dict(positions=[1, 2, 1, 2], indices=[0, 2], draft_step=0)
    probe.start_mtp(
        dict(
            input_ids=torch.tensor([30, 999, 40, 998]),
            positions=positions,
            hidden_states=torch.arange(4).reshape(4, 1),
        )
    )
    cache = torch.arange(4).reshape(2, 2, 1, 1)
    probe.current_kv([cache, cache], 1, meta)
    probe.indexer_kv(cache, torch.tensor([[0], [1]]), torch.tensor([3, 3]), 1, "key")
    for internal, context, kv in (("a", [11, 30], 0), ("b", [21, 40], 2)):
        archive = probe.archives[internal]
        assert archive.calls[1]["context_token_ids"] == context
        assert archive.calls[1]["positions"] == [1]
        assert archive.calls[1]["rejected_query_rows"] == 1
        assert value_for(archive, "kv_current", "nope", call=1).flatten().tolist() == [kv]
        assert value_for(archive, "model_input", "hidden_states", call=1).flatten().tolist() == [kv]
        assert record_for(archive, "kv_indexer", "key", call=1)["positions"] == [0, 1]


def record_for(archive, kind, name, call=0):
    return next(row for row in records(archive) if (row["kind"], row["name"], row["call"]) == (kind, name, call))


def value_for(archive, kind, name, call=0):
    row = record_for(archive, kind, name, call)
    return torch.load(archive.root / row["path"], weights_only=True)


def manifest(archive):
    return json.loads((archive.root / "manifest.json").read_text())


def fill_expected(probe, meta):
    # Identity tests use all required roles; tensor-specific tests below use the
    # actual KV extraction methods instead of this minimal completed forward.
    value = torch.arange(meta.num_actual_tokens).reshape(-1, 1).float()
    for expected in probe.expected():
        key = (expected["layer"], expected["kind"], expected["name"])
        if key[1] == "logits":
            continue
        if all(key in entry["observed"] for entry in probe.active["entries"]):
            continue
        probe.emit(value, *key, meta, prefer_local=False)


def finish_forward(probe, *, sampled=None):
    entries = probe.last["entries"]
    probe.runner.logits_indices = torch.tensor([entry["end"] - 1 for entry in entries])
    probe.logits_factory(lambda: torch.zeros(len(entries), 3))()
    internal_ids = [entry["archive"].metadata["internal_request_id"] for entry in entries]
    probe.sampled(internal_ids, sampled if sampled is not None else [[] for _ in entries])


def test_same_positions_in_two_requests_never_mix(module, tmp_path):
    probe, _, meta = make_probe(module, tmp_path, {"a": [10, 11], "b": [20, 21]})
    ids, positions = batch(meta, [([0, 1], [10, 11]), ([0, 1], [20, 21])])
    with probe.forward(ids, positions):
        probe.emit(torch.tensor([[100], [101], [200], [201]]), 0, "extra", "hidden", meta)
        fill_expected(probe, meta)
    finish_forward(probe)
    for key, expected in (("a", [100, 101]), ("b", [200, 201])):
        archive = probe.archives[key]
        row = record_for(archive, "extra", "hidden")
        assert row["positions"] == [0, 1]
        assert row["token_ids"] == ([10, 11] if key == "a" else [20, 21])
        assert value_for(archive, "extra", "hidden").flatten().tolist() == expected
        assert manifest(archive)["internal_request_id"] == key
        assert manifest(archive)["complete"]


@pytest.mark.parametrize("local_start,local_end,capacity", [(1, 3, 3), (4, 4, 2)])
def test_tp_padding_and_empty_request_shards(module, tmp_path, local_start, local_end, capacity):
    cp = NS(local_start=local_start, local_end=local_end, local_end_with_pad=local_start + capacity)
    probe, _, meta = make_probe(module, tmp_path, {"a": [10, 11], "b": [20, 21]}, cp=cp)
    ids, positions = batch(meta, [([0, 1], [10, 11]), ([0, 1], [20, 21])])
    probe.start(ids, positions)
    values = torch.full((capacity, 1), 999.0)
    values[: local_end - local_start, 0] = torch.arange(local_start, local_end).float()
    probe.emit(values, 0, "decoder", "input", meta)
    for key, global_start, global_end in (("a", 0, 2), ("b", 2, 4)):
        actual = value_for(probe.archives[key], "decoder", "input").flatten().tolist()
        expected = list(range(max(local_start, global_start), min(local_end, global_end)))
        assert actual == expected
        assert 999 not in actual
        row = record_for(probe.archives[key], "decoder", "input")
        assert row["tensor_layout"] == "sequence_sharded"
        assert row["positions"] == [index - global_start for index in expected]


def test_multiple_chunks_and_target_drafts_record_actual_input_context(module, tmp_path):
    probe, runner, meta = make_probe(module, tmp_path, {"a": [10, 11, 12, 13]})
    probe.role = "P"  # P still archives every compute-prefill chunk.
    for query_positions, query_ids in (([0, 1], [10, 11]), ([2, 3], [12, 13]), ([4, 5, 6], [20, 91, 92])):
        runner.requests["a"].output_token_ids = [20] if query_positions[0] == 4 else []
        ids, positions = batch(meta, [(query_positions, query_ids)])
        with probe.forward(ids, positions):
            fill_expected(probe, meta)
        finish_forward(probe)
    archive = probe.archives["a"]
    contexts = [archive.calls[index]["context_token_ids"] for index in range(3)]
    assert contexts == [[10, 11], [10, 11, 12, 13], [10, 11, 12, 13, 20, 91, 92]]
    assert archive.calls[2]["phase"] == "decode"
    assert archive.calls[2]["context_complete"]
    assert runner.requests["a"].output_token_ids == [20]
    assert record_for(archive, "model_input", "input_ids", 2)["token_ids"] == [20, 91, 92]


def test_d_captures_first_handoff_call_then_skips_without_readback(module, tmp_path, monkeypatch, capsys):
    probe, runner, meta = make_probe(module, tmp_path, {"a": [10, 11, 12]})
    ids, positions = batch(meta, [([2, 3], [12, 30])])
    with probe.forward(ids, positions):
        fill_expected(probe, meta)
    finish_forward(probe, sampled=[[31]])
    archive = probe.archives["a"]
    before = records(archive)
    assert archive.metadata["capture_policy"] == "first_D_forward"
    assert archive.calls[0]["positions"] == [2, 3]
    assert archive.calls[0]["phase"] == "prefill"
    assert "later_capture=off" in capsys.readouterr().out

    def no_readback(_):
        raise AssertionError("later decode performed a diagnostic CPU copy")

    monkeypatch.setattr(module, "cpu_tensor", no_readback)
    runner.requests["a"].output_token_ids = [30, 31]
    ids, positions = batch(meta, [([4], [31])])
    with probe.forward(ids, positions):
        assert probe.active is None and probe.last is None
        result = torch.ones(1, 3)
        assert probe.kernel_factory("attention")(lambda **kwargs: result)() is result
    assert probe.logits_factory(lambda: result)() is result
    probe.sampled(["a"], [[32]])
    assert records(archive) == before and len(archive.calls) == 1
    assert probe.progress.thread is None
    probe.observe_scheduler(NS(scheduled_new_reqs=[], finished_req_ids={"a"}, num_scheduled_tokens={}))
    assert not probe.captured_decode
    assert manifest(archive)["request_finished"]


def test_new_d_request_in_mixed_batch_keeps_original_row_offsets(module, tmp_path):
    probe, _, meta = make_probe(module, tmp_path, {"a": [10, 11], "b": [20, 21]})
    ids, positions = batch(meta, [([1], [11]), ([], [])])
    with probe.forward(ids, positions):
        fill_expected(probe, meta)
    finish_forward(probe, sampled=[[30]])
    before = records(probe.archives["a"])
    ids, positions = batch(meta, [([2], [30]), ([1], [21])])
    with probe.forward(ids, positions):
        (entry,) = probe.active["entries"]
        assert entry["row"] == 1 and entry["start"] == 1
        fill_expected(probe, meta)
    finish_forward(probe, sampled=[[40]])
    assert records(probe.archives["a"]) == before
    assert value_for(probe.archives["b"], "model_input", "input_ids").tolist() == [21]


def test_progress_names_blocked_copy_and_write_without_device_access(module, monkeypatch, capsys):
    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    progress = module.CaptureProgress(dict(host="test", dp_rank=1, tp_rank=2))
    progress.set("m=main l=3 kv_consumed/nope cpu_copy")
    now[0] = 6
    progress.report_wait()
    assert "kv_consumed/nope cpu_copy waiting=6.0s" in capsys.readouterr().out
    progress.set("m=main l=3 kv_consumed/nope file_write")
    now[0] = 12
    progress.report_wait()
    assert "file_write waiting=6.0s" in capsys.readouterr().out
    progress.stop()
    progress.report_wait()
    assert not capsys.readouterr().out


def test_unknown_context_gap_cannot_be_reported_complete(module, tmp_path):
    probe, _, meta = make_probe(module, tmp_path, {"a": [10, 11]})
    ids, positions = batch(meta, [([4], [99])])
    try:
        with probe.forward(ids, positions):
            fill_expected(probe, meta)
        finish_forward(probe)
    except ValueError as error:
        assert "context" in str(error).lower()
    else:
        assert not manifest(probe.archives["a"])["complete"]


def test_missing_external_id_never_guesses_internal_id(module, tmp_path):
    probe, _, meta = make_probe(
        module, tmp_path, {"cmpl-user-0-abcdef01": [1]}, external={"cmpl-user-0-abcdef01": None}
    )
    ids, positions = batch(meta, [([0], [1])])
    with pytest.raises(ValueError, match="external_req_id"), probe.forward(ids, positions):
        pytest.fail("model executed without known request identity")
    assert not list(tmp_path.rglob("manifest.json"))


def test_forward_failure_keeps_manifest_incomplete(module, tmp_path):
    probe, _, meta = make_probe(module, tmp_path, {"a": [10]})
    ids, positions = batch(meta, [([0], [10])])
    with pytest.raises(RuntimeError, match="model failed"), probe.forward(ids, positions):
        fill_expected(probe, meta)
        raise RuntimeError("model failed")
    archive = probe.archives["a"]
    state = manifest(archive)
    assert not state["complete"]
    assert not archive.calls[0]["complete"]
    assert "model failed" in state["errors"][-1]
    assert probe.active is None


def test_tensor_save_failure_preserves_exception_and_incomplete_manifest(module, tmp_path, monkeypatch):
    probe, _, meta = make_probe(module, tmp_path, {"a": [10]})
    ids, positions = batch(meta, [([0], [10])])
    monkeypatch.setattr(module.torch, "save", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"), probe.forward(ids, positions):
        pass
    state = manifest(probe.archives["a"])
    assert not state["complete"]
    assert state["records"] == 0
    assert "disk full" in state["errors"][-1]


def test_existing_archive_is_not_overwritten(module, tmp_path):
    probe, _, meta = make_probe(module, tmp_path, {"a": [10]})
    ids, positions = batch(meta, [([0], [10])])
    probe.start(ids, positions)
    archive = probe.archives["a"]
    before = (archive.root / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        module.RequestArchive(tmp_path, archive.metadata)
    assert (archive.root / "manifest.json").read_bytes() == before


def test_scheduler_finish_is_explicit_and_does_not_complete_failed_call(module, tmp_path):
    probe, _, meta = make_probe(module, tmp_path, {"a": [10]})
    ids, positions = batch(meta, [([0], [10])])
    with pytest.raises(RuntimeError), probe.forward(ids, positions):
        raise RuntimeError("forward failed")
    archive = probe.archives["a"]
    probe.observe_scheduler(NS(scheduled_new_reqs=[], finished_req_ids={"a"}, num_scheduled_tokens={}))
    assert manifest(archive)["request_finished"]
    assert not manifest(archive)["complete"]
    assert "a" not in probe.archives and "a" not in probe.external_ids


def test_logits_split_by_real_sampling_indices_and_scheduler_finish(module, tmp_path):
    probe, runner, meta = make_probe(module, tmp_path, {"a": [10, 11], "b": [20, 21]})
    ids, positions = batch(meta, [([0, 1], [10, 11]), ([0, 1], [20, 21])])
    with probe.forward(ids, positions):
        fill_expected(probe, meta)
    archives = dict(probe.archives)
    runner.logits_indices = torch.tensor([1, 2, 3])
    result = torch.tensor([[101.0, 102.0], [201.0, 202.0], [203.0, 204.0]])
    assert probe.logits_factory(lambda: result)() is result
    assert record_for(archives["a"], "logits", "output")["positions"] == [1]
    assert record_for(archives["b"], "logits", "output")["positions"] == [0, 1]
    assert torch.equal(value_for(archives["a"], "logits", "output"), result[:1])
    assert torch.equal(value_for(archives["b"], "logits", "output"), result[1:])
    probe.sampled(["a", "b"], [[31], [41]])
    probe.observe_scheduler(NS(scheduled_new_reqs=[], finished_req_ids={"a", "b"}, num_scheduled_tokens={}))
    assert all(manifest(archive)["complete"] for archive in archives.values())
    assert all(manifest(archive)["request_finished"] for archive in archives.values())


def test_sampling_counts_accepted_tokens_not_calls_or_padding(module, tmp_path):
    probe, _, meta = make_probe(module, tmp_path, {"a": [10], "b": [20]})
    ids, positions = batch(meta, [([0], [10]), ([0], [20])])
    with probe.forward(ids, positions):
        fill_expected(probe, meta)
    finish_forward(probe, sampled=[[31, 32, -1], [41, -1, 1000]])
    for key, expected in (("a", [31, 32]), ("b", [41])):
        row = json.loads((probe.archives[key].root / "sampled.jsonl").read_text())
        assert row == {"after_call": 0, "token_ids": expected}


def attention_inputs():
    # A uses physical block 2, B block 0; identical logical positions must not
    # merge their consumed cache rows. Each query selects both history and self.
    return dict(
        query=torch.zeros(2, 1, 1),
        sparse_indices=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
        block_table=torch.tensor([[2], [0]], dtype=torch.int32),
        actual_seq_lengths_kv=torch.tensor([2, 2]),
        actual_seq_lengths_query=torch.tensor([1, 2]),
        key=torch.arange(6).reshape(3, 2, 1, 1).float(),
        key_rope=(torch.arange(6) + 10).reshape(3, 2, 1, 1).float(),
    )


def begin_decode(module, tmp_path):
    probe, _, meta = make_probe(module, tmp_path, {"a": [10, 11], "b": [20, 21]})
    ids, positions = batch(meta, [([1], [11]), ([1], [21])])
    probe.start(ids, positions)
    probe.capture_topk(torch.tensor([[0, 1], [0, 1]]), meta)
    return probe, meta


def test_consumed_kv_uses_each_requests_actual_block_table(module, tmp_path):
    probe, meta = begin_decode(module, tmp_path)
    probe.attention_kv(attention_inputs(), 0, meta)
    assert value_for(probe.archives["a"], "kv_consumed", "nope").flatten().tolist() == [4, 5]
    assert value_for(probe.archives["b"], "kv_consumed", "nope").flatten().tolist() == [0, 1]
    assert record_for(probe.archives["b"], "kv_consumed", "rope")["token_ids"] == [20, 21]


def test_empty_tp_query_shard_records_no_padding_kv(module, tmp_path):
    probe, meta = begin_decode(module, tmp_path)
    meta.dsa_cp_context = NS(local_start=2, local_end=2, local_end_with_pad=4)
    probe.capture_topk(torch.tensor([[0, 1], [0, 1]]), meta)
    values = attention_inputs()
    values["sparse_indices"].fill_(-1)
    probe.attention_kv(values, 0, meta)
    for archive in probe.archives.values():
        assert record_for(archive, "kv_consumed", "nope")["positions"] == []
        assert value_for(archive, "kv_consumed", "nope").shape == (0, 1, 1)


@pytest.mark.parametrize("invalid", ["table_width", "physical_slot"])
def test_consumed_kv_invalid_mapping_fails_closed(module, tmp_path, invalid):
    probe, meta = begin_decode(module, tmp_path)
    values = attention_inputs()
    if invalid == "table_width":
        values["sparse_indices"][0, 0] = 2
        values["actual_seq_lengths_kv"][0] = 3
    else:
        values["block_table"][0, 0] = 3
    with pytest.raises(ValueError, match="block table|physical slot"):
        probe.attention_kv(values, 0, meta)


@pytest.mark.parametrize("values", [[4.0, 5.0], [4.0, 4.0], [float("nan"), float("nan")]])
def test_consumed_kv_preserves_every_physical_alias_for_offline_analysis(module, tmp_path, values):
    probe, meta = begin_decode(module, tmp_path)
    probe.capture_topk(torch.tensor([[0, 0], [0, 1]]), meta)
    inputs = attention_inputs()
    inputs["key"].reshape(-1)[4:6] = torch.tensor(values)
    probe.attention_kv(inputs, 0, meta)
    archive = probe.archives["a"]
    row = record_for(archive, "kv_consumed", "nope")
    assert row["positions"] == [0, 0]
    assert row["physical_slots"] == [4, 5]
    torch.testing.assert_close(
        value_for(archive, "kv_consumed", "nope").flatten(), torch.tensor(values), equal_nan=True
    )
    assert record_for(archive, "attention", "logical_topk")["positions"] == [1]
    assert manifest(archive)["errors"] == []
    # Numerical observations must not prevent the remaining forward/sampling.
    fill_expected(probe, meta)
    finish_forward(probe, sampled=[[31], [41]])
    assert manifest(archive)["complete"]


def test_consumed_kv_deduplicates_only_the_same_logical_physical_pair(module, tmp_path):
    probe, meta = begin_decode(module, tmp_path)
    probe.capture_topk(torch.tensor([[0, 0], [0, 1]]), meta)
    inputs = attention_inputs()
    inputs["sparse_indices"][0] = torch.tensor([0, 0])
    probe.attention_kv(inputs, 0, meta)
    row = record_for(probe.archives["a"], "kv_consumed", "nope")
    assert row["positions"] == [0]
    assert row["physical_slots"] == [4]


def test_indexer_kv_reads_history_by_request_table(module, tmp_path):
    probe, _ = begin_decode(module, tmp_path)
    values = attention_inputs()
    probe.indexer_kv(values["key"], values["block_table"], torch.tensor([2, 1]), 0, "key")
    assert value_for(probe.archives["a"], "kv_indexer", "key").flatten().tolist() == [4, 5]
    assert value_for(probe.archives["b"], "kv_indexer", "key").flatten().tolist() == [0]


@pytest.mark.parametrize("limit,expected", [(128, 128), (16, 16), (0, 300)])
def test_sampling_slices_before_readback_and_preserves_feature_axes(module, tmp_path, monkeypatch, limit, expected):
    probe, _, meta = make_probe(module, tmp_path, {"a": list(range(300))}, max_token_rows=limit)
    ids, positions = batch(meta, [(list(range(300)), list(range(300)))])
    probe.start(ids, positions)
    copied = []
    original = module.cpu_tensor

    def observe(value):
        copied.append(tuple(value.shape))
        return original(value)

    monkeypatch.setattr(module, "cpu_tensor", observe)
    value = torch.arange(300 * 2 * 3).reshape(300, 2, 3)
    probe.emit(value, 0, "decoder", "input", meta)
    assert copied == [(expected, 2, 3)]
    archive = probe.archives["a"]
    record = record_for(archive, "decoder", "input")
    assert record["positions"] == list(range(expected))
    assert record["row_capture"]["source_rows"] == 300
    torch.testing.assert_close(value_for(archive, "decoder", "input"), value[:expected])
    # Sampling decision evidence has no token-row axis. Even a long vocabulary
    # or multi-draft vector must remain intact for offline acceptance analysis.
    entry = probe.active["entries"][0]
    probe.record(entry, torch.ones(1, 300), -1, "rejection", "target_logits")
    assert value_for(archive, "rejection", "target_logits").shape == (1, 300)


def test_sampling_is_per_request_and_per_tp_shard(module, tmp_path):
    cp = NS(local_start=100, local_end=400, local_end_with_pad=410)
    prompts = {"a": list(range(250)), "b": list(range(250))}
    probe, _, meta = make_probe(module, tmp_path, prompts, cp=cp)
    inputs, positions = batch(meta, [(list(range(250)), list(range(250)))] * 2)
    probe.start(inputs, positions)
    values = torch.arange(310).reshape(-1, 1)
    probe.emit(values, 0, "sfa", "input", meta)
    for request, first, source in (("a", 100, 150), ("b", 0, 150)):
        row = record_for(probe.archives[request], "sfa", "input")
        assert row["positions"] == list(range(first, first + 64))
        assert row["row_capture"]["source_rows"] == source
    assert value_for(probe.archives["b"], "sfa", "input")[0].item() == 150


@pytest.mark.parametrize("role", ["P", "D"])
def test_history_and_current_kv_gather_are_bounded_before_index_select(module, tmp_path, monkeypatch, role):
    probe, _, meta = make_probe(module, tmp_path, {"a": list(range(8192))})
    probe.role = role
    ids, positions = batch(meta, [(list(range(4096, 8192)), list(range(4096, 8192)))])
    probe.start(ids, positions)
    meta.slot_mapping = torch.arange(4096, 8192)
    cache = torch.arange(8192 * 2).reshape(2, 4096, 1, 2)
    selected_rows = []
    original = torch.Tensor.index_select

    def select(value, dim, indices):
        selected_rows.append(len(indices))
        return original(value, dim, indices)

    monkeypatch.setattr(torch.Tensor, "index_select", select)
    probe.current_kv([cache, cache], 0, meta)
    probe.indexer_kv(cache, torch.tensor([[0, 1]]), torch.tensor([8192]), 0, "key")
    assert selected_rows == [64, 64, 128]
    archive = probe.archives["a"]
    assert record_for(archive, "kv_current", "nope")["positions"] == list(range(4096, 4160))
    history = record_for(archive, "kv_indexer", "key")
    assert history["positions"] == list(range(64)) + list(range(4096, 4160))
    assert history["row_capture"]["source_rows"] == 8192
    assert [s["length"] for s in history["row_capture"]["segments"]] == [4096, 4096]


def test_topk_snapshot_is_bounded_and_survives_inplace_remapping(module, tmp_path, monkeypatch):
    probe, _, meta = make_probe(module, tmp_path, {"a": list(range(300))}, max_token_rows=2)
    ids, positions = batch(meta, [(list(range(297, 300)), list(range(297, 300)))])
    probe.start(ids, positions)
    indices = torch.arange(300).repeat(3, 1)
    copies, gathered = [], []
    original_cpu, original_select = module.cpu_tensor, torch.Tensor.index_select

    def cpu(value):
        copies.append(tuple(value.shape))
        return original_cpu(value)

    def select(value, dim, index):
        gathered.append(len(index))
        return original_select(value, dim, index)

    monkeypatch.setattr(module, "cpu_tensor", cpu)
    monkeypatch.setattr(torch.Tensor, "index_select", select)
    probe.sfa_stack.append((0, meta))
    # Exercise the real wrapper used by layers reusing an indexer result.
    assert probe.shared_topk_factory(lambda: indices)() is indices
    assert copies == [(2, 300)]
    indices[0].zero_()  # Production post-processing may mutate this storage.
    cache = torch.arange(300).reshape(1, 300, 1, 1)
    probe.attention_kv(
        dict(
            query=torch.zeros(3, 1, 1),
            sparse_indices=indices,
            block_table=torch.tensor([[0]]),
            actual_seq_lengths_kv=torch.tensor([300]),
            actual_seq_lengths_query=torch.tensor([3]),
            key=cache,
            key_rope=cache,
        ),
        0,
        meta,
    )
    archive = probe.archives["a"]
    logical = value_for(archive, "attention", "logical_topk")
    assert logical.shape == (2, 300) and logical[0, 1].item() == 1
    assert (3, 300) not in copies
    assert gathered == [2, 2]
    consumed = record_for(archive, "kv_consumed", "nope")
    assert len(consumed["positions"]) == 2
    assert consumed["row_capture"]["query_rows"]["source_rows"] == 3
    assert consumed["row_capture"]["query_rows"]["saved_rows"] == 2


def test_negative_row_limit_is_rejected_before_install(module, tmp_path):
    with pytest.raises(ValueError, match="integer in"):
        make_probe(module, tmp_path, {"a": [1]}, max_token_rows=-1)


def test_fixed_chunks_preserve_same_kv_offsets_across_p_and_d_and_partial_tail(module, tmp_path):
    copied = []
    for role in ("P", "D"):
        probe, _, meta = make_probe(module, tmp_path / role, {"a": list(range(8200))})
        probe.role = role
        query = list(range(8192, 8200)) if role == "P" else [8199]
        ids, positions = batch(meta, [(query, query)])
        probe.start(ids, positions)
        cache = torch.arange(12288).reshape(3, 4096, 1, 1)
        # A zero-filled later physical block must never be mistaken for a
        # complete 4096-token logical tail: only eight positions are live.
        probe.indexer_kv(cache, torch.tensor([[0, 1, 2]]), torch.tensor([8200]), 0, "key")
        row = record_for(probe.archives["a"], "kv_indexer", "key")
        assert row["positions"] == list(range(64)) + list(range(4096, 4160)) + list(range(8192, 8200))
        assert [s["length"] for s in row["row_capture"]["segments"]] == [4096, 4096, 8]
        assert row["shape"][0] == 136
        # Query capture uses visible rows: D's last prompt token must survive
        # even if it lies outside a fixed KV sample window.
        assert record_for(probe.archives["a"], "model_input", "positions")["positions"] == query
        copied.append(value_for(probe.archives["a"], "kv_indexer", "key"))
    torch.testing.assert_close(*copied)
    assert module.sample_rows(range(4000, 4096), 64, 4096, aligned=True)[0] == []
    assert module.sample_rows(range(4000, 4096), 64, 4096)[0] == list(range(4000, 4064))


@pytest.mark.parametrize("features", [0, 8])
def test_feature_capture_keeps_all_token_and_head_positions_before_readback(module, tmp_path, monkeypatch, features):
    probe, _, meta = make_probe(module, tmp_path, {"a": list(range(400))}, max_token_rows=0, max_features=features)
    ids, positions = batch(meta, [(list(range(400)), list(range(400)))])
    probe.start(ids, positions)
    copied = []
    original = module.cpu_tensor

    def copy(value):
        copied.append((tuple(value.shape), value.is_contiguous()))
        return original(value)

    monkeypatch.setattr(module, "cpu_tensor", copy)
    value = torch.arange(400 * 16 * 32).reshape(400, 16, 32).float()
    probe.emit(value, 0, "attention", "query_nope", meta)
    width = features or 32
    assert copied == [((400, 16, width), True)]
    archive = probe.archives["a"]
    row = record_for(archive, "attention", "query_nope")
    assert row["positions"] == list(range(400))
    if features:
        assert row["feature_capture"] == dict(axis=2, start=0, source_size=32, saved_size=8)
    else:
        assert "feature_capture" not in row
    torch.testing.assert_close(value_for(archive, "attention", "query_nope"), value[..., :width])


@pytest.mark.parametrize(
    "kind,name",
    [
        ("attention", "logical_topk"),
        ("indexer", "topk"),
        ("mapping", "sparse_indices"),
        ("model_input", "input_ids"),
        ("model_input", "positions"),
        ("logits", "output"),
        ("rejection", "target_logits"),
        ("rejection", "draft_probs"),
    ],
)
def test_feature_cap_never_changes_routing_or_mtp_acceptance_evidence(module, kind, name):
    value = torch.arange(64).reshape(2, 32).float()
    captured, metadata = module.feature_prefix(value, kind, name, 8)
    assert captured is value and metadata is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test_feature_cap_is_applied_before_kv_gather_and_preserves_quantized_features(module, tmp_path, monkeypatch, dtype):
    probe, _, meta = make_probe(module, tmp_path, {"a": list(range(200))}, max_token_rows=0)
    ids, positions = batch(meta, [(list(range(200)), list(range(200)))])
    probe.start(ids, positions)
    meta.slot_mapping = torch.arange(200)
    cache = torch.arange(200 * 16 * 32).reshape(1, 200, 16, 32).to(dtype)
    original = torch.Tensor.index_select
    gathered = []

    def select(value, dim, indices):
        gathered.append((tuple(value.shape), len(indices)))
        return original(value, dim, indices)

    monkeypatch.setattr(torch.Tensor, "index_select", select)
    probe.current_kv([cache, cache], 0, meta)
    probe.indexer_kv(cache, torch.tensor([[0]]), torch.tensor([200]), 0, "key")
    assert gathered == [((200, 16, 8), 200)] * 3
    for kind, name in (("kv_current", "nope"), ("kv_indexer", "key")):
        row = record_for(probe.archives["a"], kind, name)
        assert row["feature_capture"]["source_size"] == 32
        torch.testing.assert_close(value_for(probe.archives["a"], kind, name), cache.reshape(200, 16, 32)[..., :8])


def test_decode_topk_kv_outside_old_64_row_window_is_captured(module, tmp_path):
    probe, _, meta = make_probe(module, tmp_path, {"a": list(range(200))}, max_token_rows=0)
    ids, positions = batch(meta, [([199], [199])])
    probe.start(ids, positions)
    topk = torch.tensor([[99, 100, 199]])
    probe.capture_topk(topk, meta)
    cache = torch.arange(200 * 16).reshape(1, 200, 1, 16).float()
    probe.attention_kv(
        dict(
            query=torch.zeros(1, 1, 16),
            sparse_indices=topk,
            block_table=torch.tensor([[0]]),
            actual_seq_lengths_kv=torch.tensor([200]),
            actual_seq_lengths_query=torch.tensor([1]),
            key=cache,
            key_rope=cache,
        ),
        0,
        meta,
    )
    archive = probe.archives["a"]
    row = record_for(archive, "kv_consumed", "nope")
    assert row["positions"] == [99, 100, 199]
    assert row["physical_slots"] == [99, 100, 199]
    assert row["feature_capture"]["saved_size"] == 8
    torch.testing.assert_close(value_for(archive, "kv_consumed", "nope"), cache[0, [99, 100, 199], :, :8])


@pytest.mark.parametrize("rows", [0, 1, 10000])
def test_vectorized_pair_dedup_matches_torch_including_large_addresses(module, rows):
    generator = torch.Generator().manual_seed(7)
    positions = torch.randint(0, 100, (rows,), generator=generator, dtype=torch.long)
    slots = torch.randint(0, 200, (rows,), generator=generator, dtype=torch.long) + 2**62
    expected = torch.unique(torch.stack((positions, slots), dim=1), dim=0, sorted=True)
    actual = module.unique_kv_pairs(positions, slots)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("local_lengths", [[0, 0], [1, 1]])
def test_p_indexer_captures_global_computed_prefix_not_local_or_future_rows(module, tmp_path, local_lengths):
    cp = NS(local_start=4, local_end=4, local_end_with_pad=6)
    probe, _, meta = make_probe(module, tmp_path, {"a": [10, 11, 12, 13], "b": [20, 21, 22, 23]}, cp=cp)
    probe.role = "P"
    ids, positions = batch(meta, [([0, 1], [10, 11]), ([0, 1], [20, 21])])
    probe.start(ids, positions)
    values = attention_inputs()
    probe.indexer_kv(values["key"], values["block_table"], torch.tensor(local_lengths), 0, "key")
    for key, expected in (("a", [4, 5]), ("b", [0, 1])):
        archive = probe.archives[key]
        assert record_for(archive, "kv_indexer", "key")["positions"] == [0, 1]
        assert value_for(archive, "kv_indexer", "key").flatten().tolist() == expected
        assert len(archive.metadata["prompt_token_ids"]) == 4
        assert len(archive.calls[0]["context_token_ids"]) == 2


@pytest.mark.parametrize("slot", [-1, 6])
def test_current_kv_checks_actual_write_slots(module, tmp_path, slot):
    probe, meta = begin_decode(module, tmp_path)
    meta.slot_mapping = torch.tensor([slot, 0])
    with pytest.raises(ValueError, match="Current KV slot out of bounds"):
        probe.current_kv([attention_inputs()["key"]] * 2, 0, meta)


@pytest.mark.parametrize("cp", [False, True])
def test_current_kv_reads_all_global_slots_after_cp_gather(module, tmp_path, cp):
    probe, meta = begin_decode(module, tmp_path)
    if cp:
        meta.dsa_cp_context = NS(local_start=1, local_end=2, local_end_with_pad=3)
    # This rank's query shard owns only B in CP mode, but gathered current KV
    # includes both requests. Padding slots must not become archived rows.
    meta.slot_mapping = torch.tensor([5, 1, -1, -1])
    cache = [torch.zeros(3, 2, 1, 1), torch.zeros(3, 2, 1, 1)]
    for part in cache:
        part.reshape(-1)[torch.tensor([5, 1])] = torch.tensor([4.0, 8.0])
    probe.current_kv(cache, 0, meta)
    assert value_for(probe.archives["a"], "kv_current", "nope").flatten().tolist() == [4]
    assert value_for(probe.archives["b"], "kv_current", "rope").flatten().tolist() == [8]
    assert record_for(probe.archives["a"], "kv_current", "nope")["positions"] == [1]


def test_current_kv_rejects_truncated_global_slot_mapping(module, tmp_path):
    probe, meta = begin_decode(module, tmp_path)
    meta.slot_mapping = torch.tensor([0])
    with pytest.raises(ValueError, match="misses scheduled query rows"):
        probe.current_kv([attention_inputs()["key"]] * 2, 0, meta)


def test_logits_failure_marks_already_finished_forward_incomplete(module, tmp_path):
    probe, runner, meta = make_probe(module, tmp_path, {"a": [10]})
    ids, positions = batch(meta, [([0], [10])])
    with probe.forward(ids, positions):
        fill_expected(probe, meta)
    runner.logits_indices = torch.tensor([0])

    def logits():
        raise RuntimeError("logits failed")

    with pytest.raises(RuntimeError, match="logits failed"):
        probe.logits_factory(logits)()
    state = manifest(probe.archives["a"])
    assert not state["complete"]
    assert "logits failed" in state["errors"][-1]


def test_kernels_outside_main_forward_do_not_capture_mtp_draft(module, tmp_path):
    probe, _, _ = make_probe(module, tmp_path, {"a": [10]})
    result = object()
    calls = []

    def kernel(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    wrapped = probe.kernel_factory("indexer")(kernel)
    assert wrapped("draft", token=10) is result
    assert calls == [(("draft",), {"token": 10})]
    assert not list(tmp_path.rglob("*.pt"))


@pytest.mark.parametrize("sample_ids", [["a", "other"], ["a", "a"]])
def test_sampling_missing_or_duplicate_request_identity_fails_closed(module, tmp_path, sample_ids):
    probe, runner, meta = make_probe(module, tmp_path, {"a": [10], "b": [20]})
    ids, positions = batch(meta, [([0], [10]), ([0], [20])])
    with probe.forward(ids, positions):
        fill_expected(probe, meta)
    runner.logits_indices = torch.tensor([0, 1])
    probe.logits_factory(lambda: torch.zeros(2, 3))()
    with pytest.raises(ValueError, match="identities"):
        probe.sampled(sample_ids, [[31], [41]])
    for archive in probe.archives.values():
        assert not manifest(archive)["complete"]
        assert "sampling" in manifest(archive)["errors"][-1]


def test_jointly_missing_decoder_attention_layer_is_rejected(module, tmp_path):
    probe, runner, _ = make_probe(module, tmp_path, {"a": [10]})

    class TestDecoderLayer:
        pass

    class SFA:
        pass

    probe.layers.clear()
    probe.attentions.clear()
    runner.model_config.hf_text_config = NS(num_hidden_layers=2)
    runner.model = NS(
        named_modules=lambda: [
            ("model.layers.0", TestDecoderLayer()),
            ("model.layers.0.attention", NS(impl=SFA(), layer_name="model.layers.0.attention")),
        ]
    )
    with pytest.raises(ValueError, match="inventory|layer"):
        probe.install(SFA, NS())
    assert not probe.patches and not probe.handles


def test_real_archive_can_be_analyzed_after_sample_and_request_finish(module, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("_pd_analyzer_integration", ROOT / "tools/pd_tensor_analyze.py")
    analyzer = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, analyzer)
    spec.loader.exec_module(analyzer)
    roots = [tmp_path / "off", tmp_path / "on"]
    for root in roots:
        probe, _, meta = make_probe(module, root, {"a": [10, 11]})
        probe.rank_info.update(tp_rank=0, tp_size=1, dp_rank=0, dp_size=1)
        ids, positions = batch(meta, [([0, 1], [10, 11])])
        with probe.forward(ids, positions):
            fill_expected(probe, meta)
        finish_forward(probe, sampled=[[31]])
        probe.observe_scheduler(NS(scheduled_new_reqs=[], finished_req_ids={"a"}, num_scheduled_tokens={}))
    report = analyzer.analyze([roots[0]], [roots[1]], mode="off-on", output=tmp_path / "report")
    assert report["status"] == "analysis_complete"
    assert report["issues"] == 0
    assert report["compared_tensors"] > 0
    assert report["first_difference"] is None


@pytest.mark.parametrize("binding", ["upstream", "ascend"])
@pytest.mark.parametrize("random_sampling,corrupt", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("reject_at", [0, 1, 2])
def test_rejection_observes_actual_processed_logits_rng_and_outputs(
    module, tmp_path, monkeypatch, binding, random_sampling, corrupt, reject_at
):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import pd_tensor_analyze as analyzer

    probe, runner, main_meta = make_probe(module, tmp_path, {"a": [10, 11, 12], "b": [20, 21], "c": [30]})
    probe.rank_info.update(tp_rank=0, tp_size=1, dp_rank=0, dp_size=1)
    ids, positions = batch(main_meta, [([0, 1, 2], [10, 11, 12]), ([0, 1], [20, 21]), ([0], [30])])
    with probe.forward(ids, positions):
        fill_expected(probe, main_meta)
    metadata = NS(
        num_draft_tokens=[2, 1, 0],
        max_spec_len=2,
        draft_token_ids=torch.tensor([1, 2, 2]),
        target_logits_indices=torch.tensor([0, 1, 3]),
        bonus_logits_indices=torch.tensor([2, 4, 5]),
        logits_indices=torch.arange(6),
        cu_num_draft_tokens=torch.tensor([2, 3, 3]),
    )
    sampling = NS(all_greedy=not random_sampling, temperature=torch.full((3,), 0.7 if random_sampling else 0.0))
    # Exercise the real CPU greedy rejection implementation, without NPU imports.
    source = ast.parse((ROOT / "vllm_ascend/sample/rejection_sampler.py").read_text())
    node = next(
        n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "rejection_greedy_sample_pytorch"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "rejection_greedy", "exec"), namespace)
    ascend = NS(
        generate_uniform_probs=lambda n: torch.rand(n),
        sample_recovered_tokens=lambda target_probs: target_probs.argmax(dim=-1),
    )
    returned = []

    def kernel(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        target_logits,
        bonus_token_ids,
        sampling_metadata,
    ):
        if random_sampling:
            ascend.generate_uniform_probs(len(draft_token_ids))
            ascend.sample_recovered_tokens(target_probs=target_logits.softmax(dim=-1))
        output = torch.full((3, 3), -1, dtype=torch.int32)
        namespace["rejection_greedy_sample_pytorch"](
            output,
            cu_num_draft_tokens,
            draft_token_ids,
            target_logits.argmax(dim=-1),
            bonus_token_ids,
            num_draft_tokens,
            max_spec_len,
        )
        if corrupt:
            output[0, 0] = 4  # Must be reported offline, not asserted by the probe.
        returned.append(output)
        return output

    ascend.rejection_sample = kernel
    upstream = NS(rejection_sample=kernel, PLACEHOLDER_TOKEN_ID=-1)
    owner = upstream if binding == "upstream" else ascend

    class Sampler(torch.nn.Module):
        def forward(self, metadata, draft_probs, logits, sampling_metadata):
            target = logits[metadata.target_logits_indices].clone()
            winners = [1, 2, 2]
            if reject_at < 2:
                winners[reject_at] = 3
            target[torch.arange(3), torch.tensor(winners)] += 10  # Actual postprocessor result.
            result = owner.rejection_sample(
                metadata.draft_token_ids,
                metadata.num_draft_tokens,
                metadata.max_spec_len,
                metadata.cu_num_draft_tokens,
                draft_probs,
                target,
                torch.tensor([[4], [4], [4]], dtype=torch.int32),
                sampling_metadata,
            )
            return NS(sampled_token_ids=result)

    runner.rejection_sampler = Sampler()
    logits = torch.zeros(6, 5)
    rng = torch.get_rng_state()
    plain = runner.rejection_sampler(metadata, None, logits, sampling)
    after_plain = torch.get_rng_state()
    torch.set_rng_state(rng)
    probe.install_rejection(ascend, upstream)
    result = runner.rejection_sampler(metadata, None, logits, sampling)
    assert result.sampled_token_ids is returned[-1]
    torch.testing.assert_close(result.sampled_token_ids, plain.sampled_token_ids)
    assert torch.equal(after_plain, torch.get_rng_state())
    used = [[t for t in row.tolist() if t >= 0] for row in result.sampled_token_ids]
    used[2] = []  # C is an unfinished prefill row.
    finish_forward(probe, sampled=used)
    probe.observe_scheduler(NS(scheduled_new_reqs=[], finished_req_ids={"a", "b", "c"}, num_scheduled_tokens={}))
    archive = analyzer.load_archive([tmp_path])
    assert not archive.issues
    decisions = [d for worker in archive.workers.values() for d in analyzer._rejection_decisions(worker)]
    by_request = {d["request_id"]: d for d in decisions}
    assert by_request["external-a"]["accepted_drafts"] == reject_at
    assert by_request["external-a"]["first_rejected_index"] == (reject_at if reject_at < 2 else None)
    statuses = ["accepted"] * reject_at + (["rejected"] + ["not_reached"] * (1 - reject_at) if reject_at < 2 else [])
    assert by_request["external-a"]["draft_status"] == statuses
    assert by_request["external-b"]["first_rejected_index"] is None
    assert not by_request["external-c"]["counted_for_acceptance"]
    if random_sampling:
        assert by_request["external-a"]["greedy_decision_consistent"] is None
        assert by_request["external-a"]["random_evidence"] == ["target_probs", "uniform_probs", "recovered_token_ids"]
    else:
        assert by_request["external-a"]["greedy_decision_consistent"] is not corrupt
    worker = next(w for w in archive.workers.values() if w.manifest["request_id"] == "external-a")
    rejected_records = {r["name"]: r for r in worker.records if r["kind"] == "rejection"}
    assert analyzer._load_tensor(worker, rejected_records["raw_target_logits"]).sum() == 0
    assert analyzer._load_tensor(worker, rejected_records["target_logits"]).sum() == 20
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    summary = analyzer._write_rejection_report(archive, archive, report_dir)
    assert len(summary["workers"]) == 3  # Same collected directory is not counted twice.
    first = next(w for w in summary["workers"] if w["request_id"] == "external-a")
    assert first["acceptance_rate"] == reject_at / 2
    assert first["greedy_inconsistent_calls"] == int(corrupt)
    worker.records.remove(rejected_records["kernel_output"])
    assert next(analyzer._rejection_decisions(worker))["status"] == "incomplete"
    probe.restore()
    assert upstream.rejection_sample is kernel and ascend.rejection_sample is kernel
