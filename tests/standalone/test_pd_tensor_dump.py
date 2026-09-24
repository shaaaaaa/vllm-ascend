# SPDX-License-Identifier: Apache-2.0
"""CPU observations of the real PD recorder, without importing NPU packages."""

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
    return loaded


def make_probe(module, tmp_path, prompts, *, cp=None, external=None):
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
    )
    probe.layers = {0: object()}
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


def records(archive):
    path = archive.root / "index.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


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
    probe.logical_topk = torch.tensor([[0, 1], [0, 1]])
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


def test_consumed_kv_conflicting_logical_aliases_fail_closed(module, tmp_path):
    probe, meta = begin_decode(module, tmp_path)
    probe.logical_topk[0] = torch.tensor([0, 0])
    with pytest.raises(ValueError, match="aliased physical slots"):
        probe.attention_kv(attention_inputs(), 0, meta)


def test_indexer_kv_reads_history_by_request_table(module, tmp_path):
    probe, _ = begin_decode(module, tmp_path)
    values = attention_inputs()
    probe.indexer_kv(values["key"], values["block_table"], torch.tensor([2, 1]), 0, "key")
    assert value_for(probe.archives["a"], "kv_indexer", "key").flatten().tolist() == [4, 5]
    assert value_for(probe.archives["b"], "kv_indexer", "key").flatten().tolist() == [0]


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
