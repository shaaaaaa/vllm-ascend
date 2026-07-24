import json
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.worker.model_runner_v1 as model_runner_v1
from vllm_ascend.worker.model_runner_v1 import (
    ExecuteModelState,
    NPUModelRunner,
    deserialize_final_hidden_state,
    final_hidden_parity_summary,
    final_hidden_row_summary,
    final_hidden_value_summary,
    serialize_final_hidden_state,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_final_hidden_payload_round_trip_is_bit_exact(dtype):
    hidden_state = torch.linspace(-4, 4, 31, dtype=dtype)

    payload = serialize_final_hidden_state(hidden_state)
    json.dumps(payload)
    restored = deserialize_final_hidden_state(payload)

    assert restored.dtype == dtype
    assert restored.shape == hidden_state.shape
    assert torch.equal(restored.view(torch.uint8), hidden_state.view(torch.uint8))


def test_final_hidden_payload_rejects_wrong_data_size():
    payload = serialize_final_hidden_state(torch.ones(8, dtype=torch.bfloat16))
    payload["shape"] = [9]

    with pytest.raises(ValueError, match="payload size mismatch"):
        deserialize_final_hidden_state(payload)


def test_final_hidden_payload_rejects_checksum_mismatch():
    payload = serialize_final_hidden_state(torch.ones(8, dtype=torch.bfloat16))
    payload["data_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        deserialize_final_hidden_state(payload)


def test_final_hidden_payload_rejects_oversize_before_decode():
    payload = serialize_final_hidden_state(torch.ones(1, dtype=torch.bfloat16))
    payload["shape"] = [model_runner_v1._MAX_FINAL_HIDDEN_BYTES // 2 + 1]

    with pytest.raises(ValueError, match="16 MiB decoded size limit"):
        deserialize_final_hidden_state(payload)


def test_final_hidden_parity_summary_reports_exact_match():
    hidden_states = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    logits = torch.tensor([[0.1, 0.7, 0.2]], dtype=torch.float32)

    summary = final_hidden_parity_summary(
        hidden_states,
        hidden_states.clone(),
        logits,
        logits.clone(),
    )

    assert summary == {
        "hidden_exact": [True],
        "hidden_max_abs": [0.0],
        "logits_exact": [True],
        "logits_max_abs": [0.0],
        "transferred_top_ids": [[1, 2, 0]],
        "reference_top_ids": [[1, 2, 0]],
        "top1_match": [True],
    }


def test_final_hidden_parity_summary_reports_top1_mismatch():
    transferred_hidden = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    reference_hidden = torch.tensor([[1.0, 3.0]], dtype=torch.bfloat16)
    transferred_logits = torch.tensor([[0.8, 0.1]], dtype=torch.float32)
    reference_logits = torch.tensor([[0.1, 0.8]], dtype=torch.float32)

    summary = final_hidden_parity_summary(
        transferred_hidden,
        reference_hidden,
        transferred_logits,
        reference_logits,
    )

    assert summary["hidden_exact"] == [False]
    assert summary["hidden_max_abs"] == [1.0]
    assert summary["logits_exact"] == [False]
    assert summary["logits_max_abs"] == pytest.approx([0.7])
    assert summary["transferred_top_ids"] == [[0, 1]]
    assert summary["reference_top_ids"] == [[1, 0]]
    assert summary["top1_match"] == [False]


def test_final_hidden_value_summary_counts_nonfinite_values():
    values = torch.tensor(
        [1.5, float("nan"), float("inf"), -float("inf")],
        dtype=torch.float32,
    )

    summary = final_hidden_value_summary(values)

    assert summary == {
        "device": "cpu",
        "dtype": "torch.float32",
        "shape": [4],
        "numel": 4,
        "finite_count": 1,
        "nan_count": 1,
        "posinf_count": 1,
        "neginf_count": 1,
        "finite_abs_max": 1.5,
    }


def test_final_hidden_row_summary_reports_nonfinite_row_classes():
    values = torch.tensor(
        [
            [1.0, 2.0],
            [float("nan"), 3.0],
            [float("nan"), float("inf")],
        ],
        dtype=torch.float32,
    )

    assert final_hidden_row_summary(values) == {
        "rows": 3,
        "row_width": 2,
        "fully_finite_rows": 1,
        "partially_finite_rows": 1,
        "fully_nonfinite_rows": 1,
        "first_nonfinite_row_indices": [1, 2],
    }


def test_capture_final_hidden_uses_last_scheduled_row_per_request():
    runner = object.__new__(NPUModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["request-a", "request-b"])
    scheduler_output = SimpleNamespace(
        capture_final_hidden_req_ids={"request-b"}
    )
    hidden_states = torch.arange(20, dtype=torch.float32).reshape(5, 4)

    payloads = runner._capture_final_hidden_states(
        scheduler_output,
        hidden_states,
        num_scheduled_tokens=torch.tensor([2, 3]).numpy(),
        logits_indices=torch.tensor([1, 4]),
    )

    restored = deserialize_final_hidden_state(payloads["request-b"])
    assert torch.equal(restored, hidden_states[4])


def test_capture_diagnostics_compare_legacy_and_sampler_rows(monkeypatch):
    monkeypatch.setattr(
        model_runner_v1.envs_ascend,
        "VLLM_ASCEND_FINAL_HIDDEN_PARITY_TRACE",
        True,
    )
    runner = object.__new__(NPUModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["request-a", "request-b"])
    scheduler_output = SimpleNamespace(
        capture_final_hidden_req_ids={"request-b"}
    )
    hidden_states = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [float("nan"), float("nan")],
        ]
    )

    payloads = runner._capture_final_hidden_states(
        scheduler_output,
        hidden_states,
        num_scheduled_tokens=torch.tensor([2, 3]).numpy(),
        logits_indices=torch.tensor([1, 3]),
    )

    diagnostics = payloads["request-b"]["capture_diagnostics"]
    assert diagnostics["legacy_row_index"] == 4
    assert diagnostics["sampler_row_index"] == 3
    assert diagnostics["legacy_npu_before_serialize"]["nan_count"] == 2
    assert diagnostics["sampler_npu_before_serialize"]["nan_count"] == 0
    assert diagnostics["legacy_npu_after_serialize"]["nan_count"] == 2
    assert diagnostics["sampler_npu_after_serialize"]["nan_count"] == 0
    assert diagnostics["payload_cpu"]["nan_count"] == 2
    assert not diagnostics["legacy_equals_sampler"]
    assert diagnostics["legacy_sha256"] != diagnostics["sampler_sha256"]


@pytest.mark.parametrize(
    ("pp_last", "tp_rank", "expected"),
    [(True, 0, True), (True, 1, False), (False, 0, False)],
)
def test_final_hidden_capture_is_limited_to_executor_output_rank(
    monkeypatch, pp_last, tp_rank, expected
):
    monkeypatch.setattr(
        model_runner_v1,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=pp_last),
    )
    monkeypatch.setattr(
        model_runner_v1,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=tp_rank),
    )

    assert NPUModelRunner._is_final_hidden_output_rank() is expected


def test_wait_for_bootstrap_kv_load_advances_every_group0_layer(monkeypatch):
    calls = []
    connector = SimpleNamespace(wait_for_layer_load=calls.append)
    monkeypatch.setattr(
        model_runner_v1,
        "get_kv_transfer_group",
        lambda: connector,
    )
    runner = object.__new__(NPUModelRunner)
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["layer.0", "layer.1"])]
    )
    runner.dsa_two_groups = False

    runner._wait_for_bootstrap_kv_load()

    assert calls == ["layer.0", "layer.1"]


def test_wait_for_bootstrap_kv_load_interleaves_dsa_groups(monkeypatch):
    calls = []
    connector = SimpleNamespace(wait_for_layer_load=calls.append)
    monkeypatch.setattr(
        model_runner_v1,
        "get_kv_transfer_group",
        lambda: connector,
    )
    runner = object.__new__(NPUModelRunner)
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["latent.0", "latent.1"]),
            SimpleNamespace(layer_names=["indexer.0", "indexer.1"]),
        ]
    )
    runner.dsa_two_groups = True

    runner._wait_for_bootstrap_kv_load()

    assert calls == ["latent.0", "indexer.0", "latent.1", "indexer.1"]


def test_bootstrap_load_failure_only_matches_current_request_blocks():
    runner = object.__new__(NPUModelRunner)
    runner.requests = {
        "bootstrap": SimpleNamespace(block_ids=([1, 0, 0, 2], [11, 12])),
    }
    scheduler_output = SimpleNamespace(
        bootstrap_sample_req_ids={"bootstrap"}
    )

    unrelated = SimpleNamespace(invalid_block_ids={99})
    assert not runner._bootstrap_load_failed_req_ids(
        scheduler_output, unrelated
    )

    null_block_only = SimpleNamespace(invalid_block_ids={0})
    assert not runner._bootstrap_load_failed_req_ids(
        scheduler_output, null_block_only
    )

    current_request_failed = SimpleNamespace(invalid_block_ids={12, 99})
    assert runner._bootstrap_load_failed_req_ids(
        scheduler_output, current_request_failed
    ) == {"bootstrap"}


def test_bootstrap_load_failure_is_synchronized_across_tp(monkeypatch):
    runner = object.__new__(NPUModelRunner)
    cpu_group = object()
    monkeypatch.setattr(
        model_runner_v1,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=8, cpu_group=cpu_group),
    )

    def all_reduce(flags, op, group):
        assert op == torch.distributed.ReduceOp.MAX
        assert group is cpu_group
        flags[1] = 1

    monkeypatch.setattr(model_runner_v1.dist, "all_reduce", all_reduce)

    assert runner._sync_tp_bootstrap_load_failed_req_ids(
        ["request-a", "request-b"], {"request-a"}
    ) == {"request-a", "request-b"}


def test_bootstrap_last_token_input_state_restores_on_error():
    runner = object.__new__(NPUModelRunner)
    computed = torch.tensor([120_000, 80_000], dtype=torch.int32).numpy()
    runner.input_batch = SimpleNamespace(num_computed_tokens_cpu=computed)

    with pytest.raises(RuntimeError, match="prepare failed"):
        with runner._bootstrap_last_token_input_state(num_reqs=2):
            assert computed.tolist() == [119_999, 79_999]
            raise RuntimeError("prepare failed")

    assert computed.tolist() == [120_000, 80_000]


@pytest.mark.parametrize("method", ["mtp", "deepseek_mtp"])
def test_bootstrap_accepts_mtp_method_aliases(method):
    runner = object.__new__(NPUModelRunner)
    runner.speculative_config = SimpleNamespace(
        method=method,
        num_speculative_tokens=1,
    )

    runner._validate_bootstrap_spec_config()


def test_bootstrap_rejects_other_speculative_config():
    runner = object.__new__(NPUModelRunner)
    runner.speculative_config = SimpleNamespace(
        method="eagle",
        num_speculative_tokens=1,
    )

    with pytest.raises(RuntimeError, match="supports only MTP"):
        runner._validate_bootstrap_spec_config()


def test_bootstrap_runs_draft_and_finalizes_connector(monkeypatch):
    runner = object.__new__(NPUModelRunner)
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1,
        bootstrap_sample_req_ids={"request"},
    )
    logits = torch.zeros(1, 8)
    hidden_states = torch.zeros(1, 4)
    runner.execute_model_state = ExecuteModelState(
        scheduler_output=scheduler_output,
        logits=logits,
        spec_decode_metadata=None,
        spec_decode_common_attn_metadata=SimpleNamespace(),
        hidden_states=hidden_states,
        sample_hidden_states=hidden_states,
        aux_hidden_states=None,
        attn_metadata={},
        positions=torch.zeros(1, dtype=torch.int64),
        ec_connector_output=None,
        cudagraph_stats=None,
        batch_desc=None,
    )
    runner.kv_connector_output = SimpleNamespace()
    runner.speculative_config = SimpleNamespace(
        method="deepseek_mtp",
        use_eagle=lambda: False,
        uses_draft_model=lambda: False,
        disable_padded_drafter_batch=False,
    )
    runner.need_accepted_tokens = False
    runner.model_config = SimpleNamespace(enable_return_routed_experts=False)
    runner.supports_mm_inputs = False
    runner.dynamic_eplb = False
    runner.debugger = None
    runner.use_async_scheduling = False
    runner.input_batch = SimpleNamespace(sampling_metadata=SimpleNamespace())
    runner._draft_token_ids = [[123]]
    runner._sample = lambda logits, metadata: SimpleNamespace(
        sampled_token_ids=torch.tensor([[7]]),
        logprobs_tensors=None,
    )
    runner._bookkeeping_sync = lambda *args: (
        None,
        [[7]],
        {},
        ["request"],
        {"request": 0},
        [],
    )
    propose_calls = []
    runner.propose_draft_token_ids = (
        lambda *args: propose_calls.append(args) or [[8]]
    )
    runner._copy_draft_token_ids_to_cpu = lambda scheduler_output: None
    finalize_calls = []
    runner.finalize_kv_connector = lambda: finalize_calls.append(True) or {}
    monkeypatch.setattr(model_runner_v1, "has_kv_transfer_group", lambda: True)

    runner.sample_tokens(grammar_output=None)

    assert len(propose_calls) == 1
    assert propose_calls[0][7] is hidden_states
    assert propose_calls[0][9] is hidden_states
    assert runner._draft_token_ids == [[8]]
    assert finalize_calls == [True]
