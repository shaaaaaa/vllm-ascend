import json
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.worker.model_runner_v1 as model_runner_v1
from vllm_ascend.worker.model_runner_v1 import (
    ExecuteModelState,
    NPUModelRunner,
    deserialize_final_hidden_state,
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
    )

    restored = deserialize_final_hidden_state(payloads["request-b"])
    assert torch.equal(restored, hidden_states[4])


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


def test_compact_decode_residency_is_logged_once_per_live_request():
    runner = object.__new__(NPUModelRunner)
    runner.dsa_two_groups = True
    request_state = SimpleNamespace(
        num_computed_tokens=120_000,
        num_prompt_tokens=120_000,
        block_ids=([1, 2, 0, 0, 9, 10], [101, 102, 103, 104]),
        compact_residency_log_pending=True,
    )
    runner.requests = {"request": request_state}
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"request": 1})

    runner._log_compact_decode_residency_once(scheduler_output)
    assert request_state.compact_residency_logged
    assert not request_state.compact_residency_log_pending

    # The second decode step must not repeat the request-level residency log.
    runner._log_compact_decode_residency_once(scheduler_output)
    assert request_state.compact_residency_logged


def test_normal_prefill_does_not_enter_compact_residency_path():
    runner = object.__new__(NPUModelRunner)
    runner.dsa_two_groups = True
    request_state = SimpleNamespace(
        num_computed_tokens=4096,
        num_prompt_tokens=120_000,
        block_ids=([1, 2, 3, 4], [101, 102, 103, 104]),
    )
    runner.requests = {"request": request_state}

    runner._log_compact_decode_residency_once(
        SimpleNamespace(num_scheduled_tokens={"request": 4096})
    )

    assert not hasattr(request_state, "compact_residency_logged")


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
        final_hidden_states={},
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
