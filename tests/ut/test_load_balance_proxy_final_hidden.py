# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from examples.disaggregated_prefill_v1.load_balance_proxy_server_example import (
    build_prefill_kv_transfer_params,
    update_decode_kv_transfer_params,
)


def test_prefill_transfer_params_enable_final_hidden_handoff() -> None:
    params = build_prefill_kv_transfer_params({"old-request"}, True)

    assert params["do_remote_decode"]
    assert not params["do_remote_prefill"]
    assert params["aborted_request"] == ["old-request"]
    assert params["ret_final_hidden"] is True


def test_prefill_transfer_params_keep_legacy_behavior_by_default() -> None:
    params = build_prefill_kv_transfer_params(set(), False)

    assert "ret_final_hidden" not in params


def test_empty_prefill_response_clears_stale_transfer_envelope() -> None:
    req_data = {
        "prompt": "new prompt",
        "kv_transfer_params": {
            "remote_engine_id": "old-engine",
            "bootstrap_final_hidden": {"prompt_sha256": "old"},
        },
    }

    params = update_decode_kv_transfer_params(req_data, None)

    assert params == {}
    assert "kv_transfer_params" not in req_data
