# SPDX-License-Identifier: Apache-2.0
"""Check the query bitmap and fixed-width remap against the position oracle."""

from collections import Counter

import pytest
import torch
from benchmark import stage_snapshot
from resident_experiment import Case, assert_result, make_case, membership_reference, reference


def _put_row(case, row, tokens):
    case["topk"][row].fill_(-1)
    if tokens:
        case["topk"][row, 0, :len(tokens)] = torch.tensor(tokens, dtype=torch.int32)


def _boundary_case(history):
    case = make_case(1, 2, 4, scenario="cold")
    # All history in shard 0; the repeated token is placed across sort/compare
    # boundaries for selected sizes 32, 64, 96, etc.
    tokens = [8192 + case.shards * i for i in range(history)]
    _put_row(case, 0, [16_777_219, *reversed(tokens)])
    _put_row(case, 1, [*tokens[-1:], 16_777_221])
    case["boundary"].fill_(16_777_216)
    return case


def test_reference_uses_each_query_boundary_and_keeps_live_integer_indices():
    case = _boundary_case(2)
    case["boundary"][1] = 8200  # 8200 is history for query 0, live for query 1.
    original, _ = reference(case)
    bitmap = membership_reference(case, original)
    assert bitmap["packed"][0, 0, :2].tolist() == [8192, 8200]
    assert bitmap["mapping"][0, 0, :2].tolist() == [1, 1]
    assert bitmap["topk"][0, 0, :4].tolist() == [0, 1, 16_777_219, -1]
    assert bitmap["topk"][1, 0, :3].tolist() == [8200, 16_777_221, -1]
    for q in range(2):
        assert Counter(bitmap["topk"][q, 0].tolist()) == Counter(original["topk"][q, 0].tolist())


@pytest.mark.parametrize("tokens,message", [([8192, -1, 8200], "prefix"), ([8192, 8192], "unique")])
def test_invalid_host_fixtures_are_rejected_before_timing(tokens, message):
    case = _boundary_case(0)
    _put_row(case, 0, tokens)
    expected, _ = reference(case)
    with pytest.raises(ValueError, match=message):
        membership_reference(case, expected)


@pytest.mark.parametrize("variant", ["exact_combined", "query_membership"])
def test_update_benchmark_prepares_the_matching_workspace(monkeypatch, variant):
    case = _boundary_case(3)
    calls = []

    def run(snapshot, name, stage):
        calls.append((name, stage))

    monkeypatch.setattr(Case, "run", run)
    monkeypatch.setattr(torch, "npu", type("NPU", (), {"synchronize": lambda: None}), raising=False)
    stage_snapshot(case, "cpu", "update", variant)
    assert calls == [(variant, "union"), (variant, "finalize")]


@pytest.mark.parametrize("history", [0, 1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 129])
def test_native_membership_adjacent_pairs_and_unaligned_rows(native, history):
    case = _boundary_case(history)
    # Add tiny segments in other shards and one shared token that is live in
    # the second row. This exercises partial mask words and the final append.
    case["topk"][0, 0, history + 1:history + 4] = torch.tensor([8193, 8194, 8195], dtype=torch.int32)
    case["boundary"][1] = 8192 if history % 2 else 16_777_216
    old, _ = reference(case)
    expected = membership_reference(case, old)
    actual = case.clone(native)
    actual.run("query_membership")
    torch.npu.synchronize()
    assert_result(actual, expected, membership=True)


@pytest.mark.parametrize("scenario", ["mixed_live", "full_shard"])
def test_native_membership_reuses_exact_miss_slots(native, scenario):
    if scenario == "mixed_live":
        case = make_case(3, 2, 4, 0.9, live_tokens=127, overlap=1024, block_size=127)
        # The final row is graph padding, not a valid row with an empty union.
        case["row_requests"][-1] = -1
    else:
        # The complete disjoint 4096-entry union falls in one shard.
        case = make_case(1, 2, 4, 0.9, scenario="skewed", overlap=0)
    old, _ = reference(case)
    expected = membership_reference(case, old)
    baseline = case.clone(native)
    actual = case.clone(native)
    baseline.run("exact_combined")
    actual.run("query_membership")
    torch.npu.synchronize()
    assert_result(baseline, old)
    assert_result(actual, expected, membership=True)
