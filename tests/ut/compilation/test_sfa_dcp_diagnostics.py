# SPDX-License-Identifier: Apache-2.0
"""CPU checks for test-only DCP stage observation and failure reporting."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sfa_test_support import ROOT, extract

SOURCE = ROOT / "tests/e2e/multicard/test_sfa_dcp.py"


def test_stage_observer_preserves_calls_and_original_tensor_identity(monkeypatch):
    partial, maximum, total = torch.ones(2, 8, 512), torch.ones(2, 8), torch.ones(2, 8)
    send, recv = torch.ones(2, 2, 4, 528), torch.zeros(2, 2, 4, 528)
    group, result, work = object(), object(), object()
    native_exchange = Mock(return_value=work)
    dist = SimpleNamespace(all_to_all_single=native_exchange)

    def merge(*args):
        assert args == (partial, maximum, total, group)
        assert dist.all_to_all_single(recv, send, group=group) is work
        return result

    original_merge = Mock(side_effect=merge)
    module = SimpleNamespace(merge_sfa_dcp=original_merge)
    retain = extract(SOURCE, "_retain_dcp_stages", {"dist": dist})
    with monkeypatch.context() as patch:
        stages = retain(patch, module)
        assert module.merge_sfa_dcp(partial, maximum, total, group) is result
        for name, tensor in (("partial", partial), ("maximum", maximum), ("total", total),
                             ("send", send), ("recv", recv)):
            assert stages[name] is tensor
        native_exchange.assert_called_once_with(recv, send, group=group)
        original_merge.assert_called_once_with(partial, maximum, total, group)
    assert module.merge_sfa_dcp is original_merge
    assert dist.all_to_all_single is native_exchange


@pytest.mark.parametrize("broken_stage", [None, "partial", "maximum", "total", "send", "recv"])
def test_empty_stage_checks_localize_stale_native_or_exchange_data(broken_stage):
    stages = {
        "partial": torch.zeros(2, 8, 512, dtype=torch.bfloat16),
        "maximum": torch.zeros(2, 8), "total": torch.zeros(2, 8),
        "send": torch.zeros(2, 2, 4, 528, dtype=torch.bfloat16),
        "recv": torch.zeros(2, 2, 4, 528, dtype=torch.bfloat16),
    }
    for name in ("send", "recv"):
        stages[name][..., 512:516] = torch.tensor([0, 0, 128, 255], dtype=torch.bfloat16)
    if broken_stage:
        stages[broken_stage].view(-1)[0] = 1
    compare = extract(SOURCE, "_empty_dcp_stage_comparisons", {"torch": torch})
    check = extract(SOURCE, "_assert_dcp_comparisons", {"torch": torch})
    comparisons = compare(stages, "replay=1")
    if broken_stage:
        with pytest.raises(AssertionError, match=f"replay=1 kind={broken_stage}") as error:
            check(comparisons)
        assert str(error.value).count("DCP parity") == 1
    else:
        check(comparisons)


def test_report_contains_all_replays_and_kinds_after_a_failure():
    check = extract(SOURCE, "_assert_dcp_comparisons", {"torch": torch})
    with pytest.raises(AssertionError) as error:
        check([
            ("replay=0 kind=attention", torch.zeros(1), torch.zeros(1)),
            ("replay=1 kind=attention", torch.ones(1), torch.zeros(1)),
            ("replay=1 kind=prefetch", torch.ones(1), torch.zeros(1)),
            ("replay=2 kind=attention", torch.ones(1), torch.zeros(1)),
        ])
    message = str(error.value)
    assert "replay=0" not in message
    assert all(label in message for label in (
        "replay=1 kind=attention", "replay=1 kind=prefetch", "replay=2 kind=attention",
    ))
