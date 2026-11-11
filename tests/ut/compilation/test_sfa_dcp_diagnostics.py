# SPDX-License-Identifier: Apache-2.0
"""CPU checks for test-only DCP stage observation and failure reporting."""

import ast
import os
import subprocess
import sys
from datetime import timedelta
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sfa_test_support import ROOT, extract

SOURCE = ROOT / "tests/e2e/multicard/test_sfa_dcp.py"


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("shape", ["probe", "attention"])
@pytest.mark.parametrize("async_op", [False, True])
def test_variable_exchange_preserves_payload_and_work(monkeypatch, world, shape, async_op):
    dims = (world * 64,) if shape == "probe" else (world, 2, 64 // world, 528)
    send, recv = torch.randn(dims), torch.empty(dims)
    original = send.clone()
    work, group = object(), object()

    def exchange(output, input, output_splits, input_splits, *, group, async_op):
        assert input is send and output is recv
        # HCCL multiplies splits by the row extent. The alternate API must
        # address exactly the same contiguous per-peer segments as fixed A2A.
        row_extent = input.numel() // input.shape[0]
        assert [size * row_extent for size in input_splits] == [input.numel() // world] * world
        assert output_splits == input_splits
        return work

    exchange = Mock(side_effect=exchange)
    dist = SimpleNamespace(all_to_all_single=exchange, get_world_size=Mock(return_value=world))
    install = extract(SOURCE, "_use_variable_dcp_exchange", {"dist": dist})
    with monkeypatch.context() as patch:
        install(patch)
        assert dist.all_to_all_single(recv, send, group=group, async_op=async_op) is work
        assert exchange.call_args.kwargs == {"group": group, "async_op": async_op}
        torch.testing.assert_close(send, original, rtol=0, atol=0)
    assert dist.all_to_all_single is exchange


def test_variable_exchange_preserves_explicit_unequal_splits(monkeypatch):
    exchange = Mock()
    dist = SimpleNamespace(all_to_all_single=exchange, get_world_size=lambda group: 2)
    install = extract(SOURCE, "_use_variable_dcp_exchange", {"dist": dist})
    install(monkeypatch)
    send, recv = torch.empty(3, 5), torch.empty(7, 5)
    outgoing, incoming = [1, 2], [4, 3]
    dist.all_to_all_single(recv, send, incoming, outgoing)
    exchange.assert_called_once_with(recv, send, incoming, outgoing, group=None, async_op=False)


@pytest.mark.parametrize("mode", ["group-inherit", "group-host"])
def test_group_override_is_local_and_preserves_production_buffer(monkeypatch, mode):
    # Simulate the binding's copy-on-read getter: mutating that copy would
    # silently discard the override on the installed Torch-NPU revision.
    class Options:
        def __init__(self):
            self._config = {}

        @property
        def hccl_config(self):
            return dict(self._config)

        @hccl_config.setter
        def hccl_config(self, value):
            self._config = dict(value)

    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")
    options = Options()
    backend = SimpleNamespace(ProcessGroupHCCL=SimpleNamespace(Options=lambda: options))
    npu = SimpleNamespace(_C=SimpleNamespace(_distributed_c10d=backend))
    group = object()
    dist = SimpleNamespace(new_group=Mock(return_value=group))
    create = extract(SOURCE, "_dcp_test_group", {"torch_npu": npu, "dist": dist, "timedelta": timedelta})
    assert create(mode) is group
    assert options.hccl_config == ({"hccl_buffer_size": 200, "hccl_op_expansion_mode": 1}
                                   if mode == "group-host" else {"hccl_buffer_size": 200})
    dist.new_group.assert_called_once_with(backend="hccl", pg_options=options, timeout=timedelta(seconds=120))
    assert os.environ["HCCL_OP_EXPANSION_MODE"] == "AIV"


@pytest.mark.parametrize("mode", ["fixed", "alltoallv", "group-inherit", "group-host"])
@pytest.mark.parametrize("failed", [False, True])
def test_transport_fixture_releases_only_its_group(mode, failed):
    world, separate = object(), object()
    dist = SimpleNamespace(group=SimpleNamespace(WORLD=world), destroy_process_group=Mock(), get_rank=lambda: 0)
    install, create = Mock(), Mock(return_value=separate)
    fixture = extract(SOURCE, "dcp_transport", {
        "dist": dist, "os": os, "_use_variable_dcp_exchange": install, "_dcp_test_group": create,
    })
    patch = object()
    session = fixture(SimpleNamespace(param=mode), patch, 8)
    owns_group = mode.startswith("group-")
    assert next(session) is (separate if owns_group else world)
    assert install.call_count == int(mode == "alltoallv")
    assert create.call_count == int(owns_group)
    if failed:
        with pytest.raises(RuntimeError, match="parity failed"):
            session.throw(RuntimeError("parity failed"))
    else:
        with pytest.raises(StopIteration):
            next(session)
    if owns_group:
        create.assert_called_once_with(mode)
        dist.destroy_process_group.assert_called_once_with(separate)
    else:
        dist.destroy_process_group.assert_not_called()


def test_pytest_reuses_process_group_after_a_case_failure(tmp_path):
    # Exercise pytest's real fixture scope/teardown with the production test
    # fixture. This validates harness lifecycle, not native HCCL behavior.
    fixture = next(node for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body
                   if getattr(node, "name", "") == "dcp_world")
    script = dedent('''
        import os
        from datetime import timedelta
        from types import SimpleNamespace as NS
        import pytest
        os.environ.update(WORLD_SIZE="8", LOCAL_RANK="0")
        events = []
        def init(backend, timeout):
            assert events == ["device"]
            assert backend == "hccl" and timeout == timedelta(seconds=120)
            events.append("init")
        def destroy():
            assert events == ["device", "init", 0, 1, 2]
            events.append("destroy")
        torch = NS(npu=NS(set_device=lambda device: events.append("device")))
        dist = NS(init_process_group=init, destroy_process_group=destroy)
    ''') + ast.unparse(fixture) + '\n' + dedent('''
        @pytest.mark.parametrize("case", [0, 1, 2])
        def test_case(case, dcp_world):
            assert dcp_world == 8 and events.count("init") == 1
            assert "destroy" not in events
            events.append(case)
            if case == 1:
                pytest.xfail("simulated rank-local numerical failure")
    ''')
    path = tmp_path / "test_group_lifetime.py"
    path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "--noconftest", "-o", "addopts=",
         "-p", "no:cacheprovider", "-q", str(path)],
        cwd=tmp_path, env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed, 1 xfailed" in result.stdout


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
