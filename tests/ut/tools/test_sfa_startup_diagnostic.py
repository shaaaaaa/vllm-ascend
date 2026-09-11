# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests of diagnostics, not proof that HCCL startup succeeds."""

import ast
import builtins
import importlib
import importlib.util
import math
import multiprocessing
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    return importlib.import_module("sfa_startup_diagnostic"), importlib.import_module("sfa_startup_trace")


def options(tmp_path, **kwargs):
    return SimpleNamespace(
        model=str(tmp_path),
        devices="0,1,2,3,4,5,6,7",
        trace_dir=str(tmp_path),
        load_model=False,
        timeout=300,
        skip_preflight=False,
        repo_root=None,
        **kwargs,
    )


def fake_torch():
    npu = SimpleNamespace(initialized=False, current=None, selections=[])
    npu.is_initialized = lambda: npu.initialized

    def current_device():
        assert npu.initialized, "Tracing must not initialize a default device"
        return npu.current

    def set_device(device):
        npu.selections.append(device)
        npu.current, npu.initialized = device, True

    npu.current_device = current_device
    npu.set_device = set_device
    groups = []

    def new_group(ranks, backend=None, pg_options=None):
        result = object()
        groups.append((ranks, backend, result))
        return result

    def init_process_group(backend, rank, world_size, init_method=None):
        return None

    return SimpleNamespace(
        npu=npu, distributed=SimpleNamespace(new_group=new_group, init_process_group=init_process_group), groups=groups
    )


def test_options_preserve_exact_parity_configuration(modules, tmp_path):
    driver, _ = modules
    args = options(tmp_path)
    actual = driver.diagnostic_options(args)
    baseline = driver.engine_options(
        SimpleNamespace(
            child="eager",
            devices=args.devices,
            model=args.model,
            reference=args.trace_dir,
            atol=1e-7,
            rtol=1e-2,
            compare_output=True,
            trace_residual=False,
        )
    )
    baseline["worker_cls"] = actual["worker_cls"]
    baseline["additional_config"]["sfa_startup"] = {"trace_dir": args.trace_dir, "load_model": False}
    assert actual == baseline
    assert actual["tensor_parallel_size"] == 8 and actual["data_parallel_size"] == 1
    assert actual["enforce_eager"] and not actual["enable_expert_parallel"]


def test_options_are_accepted_fields_of_real_sibling_engine_args(modules, tmp_path):
    source = TOOLS.parents[1] / "vllm/vllm/engine/arg_utils.py"
    if not source.is_file():
        pytest.skip("Sibling vllm checkout is absent")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    engine_args = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EngineArgs")
    fields = {node.target.id for node in engine_args.body if isinstance(node, ast.AnnAssign)}
    assert modules[0].diagnostic_options(options(tmp_path)).keys() <= fields


def test_trace_does_not_touch_device_before_original_set_device(modules, tmp_path):
    driver, trace_module = modules
    torch = fake_torch()
    originals = torch.npu.set_device, torch.distributed.new_group
    trace = trace_module.StartupTrace(tmp_path, rank=3, local_rank=3)
    previous = sys.getprofile()
    try:
        trace.install(torch)
        assert not torch.npu.initialized
        torch.npu.set_device(3)
        result = torch.distributed.new_group([0, 1, 2, 3], backend="hccl")
        assert torch.groups == [([0, 1, 2, 3], "hccl", result)]
        assert torch.npu.selections == [3]
    finally:
        trace.close()
    assert originals == (torch.npu.set_device, torch.distributed.new_group)
    assert sys.getprofile() == previous
    records = driver.read_records(tmp_path)
    assert records[0]["device"] is None
    assert any(r.get("operation") == "set_device" and r["event"] == "end" and r["device"] == 3 for r in records)


def test_existing_native_call_observed_once_and_native_exception_preserved(modules, tmp_path, monkeypatch):
    driver, trace_module = modules
    # math.sqrt is a real C call: exercise the profiler, not a Python mock of
    # HCCL. The production filter watches only get_hccl_comm_name.
    monkeypatch.setattr(trace_module, "COMM_NAME_METHOD", "sqrt")
    trace = trace_module.StartupTrace(tmp_path)
    try:
        trace.install(fake_torch())
        assert math.sqrt(9) == 3
        with pytest.raises(ValueError, match="math domain"):
            math.sqrt(-1)
    finally:
        trace.close()
    calls = [r for r in driver.read_records(tmp_path) if r["event"] == "hccl_comm_name"]
    assert [r["action"] for r in calls] == ["c_call", "c_return", "c_call", "c_exception"]
    assert trace.comm_calls == 2
    assert calls[0]["stack"]


def test_failed_group_call_is_not_retried_or_swallowed(modules, tmp_path):
    driver, trace_module = modules
    torch = fake_torch()
    error = RuntimeError("native bind failed")

    def broken(ranks, backend):
        raise error

    torch.distributed.new_group = broken
    trace = trace_module.StartupTrace(tmp_path)
    try:
        trace.install(torch)
        with pytest.raises(RuntimeError) as caught:
            torch.distributed.new_group([0, 1], backend="hccl")
    finally:
        trace.close()
    assert caught.value is error
    records = driver.read_records(tmp_path)
    assert len([r for r in records if r.get("operation") == "new_group" and r["event"] == "begin"]) == 1
    assert any(r["event"] == "error" and "native bind failed" in r["error"] for r in records)


def test_coordinator_name_and_members_reach_group_trace(modules, tmp_path):
    driver, trace_module = modules
    torch = fake_torch()

    class Coordinator:
        def __init__(self, group_ranks, group_name=None):
            self.group = torch.distributed.new_group(group_ranks[0], backend="hccl")

    original = Coordinator.__init__
    trace = trace_module.StartupTrace(tmp_path)
    try:
        trace.install(torch)
        trace.trace_coordinator(Coordinator)
        Coordinator([[0, 1]], group_name="mc2")
    finally:
        trace.close()
    assert Coordinator.__init__ is original
    records = driver.read_records(tmp_path)
    assert any(
        r["event"] == "group_created" and r["group_name"] == "mc2" and r["arguments"]["ranks"] == [0, 1]
        for r in records
    )


@pytest.mark.parametrize(
    "field,value", [("pid", 10), ("uuid", "uuid-0"), ("visible_device", "0"), ("rank", 0), ("device", 0)]
)
def test_duplicate_or_wrong_rank_mapping_fails(modules, field, value):
    _, trace_module = modules
    records = [
        dict(
            event="device_mapping", rank=i, local_rank=i, device=i, pid=10 + i, uuid=f"uuid-{i}", visible_device=str(i)
        )
        for i in range(2)
    ]
    assert trace_module.mapping_errors(records, 2, require_complete=True) == []
    records[1][field] = value
    assert trace_module.mapping_errors(records, 2, require_complete=True)


def test_unknown_uuid_is_not_reported_as_duplicate_device(modules):
    _, trace_module = modules
    records = [
        dict(event="device_mapping", rank=i, local_rank=i, device=i, pid=10 + i, uuid=None, visible_device=str(i))
        for i in range(8)
    ]
    assert trace_module.mapping_errors(records, 8, require_complete=True) == []
    assert trace_module.mapping_errors(records[:1], 8, require_complete=True)


def test_native_logs_only_include_current_run_exact_pids(modules, tmp_path, capsys):
    driver, _ = modules
    for pid in (12, 123, 13):
        (tmp_path / f"plog-{pid}_now.log").write_text(f"socket bind: port 16666 pid {pid}\nnot relevant\n")
    driver.collect_native_bind_logs({12}, 0, [tmp_path])
    output = capsys.readouterr().out
    assert "plog-12_now" in output
    assert "plog-123_" not in output and "plog-13_" not in output
    assert "not relevant" not in output


def test_failure_still_collects_native_logs_without_claiming_pass(modules, tmp_path, monkeypatch, capsys):
    driver, _ = modules
    monkeypatch.setattr(driver.sys, "platform", "linux")
    (tmp_path / "config.json").write_text("{}")
    failure = subprocess.CalledProcessError(1, ["startup"])
    run = Mock(side_effect=[None, failure])
    collect = Mock()
    monkeypatch.setattr(driver, "run_stage", run)
    monkeypatch.setattr(driver, "collect_native_bind_logs", collect)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        driver.run_diagnostic(options(tmp_path))
    assert caught.value is failure
    assert run.call_count == 2 and collect.call_count == 1
    assert "STARTUP PASS" not in capsys.readouterr().out


def test_skip_preflight_starts_only_one_eager_child_and_preserves_ports(modules, tmp_path, monkeypatch):
    driver, _ = modules
    monkeypatch.setattr(driver.sys, "platform", "linux")
    monkeypatch.setenv("HCCL_NPU_SOCKET_PORT_RANGE", "16666-16680")
    (tmp_path / "config.json").write_text("{}")
    run = Mock(side_effect=subprocess.CalledProcessError(1, ["startup"]))
    monkeypatch.setattr(driver, "run_stage", run)
    monkeypatch.setattr(driver, "collect_native_bind_logs", Mock())
    args = options(tmp_path)
    args.skip_preflight = True
    with pytest.raises(subprocess.CalledProcessError):
        driver.run_diagnostic(args)
    assert run.call_count == 1
    argv, env, _ = run.call_args.args
    assert argv[argv.index("--child") + 1] == "startup"
    assert env["VLLM_ASCEND_SFA_FULL_GRAPH"] == "0"
    assert env["HCCL_NPU_SOCKET_PORT_RANGE"] == "16666-16680"


def test_executor_stops_before_cache_initialization_or_inference(modules, tmp_path, monkeypatch):
    driver, trace_module = modules
    executor = Mock()
    executor.collective_rpc.return_value = [dict(rank=i, complete=True) for i in range(8)]
    config = object()
    engine_args = Mock()
    engine_args.return_value.create_engine_config.return_value = config
    constructor = Mock(return_value=executor)
    stubs = {
        "vllm.config": dict(set_current_vllm_config=lambda c: nullcontext()),
        "vllm.engine.arg_utils": dict(EngineArgs=engine_args),
        "vllm.v1.executor.abstract": dict(Executor=SimpleNamespace(get_class=Mock(return_value=constructor))),
    }
    for name, attributes in stubs.items():
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
    driver.run_executor(options(tmp_path), trace_module.StartupTrace(tmp_path))
    executor.collective_rpc.assert_called_once_with("startup_report", timeout=300)
    executor.shutdown.assert_called_once()
    executor.initialize_from_config.assert_not_called()
    executor.determine_available_memory.assert_not_called()
    executor.execute_model.assert_not_called()


def test_requested_checkout_mismatch_is_an_error(modules, tmp_path, monkeypatch):
    driver, trace_module = modules
    monkeypatch.setattr(
        driver.importlib.util, "find_spec", lambda name: SimpleNamespace(origin="/wrong/vllm/__init__.py")
    )
    with pytest.raises(RuntimeError, match="not being used"):
        driver.report_versions(trace_module.StartupTrace(tmp_path), root=tmp_path)


def test_child_does_not_import_torch_ahead_of_original_preflight(modules, tmp_path, monkeypatch):
    driver, _ = modules
    preflight = Mock()
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        assert name not in ("torch", "torch_npu"), "diagnostic changed the original import order"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib.import_module("sfa_full_graph_parity"), "preflight_dependencies", preflight)
    monkeypatch.setattr(driver, "report_versions", Mock())
    monkeypatch.setattr(builtins, "__import__", checked_import)
    driver.run_child(options(tmp_path, child="preflight"))
    preflight.assert_called_once_with()


def test_timeout_only_terminates_owned_new_session(modules, monkeypatch):
    driver, _ = modules
    process = Mock(pid=12345)
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("startup", 1), 0]
    context = Mock()
    context.__enter__ = Mock(return_value=process)
    context.__exit__ = Mock(return_value=False)
    launch = Mock(return_value=context)
    terminate = Mock()
    monkeypatch.setattr(driver.subprocess, "Popen", launch)
    monkeypatch.setattr(driver.os, "killpg", terminate, raising=False)
    with pytest.raises(subprocess.TimeoutExpired):
        driver.run_stage(["python", "diagnostic"], {}, 1)
    assert launch.call_args.kwargs["start_new_session"] is True
    terminate.assert_called_once_with(12345, driver.signal.SIGTERM)


def test_help_is_available_without_loading_torch_or_npu(modules):
    result = subprocess.run(
        [sys.executable, str(TOOLS / "sfa_startup_diagnostic.py"), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--load-model" in result.stdout and "--repo-root" in result.stdout


@pytest.mark.parametrize(
    "load_model,comm_result", [(False, "ok"), (True, "ok"), (False, "empty"), (False, "unobserved")]
)
def test_actual_diagnostic_worker_selects_only_requested_path(modules, tmp_path, monkeypatch, load_model, comm_result):
    _, trace_module = modules
    monkeypatch.setattr(trace_module, "COMM_NAME_METHOD", "sqrt")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5")
    torch = fake_torch()
    torch.npu.get_device_properties = lambda device: SimpleNamespace(uuid=f"uuid-{device}")
    calls = []

    class BaseWorker:
        def __init__(self, vllm_config, local_rank, rank, **kwargs):
            self.vllm_config, self.local_rank, self.rank = vllm_config, local_rank, rank
            calls.append("constructor")

        def init_device(self):
            torch.npu.set_device(self.local_rank)
            calls.append("init_device")

        def load_model(self):
            calls.append("full_load_model")
            math.sqrt(1)

        def _prepare_quant_config(self):
            calls.append("quant_remap")

    class Scheme:
        def __init__(self):
            calls.append("w4a8_scheme")
            if comm_result != "unobserved":
                math.sqrt(1)
            self.moe_all_to_all_group_name = "native-name" if comm_result != "empty" else ""

    class Coordinator:
        def __init__(self, group_ranks):
            pass

    stubs = {
        "vllm.config": dict(set_current_vllm_config=lambda c: nullcontext()),
        "vllm_ascend.worker.sfa_parity_worker": dict(SFAParityWorker=BaseWorker),
        "vllm.distributed.parallel_state": dict(GroupCoordinator=Coordinator),
        "vllm_ascend.quantization.methods.w4a8": dict(AscendW4A8DynamicFusedMoEMethod=Scheme),
    }
    for name, attributes in stubs.items():
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setitem(sys.modules, "torch", torch)
    spec = importlib.util.spec_from_file_location("tested_startup_worker", TOOLS / "sfa_startup_worker.py")
    worker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker_module)
    config = SimpleNamespace(additional_config={"sfa_startup": dict(trace_dir=str(tmp_path), load_model=load_model)})
    worker = worker_module.SFAStartupWorker(config, local_rank=1, rank=1, distributed_init_method="tcp://test")
    try:
        worker.init_device()
        if comm_result == "ok":
            worker.load_model()
            assert worker.startup_report()["complete"]
        else:
            with pytest.raises(RuntimeError, match="skipped HCCL|coverage is incomplete"):
                worker.load_model()
    finally:
        worker.startup_trace.close()
    assert calls == ["constructor", "init_device"] + (
        ["full_load_model"] if load_model else ["quant_remap", "w4a8_scheme"]
    )
    records = modules[0].read_records(tmp_path)
    mapping = next(r for r in records if r["event"] == "device_mapping")
    assert mapping["visible_device"] == "5" and mapping["device"] == 1
    assert worker.startup_trace.comm_calls == int(comm_result != "unobserved")
    assert any(r["event"] == "startup_complete" for r in records) == (comm_result == "ok")


def gloo_trace_worker(rank, world_size, directory, rendezvous, results):
    import torch
    import torch.distributed as dist

    sys.path.insert(0, str(TOOLS))
    from sfa_startup_trace import StartupTrace

    trace = StartupTrace(directory, rank=rank, local_rank=rank)
    try:
        # Real CPU collectives; only the device-observation interface is a
        # fixture. This does not simulate a passing NPU/HCCL initialization.
        trace.install(SimpleNamespace(npu=fake_torch().npu, distributed=dist))
        dist.init_process_group(
            "gloo", rank=rank, world_size=world_size, init_method=rendezvous, timeout=timedelta(seconds=30)
        )
        group = dist.new_group(list(range(world_size)), backend="gloo")
        tensor = torch.tensor([rank + 1])
        dist.all_reduce(tensor, group=group)
        results.put((rank, tensor.item()))
    finally:
        trace.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_tracing_real_two_process_gloo_preserves_collective(modules, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(
            target=gloo_trace_worker, args=(rank, 2, str(tmp_path), (tmp_path / "rendezvous").as_uri(), results)
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 45
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        assert [process.exitcode for process in processes] == [0, 0]
        assert dict(results.get(timeout=2) for _ in processes) == {0: 3, 1: 3}
        records = modules[0].read_records(tmp_path)
        assert len([r for r in records if r["event"] == "group_created"]) == 2
        assert len([r for r in records if r.get("operation") == "init_process_group" and r["event"] == "end"]) == 2
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=5)
        results.close()
        results.join_thread()
