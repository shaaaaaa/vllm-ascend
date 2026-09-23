from decimal import Decimal

import pytest
from profile_timing import kernel_names, parse_trace


def trace(variant, iterations=3):
    events = []
    for i in range(iterations):
        # A large host replay span is not kernel execution time.
        events.append({"ph": "X", "cat": "cpu_op", "name": "resident_experiment",
                       "ts": i * 1000, "dur": 120, "pid": 1, "tid": 1})
        for j, name in enumerate(kernel_names(variant).values()):
            events.append({"ph": "X", "cat": "kernel", "name": name + "_0",
                           "ts": i * 1000 + j * 15, "dur": 10, "pid": 2, "tid": 5})
    return {"traceEvents": events}


@pytest.mark.parametrize("variant", [
    "baseline", "optimized", "compact_remap", "sharded_finalize", "combined", "vector_union",
])
def test_device_durations_exclude_host_floor(variant):
    result = parse_trace(trace(variant), variant, "full", 3)
    assert result["kernel_sum"]["mean_us"] == 30
    assert result["chain_span"]["mean_us"] == 40
    assert all(v["mean_us"] == 10 for v in result["kernels"].values())


def test_missing_device_tasks_do_not_fall_back_to_host_spans():
    data = trace("baseline")
    for event in data["traceEvents"]:
        event["cat"] = "cpu_op"
    with pytest.raises(RuntimeError, match="refusing"):
        parse_trace(data, "baseline", "full", 3)


def test_hardware_lane_can_supply_kernel_name_as_argument():
    data = trace("baseline")
    data["traceEvents"].append({"ph": "M", "name": "process_name", "pid": 2,
                                "args": {"name": "Ascend Hardware"}})
    for event in data["traceEvents"]:
        if event.get("cat") == "kernel":
            event["args"] = {"Kernel Name": event["name"]}
            event["name"] = "resident_experiment"
            event["cat"] = ""
    assert parse_trace(data, "baseline", "full", 3)["kernel_sum"]["mean_us"] == 30


@pytest.mark.parametrize("fault", ["extra", "missing", "stream", "order"])
def test_trace_count_and_order_fail_closed(fault):
    data = trace("combined")
    if fault == "extra":
        data["traceEvents"].append(data["traceEvents"][-1].copy())
    elif fault == "missing":
        data["traceEvents"].pop()
    elif fault == "stream":
        data["traceEvents"][-1]["tid"] = 7
    else:
        data["traceEvents"][-1]["ts"] -= 16
    with pytest.raises(RuntimeError):
        parse_trace(data, "combined", "full", 3)


def test_large_timestamps_keep_submicrosecond_order_and_span():
    start = Decimal("1800000000000000.13")
    events = [
        {"ph": "X", "cat": "kernel", "name": name,
         "ts": str(start + Decimal("0.15") * i), "dur": "0.13", "pid": 2, "tid": 5}
        for i, name in enumerate(kernel_names("baseline").values())
    ]
    result = parse_trace(events, "baseline", "full", 1)
    assert result["kernel_sum"]["mean_us"] == pytest.approx(0.39)
    assert result["chain_span"]["mean_us"] == pytest.approx(0.43)
    # Preserve rejection of real overlap at the same absolute timestamp scale.
    events[1]["ts"] = str(start + Decimal("0.10"))
    with pytest.raises(RuntimeError, match="overlap"):
        parse_trace(events, "baseline", "full", 1)


@pytest.mark.parametrize("stage", ["union_sort", "union_dedup"])
@pytest.mark.parametrize("variant", ["baseline", "vector_union"])
def test_union_prefix_probe_has_its_own_device_symbol(stage, variant):
    name = kernel_names(variant)["union"] + "_" + stage.removeprefix("union_")
    events = [{"ph": "X", "cat": "kernel", "name": name + "_0",
               "ts": 100, "dur": 40, "pid": 2, "tid": 5}]
    assert parse_trace(events, variant, stage, 1)["kernels"][stage]["mean_us"] == 40


def test_intersection_variant_changes_only_union_symbol():
    original = kernel_names('baseline')
    optimized = kernel_names('vector_intersection')
    assert optimized['union'] == 'dsa_resident_sharded_union_kernel_intersection'
    assert optimized['finalize'] == original['finalize']
    assert optimized['update'] == original['update']
