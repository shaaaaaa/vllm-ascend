"""Run old/new compiled kernels against an independent CPU set oracle."""

import itertools

import pytest
import torch
from resident_experiment import assert_result, make_case, reference


@pytest.mark.parametrize("mtp,shards", itertools.product((1, 2), (1, 2, 4)))
@pytest.mark.parametrize(
    "scenario",
    (
        "normal",
        "cold",
        "generation",
        "zero_boundary",
        "subset",
        "padding",
        "inactive",
        "invalid_indices",
        "one_shard_miss",
        "skewed",
    ),
)
def test_old_and_new_match_oracle(native, mtp, shards, scenario):
    initial = make_case(2, mtp, shards, 0.9, scenario)
    expected, _ = reference(initial)
    for optimized in (False, True):
        device_case = initial.clone(native)
        device_case.run(optimized)
        torch.npu.synchronize()
        assert_result(device_case, expected)


@pytest.mark.parametrize("mtp,shards", [(1, 1), (1, 4), (2, 1), (2, 4)])
def test_all_hit_full_set_uses_fast_paths(native, mtp, shards):
    initial = make_case(2, mtp, shards, 1.0)
    expected, stats = reference(initial)
    assert stats["misses"] == 0
    assert stats["unchanged_shards"] == initial.requests * initial.shards
    for optimized in (False, True):
        case = initial.clone(native)
        case.run(optimized)
        torch.npu.synchronize()
        assert_result(case, expected)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("mtp,shards", [(1, 4), (2, 1), (2, 4)])
def test_graph_replay_changes_generation_and_padding_without_host_fences(native, optimized, mtp, shards):
    initial = make_case(2, mtp, shards, 0.9)
    case = initial.clone(native)
    for _ in range(3):
        case.reset_from(initial.clone(native))
        case.run(optimized)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        case.run(optimized)
    case.reset_from(initial.clone(native))
    cpu = initial.clone()
    pending = []
    # All launches/copies/snapshots queue before a single final host fence.
    for step in range(6):
        cpu["topk"].copy_(initial["topk"] + (step % 3) * 113)
        if step == 2:
            cpu["request_generations"].add_(1)
        if step == 3:
            cpu["row_requests"][-1] = -1
        if step == 4:
            cpu["row_requests"].copy_(initial["row_requests"])
            cpu["request_states"].fill_(-1)
        for name in ("topk", "request_generations", "row_requests", "request_states"):
            case[name].copy_(cpu[name].to(native))
        expected, _ = reference(cpu)
        graph.replay()
        pending.append((case.clone(), expected))
        cpu = expected
    torch.npu.synchronize()
    for actual, expected in pending:
        assert_result(actual, expected)


@pytest.mark.parametrize("requests,block_size", [(17, 16), (33, 128), (2, 127)])
def test_grid_stride_and_fragmented_physical_blocks(native, requests, block_size):
    initial = make_case(requests, 2, 4, 0.97, block_size=block_size)
    expected, _ = reference(initial)
    for optimized in (False, True):
        case = initial.clone(native)
        case.run(optimized)
        torch.npu.synchronize()
        assert_result(case, expected)


@pytest.mark.parametrize("fault", ["dtype", "shape", "alias", "alignment", "block_size", "stage"])
def test_invalid_launch_rejected_before_kernel(native, fault):
    case = make_case().clone(native)
    stage = 0
    if fault == "dtype":
        case.tensors[4] = case.tensors[4].to(torch.int32)
    elif fault == "shape":
        case.tensors[17] = case.tensors[17][:, :1].contiguous()
    elif fault == "alias":
        case.tensors[12] = case.tensors[4]
    elif fault == "alignment":
        case.tensors[17] = torch.empty(case["miss_counts"].numel() + 1, device=native, dtype=torch.int32)[
            1:
        ].reshape_as(case["miss_counts"])
    elif fault == "block_size":
        case.block_size = 0
    else:
        stage = 5
    with pytest.raises(RuntimeError):
        torch.ops.resident_experiment.run_(case.tensors, case.dummy_base, case.block_size, True, stage)
