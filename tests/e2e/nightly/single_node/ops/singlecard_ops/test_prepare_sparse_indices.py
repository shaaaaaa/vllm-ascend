import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
    _prepare_sparse_indices_reuse_torch,
    _prepare_sparse_indices_torch,
    prepare_sparse_indices,
)
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module", autouse=True)
def _load_dsa_union_operator():
    if not enable_custom_op():
        pytest.fail("vllm-ascend custom operators could not be loaded")
    if not hasattr(torch.ops._C_ascend, "npu_dsa_prepare_sparse_indices_"):
        pytest.fail("vllm_ascend_C does not contain the DSA union operator")


def _buffers(requests: int, capacity: int):
    return (
        torch.empty((requests, capacity), dtype=torch.int32, device="npu"),
        torch.empty((requests, 16), dtype=torch.int32, device="npu"),
        torch.empty((requests, capacity), dtype=torch.long, device="npu"),
    )


def _aligned(values, width=16):
    result = torch.zeros((len(values), width), dtype=torch.int32)
    for row, entries in enumerate(values):
        result[row, : len(entries)] = torch.tensor(entries, dtype=torch.int32)
    return result


def _reuse_topk(values, width=16, live_position=30):
    result = torch.full(
        (len(values), width), live_position, dtype=torch.int32
    )
    for row, entries in enumerate(values):
        result[row, : len(entries)] = torch.tensor(entries, dtype=torch.int32)
    return result


def _require_reuse_operator():
    if not hasattr(
        torch.ops._C_ascend, "npu_dsa_prepare_sparse_indices_reuse_"
    ):
        pytest.fail(
            "vllm_ascend_C does not contain the DSA scratch-reuse operator; "
            "rebuild the custom-op extension"
        )


def _run_reuse_step(
    topk_cpu,
    boundaries_cpu,
    row_req_cpu,
    table_cpu,
    state_indices_cpu,
    generations_cpu,
    resident_cpu,
    resident_generations_cpu,
    resident_npu,
    resident_generations_npu,
    block_size=2,
    clear_invalid_rows=False,
):
    expected = _prepare_sparse_indices_reuse_torch(
        topk_cpu,
        boundaries_cpu,
        row_req_cpu,
        table_cpu,
        state_indices_cpu,
        generations_cpu,
        resident_cpu,
        resident_generations_cpu,
        block_size,
        clear_invalid_rows=clear_invalid_rows,
    )
    buffers = _buffers(table_cpu.shape[0], resident_cpu.shape[1])
    actual = prepare_sparse_indices(
        topk_cpu.npu(),
        boundaries_cpu.npu(),
        row_req_indices=row_req_cpu.npu(),
        request_block_table=table_cpu.npu(),
        selected_packed=buffers[0],
        selected_counts=buffers[1],
        target_slot_mapping=buffers[2],
        block_size=block_size,
        request_state_indices=state_indices_cpu.npu(),
        request_generations=generations_cpu.npu(),
        resident_token_ids=resident_npu,
        resident_generations=resident_generations_npu,
        clear_invalid_rows=clear_invalid_rows,
    )
    torch.npu.synchronize()
    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[2].cpu(), expected[2])
    for request, count in enumerate(expected[2].tolist()):
        assert torch.equal(
            actual[1][request, :count].cpu(),
            expected[1][request, :count],
        )
        assert torch.equal(
            actual[3][request, :count].cpu(),
            expected[3][request, :count],
        )
    assert torch.equal(resident_npu.cpu(), resident_cpu)
    assert torch.equal(
        resident_generations_npu.cpu(), resident_generations_cpu
    )
    return expected


def test_mtp_rows_build_one_sorted_union_per_request():
    topk = _aligned([[3, 1, 8], [2, 3, 9], [4, 1, 10]])
    row_requests = torch.tensor([0, 0, 1], dtype=torch.int32)
    boundaries = torch.tensor([5, 5, 5], dtype=torch.int32)
    tables = _aligned(
        [[20, 21, 22, 23, 24, 25], [30, 31, 32, 33, 34, 35]]
    )
    expected = _prepare_sparse_indices_torch(
        topk,
        boundaries,
        row_req_indices=row_requests,
        request_block_table=tables,
        block_size=2,
    )
    buffers = _buffers(2, 32)
    actual = prepare_sparse_indices(
        topk.npu(),
        boundaries.npu(),
        row_req_indices=row_requests.npu(),
        request_block_table=tables.npu(),
        selected_packed=buffers[0],
        selected_counts=buffers[1],
        target_slot_mapping=buffers[2],
        block_size=2,
    )
    torch.npu.synchronize()

    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[2].cpu(), expected[2])
    for request, count in enumerate(expected[2].tolist()):
        assert torch.equal(
            actual[1][request, :count].cpu(),
            expected[1][request, :count],
        )
        assert torch.equal(
            actual[3][request, :count].cpu(),
            expected[3][request, :count],
        )


def test_q1_requests_remain_independent():
    topk = _aligned([[1, 3, 8], [1, 4, 9]])
    row_requests = torch.tensor([0, 1], dtype=torch.int32)
    boundaries = torch.tensor([5, 5], dtype=torch.int32)
    tables = _aligned([[20, 21, 22], [30, 31, 32]])
    expected = _prepare_sparse_indices_torch(
        topk,
        boundaries,
        row_req_indices=row_requests,
        request_block_table=tables,
        block_size=2,
    )
    buffers = _buffers(2, 16)
    actual = prepare_sparse_indices(
        topk.npu(),
        boundaries.npu(),
        row_req_indices=row_requests.npu(),
        request_block_table=tables.npu(),
        selected_packed=buffers[0],
        selected_counts=buffers[1],
        target_slot_mapping=buffers[2],
        block_size=2,
    )
    torch.npu.synchronize()
    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[2].cpu(), expected[2])


def test_zero_boundary_keeps_resident_absolute_indices():
    topk = _aligned([[1, 3, 8], [2, 4, 9]])
    original = topk.clone()
    row_requests = torch.tensor([0, 0], dtype=torch.int32)
    tables = _aligned([[20, 21, 22]])
    buffers = _buffers(1, 32)
    actual = prepare_sparse_indices(
        topk.npu(),
        torch.zeros(2, dtype=torch.int32, device="npu"),
        row_req_indices=row_requests.npu(),
        request_block_table=tables.npu(),
        selected_packed=buffers[0],
        selected_counts=buffers[1],
        target_slot_mapping=buffers[2],
        block_size=2,
    )
    torch.npu.synchronize()
    assert torch.equal(actual[0].cpu(), original)
    assert actual[2].cpu().tolist() == [0]


def test_reuse_operator_two_steps_mtp_zero_miss_and_eviction():
    _require_reuse_operator()
    table = _aligned([[20, 21, 22, 23, 24, 25, 26, 27]])
    row_req = torch.tensor([0, 0], dtype=torch.int32)
    state_indices = torch.tensor([0], dtype=torch.int32)
    generations = torch.tensor([7], dtype=torch.int64)
    resident_cpu = torch.full((1, 16), -1, dtype=torch.int32)
    resident_generations_cpu = torch.full((1, 8), -1, dtype=torch.int64)
    resident_npu = resident_cpu.npu()
    resident_generations_npu = resident_generations_cpu.npu()

    _run_reuse_step(
        _reuse_topk([[1, 2, 30], [2, 3, 31]]),
        torch.tensor([4, 4], dtype=torch.int32),
        row_req,
        table,
        state_indices,
        generations,
        resident_cpu,
        resident_generations_cpu,
        resident_npu,
        resident_generations_npu,
    )
    zero_miss = _run_reuse_step(
        _reuse_topk([[3, 2, 30], [2, 3, 31]]),
        torch.tensor([4, 4], dtype=torch.int32),
        row_req,
        table,
        state_indices,
        generations,
        resident_cpu,
        resident_generations_cpu,
        resident_npu,
        resident_generations_npu,
    )
    assert zero_miss[2].tolist() == [0]

    evicted = _run_reuse_step(
        _reuse_topk([[2, 4, 30], [4, 5, 31]]),
        torch.tensor([8, 8], dtype=torch.int32),
        row_req,
        table,
        state_indices,
        generations,
        resident_cpu,
        resident_generations_cpu,
        resident_npu,
        resident_generations_npu,
    )
    assert evicted[2].tolist() == [2]
    assert evicted[1][0, :2].tolist() == [4, 5]
    # Token 2 remains in slot 1; misses use the non-union slots 0 and 2.
    assert evicted[3][0, :2].tolist() == [40, 42]


def test_reuse_operator_q1_two_steps_zero_miss():
    """The one-row path covers num_speculative_tokens=0 explicitly."""
    _require_reuse_operator()
    table = _aligned([[20, 21, 22, 23, 24, 25, 26, 27]])
    row_req = torch.tensor([0], dtype=torch.int32)
    state_indices = torch.tensor([0], dtype=torch.int32)
    generations = torch.tensor([7], dtype=torch.int64)
    resident_cpu = torch.full((1, 16), -1, dtype=torch.int32)
    resident_generations_cpu = torch.full((1, 8), -1, dtype=torch.int64)
    resident_npu = resident_cpu.npu()
    resident_generations_npu = resident_generations_cpu.npu()

    first = _run_reuse_step(
        _reuse_topk([[1, 2, 30]]),
        torch.tensor([4], dtype=torch.int32),
        row_req,
        table,
        state_indices,
        generations,
        resident_cpu,
        resident_generations_cpu,
        resident_npu,
        resident_generations_npu,
    )
    assert first[0][0, :2].tolist() == [0, 1]
    assert first[1][0, :2].tolist() == [1, 2]
    assert first[2].tolist() == [2]
    assert first[3][0, :2].tolist() == [40, 41]

    second = _run_reuse_step(
        _reuse_topk([[2, 1, 30]]),
        torch.tensor([4], dtype=torch.int32),
        row_req,
        table,
        state_indices,
        generations,
        resident_cpu,
        resident_generations_cpu,
        resident_npu,
        resident_generations_npu,
    )
    assert second[0][0, :2].tolist() == [1, 0]
    assert second[2].tolist() == [0]


def test_reuse_operator_stable_rows_and_generation_reset():
    _require_reuse_operator()
    table = _aligned(
        [
            [20, 21, 22, 23, 24, 25, 26, 27],
            [30, 31, 32, 33, 34, 35, 36, 37],
        ]
    )
    resident_cpu = torch.full((3, 16), -1, dtype=torch.int32)
    resident_cpu[0, 0] = 6
    resident_cpu[2, 3] = 1
    resident_generations_cpu = torch.full((3, 8), -1, dtype=torch.int64)
    resident_generations_cpu[0, 0] = 20
    resident_generations_cpu[2, 0] = 9
    resident_npu = resident_cpu.npu()
    resident_generations_npu = resident_generations_cpu.npu()

    result = _run_reuse_step(
        _reuse_topk([[2, 1, 30], [6, 7, 31]]),
        torch.tensor([8, 8], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        table,
        torch.tensor([2, 0], dtype=torch.int32),
        # Request 0 changes generation, so its apparent token-1 hit is stale.
        torch.tensor([10, 20], dtype=torch.int64),
        resident_cpu,
        resident_generations_cpu,
        resident_npu,
        resident_generations_npu,
    )
    assert result[2].tolist() == [2, 1]
    assert result[1][0, :2].tolist() == [2, 1]
    assert result[1][1, 0].item() == 7


def test_reuse_operator_survives_compact_request_reordering():
    """A stable state row may move to a different request AIV next step."""
    _require_reuse_operator()
    table_a = [20, 21, 22, 23, 24, 25, 26, 27]
    table_b = [30, 31, 32, 33, 34, 35, 36, 37]
    resident_cpu = torch.full((2, 16), -1, dtype=torch.int32)
    resident_generations_cpu = torch.full((2, 8), -1, dtype=torch.int64)
    resident_npu = resident_cpu.npu()
    resident_generations_npu = resident_generations_cpu.npu()

    # Compact request order is [A, B], so AIV 0 owns stable row 0 and AIV 1
    # owns stable row 1 for this launch.
    _run_reuse_step(
        _reuse_topk([[1, 2, 30], [5, 6, 31]]),
        torch.tensor([8, 8], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        _aligned([table_a, table_b]),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([10, 20], dtype=torch.int64),
        resident_cpu,
        resident_generations_cpu,
        resident_npu,
        resident_generations_npu,
    )

    # The batch compacts to [B, A]. Stable row 1 is now handled by AIV 0 and
    # stable row 0 by AIV 1; both prior-step hits must remain visible.
    reordered = _run_reuse_step(
        _reuse_topk([[6, 7, 31], [2, 3, 30]]),
        torch.tensor([8, 8], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        _aligned([table_b, table_a]),
        torch.tensor([1, 0], dtype=torch.int32),
        torch.tensor([20, 10], dtype=torch.int64),
        resident_cpu,
        resident_generations_cpu,
        resident_npu,
        resident_generations_npu,
    )
    assert reordered[0][:, :2].tolist() == [[1, 0], [1, 0]]
    assert reordered[1][:, 0].tolist() == [7, 3]
    assert reordered[2].tolist() == [1, 1]
    assert reordered[3][:, 0].tolist() == [60, 40]


def test_reuse_operator_aclgraph_replay_preserves_reordered_state():
    """Capture once, then replay two steps with stable buffers and reordered requests."""
    _require_reuse_operator()
    if not hasattr(torch.npu, "NPUGraph") or not hasattr(torch.npu, "graph"):
        pytest.fail("torch_npu does not expose the ACL graph capture API")

    table_a = [20, 21, 22, 23, 24, 25, 26, 27]
    table_b = [30, 31, 32, 33, 34, 35, 36, 37]
    topk = _reuse_topk([[30], [31], [32], [33]]).npu()
    boundaries = torch.tensor(
        [8, 8, 8, 8],
        dtype=torch.int32,
        device="npu",
    )
    row_req = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.int32,
        device="npu",
    )
    table = _aligned([table_a, table_b]).npu()
    state_indices = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    # Capture uses padding generations so it cannot mutate persistent state.
    generations = torch.tensor([-1, -2], dtype=torch.int64, device="npu")
    resident = torch.full((2, 16), -1, dtype=torch.int32, device="npu")
    resident_generations = torch.full(
        (2, 8), -1, dtype=torch.int64, device="npu"
    )
    selected, counts, targets = _buffers(2, 16)

    def launch():
        prepare_sparse_indices(
            topk,
            boundaries,
            row_req_indices=row_req,
            request_block_table=table,
            selected_packed=selected,
            selected_counts=counts,
            target_slot_mapping=targets,
            block_size=2,
            request_state_indices=state_indices,
            request_generations=generations,
            resident_token_ids=resident,
            resident_generations=resident_generations,
        )

    # Warm up the custom-op launch before entering capture, matching vLLM's
    # eager-warmup -> capture lifecycle.
    launch()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch()
    torch.npu.synchronize()

    resident_expected = torch.full((2, 16), -1, dtype=torch.int32)
    resident_generations_expected = torch.full(
        (2, 8), -1, dtype=torch.int64
    )

    def replay_and_check(
        topk_cpu,
        table_cpu,
        state_indices_cpu,
        generations_cpu,
    ):
        expected = _prepare_sparse_indices_reuse_torch(
            topk_cpu.clone(),
            torch.tensor([8, 8, 8, 8], dtype=torch.int32),
            torch.tensor([0, 0, 1, 1], dtype=torch.int32),
            table_cpu,
            state_indices_cpu,
            generations_cpu,
            resident_expected,
            resident_generations_expected,
            2,
        )
        topk.copy_(topk_cpu)
        table.copy_(table_cpu)
        state_indices.copy_(state_indices_cpu)
        generations.copy_(generations_cpu)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()

        assert torch.equal(topk.cpu(), expected[0])
        assert torch.equal(counts[:, 0].cpu(), expected[2])
        for request, count in enumerate(expected[2].tolist()):
            assert torch.equal(
                selected[request, :count].cpu(),
                expected[1][request, :count],
            )
            assert torch.equal(
                targets[request, :count].cpu(),
                expected[3][request, :count],
            )
        assert torch.equal(resident.cpu(), resident_expected)
        assert torch.equal(
            resident_generations.cpu(),
            resident_generations_expected,
        )

    replay_and_check(
        _reuse_topk([[1, 2], [2, 3], [5, 6], [6, 7]]),
        _aligned([table_a, table_b]),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([10, 20], dtype=torch.int64),
    )
    # Compact order changes from [A, B] to [B, A]. The same captured AIV
    # launches now access the opposite stable rows and must observe prior hits.
    replay_and_check(
        _reuse_topk([[6, 7], [7, 8], [2, 3], [3, 4]]),
        _aligned([table_b, table_a]),
        torch.tensor([1, 0], dtype=torch.int32),
        torch.tensor([20, 10], dtype=torch.int64),
    )


def test_reuse_operator_lazily_resets_generation_on_first_scratch_use():
    _require_reuse_operator()
    table = _aligned([[20, 21, 22, 23, 24, 25, 26, 27]])
    resident_cpu = torch.full((1, 16), -1, dtype=torch.int32)
    resident_cpu[0, :2] = torch.tensor([5, 6], dtype=torch.int32)
    resident_generations_cpu = torch.full((1, 8), -1, dtype=torch.int64)
    resident_generations_cpu[0, 0] = 4
    resident_npu = resident_cpu.npu()
    resident_generations_npu = resident_generations_cpu.npu()

    zero_boundary = _run_reuse_step(
        _reuse_topk([[5, 6, 30]]),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        table,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([5], dtype=torch.int64),
        resident_cpu,
        resident_generations_cpu,
        resident_npu,
        resident_generations_npu,
    )
    assert zero_boundary[2].tolist() == [0]
    assert resident_cpu[0, :2].tolist() == [5, 6]
    assert resident_generations_cpu[0, 0].item() == 4

    first_scratch_use = _run_reuse_step(
        _reuse_topk([[5, 7, 30]]),
        torch.tensor([8], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        table,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([5], dtype=torch.int64),
        resident_cpu,
        resident_generations_cpu,
        resident_npu,
        resident_generations_npu,
    )
    assert first_scratch_use[1][0, :2].tolist() == [5, 7]
    assert first_scratch_use[2].tolist() == [2]
    assert resident_generations_cpu[0, 0].item() == 5


def test_reuse_operator_padding_generation_does_not_touch_state():
    _require_reuse_operator()
    resident_cpu = torch.arange(16, dtype=torch.int32).reshape(1, 16)
    resident_generations_cpu = torch.full(
        (1, 8), 123, dtype=torch.int64
    )
    resident_npu = resident_cpu.npu()
    resident_generations_npu = resident_generations_cpu.npu()

    result = _run_reuse_step(
        _reuse_topk([[1, 2, 30]]),
        torch.tensor([8], dtype=torch.int32),
        torch.tensor([-1], dtype=torch.int32),
        _aligned([[20, 21, 22, 23, 24, 25, 26, 27]]),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int64),
        resident_cpu,
        resident_generations_cpu,
        resident_npu,
        resident_generations_npu,
        clear_invalid_rows=True,
    )
    assert torch.count_nonzero(result[0]).item() == 0
    assert result[2].tolist() == [0]
