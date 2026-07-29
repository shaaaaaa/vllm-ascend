import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
    _prepare_sparse_indices_torch,
    prepare_sparse_indices,
)
from vllm_ascend.distributed.kv_transfer.sparse_offload.resident_sparse_cache import (
    allocate_resident_workspace,
    prepare_resident_sparse_cache_,
)
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module", autouse=True)
def _load_dsa_union_operator():
    if not enable_custom_op():
        pytest.fail("vllm-ascend custom operators could not be loaded")
    if not hasattr(torch.ops._C_ascend, "npu_dsa_prepare_sparse_indices_"):
        pytest.fail("vllm_ascend_C does not contain the DSA union operator")
    if not hasattr(
        torch.ops._C_ascend,
        "npu_dsa_prepare_sparse_indices_staged_",
    ):
        pytest.fail(
            "vllm_ascend_C does not contain the production staged DSA operator"
        )
    if not hasattr(
        torch.ops._C_ascend,
        "npu_dsa_prepare_sparse_indices_sharded_",
    ):
        pytest.fail(
            "vllm_ascend_C does not contain the production sharded DSA operator"
        )
    if not hasattr(torch.ops._C_ascend, "npu_dsa_prepare_sparse_indices_legacy_"):
        pytest.fail("vllm_ascend_C does not contain the pre-union DSA operator")
    if not hasattr(torch.ops._C_ascend, "npu_dsa_staged_unique_finalize_"):
        pytest.fail("vllm_ascend_C does not contain the unique finalize operator")
    if not hasattr(torch.ops._C_ascend, "npu_dsa_staged_sharded_union_"):
        pytest.fail("vllm_ascend_C does not contain the sharded union operator")
    if not hasattr(
        torch.ops._C_ascend,
        "npu_dsa_staged_sharded_vector_union_",
    ):
        pytest.fail("vllm_ascend_C does not contain the vector sharded union operator")
    if not hasattr(
        torch.ops._C_ascend,
        "npu_dsa_staged_sharded_vector_dedup_",
    ):
        pytest.fail("vllm_ascend_C does not contain the vector-dedup operator")
    if not hasattr(
        torch.ops._C_ascend,
        "npu_dsa_resident_remap_rows_",
    ):
        pytest.fail(
            "vllm_ascend_C does not contain the resident parallel-map "
            "operator"
        )
    for name in (
        "npu_dsa_resident_lookup_rows_",
        "npu_dsa_resident_finalize_rows_",
    ):
        if not hasattr(torch.ops._C_ascend, name):
            pytest.fail(
                f"vllm_ascend_C does not contain the {name} operator"
            )


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


def _run_production_staged(
    source: torch.Tensor,
    boundaries: torch.Tensor,
    request_count: int,
    mtp: int,
    *,
    capture_graph: bool = False,
    use_sharded_sort: bool = True,
):
    row_width = source.numel() // source.shape[0]
    capacity = mtp * row_width
    block_size = 128
    table_width = capacity // block_size
    values = source.npu()
    boundaries_npu = boundaries.npu()
    row_requests = torch.arange(
        request_count,
        dtype=torch.int32,
        device="npu",
    ).repeat_interleave(mtp)
    selected, counts, targets = _buffers(request_count, capacity)
    mapping = torch.empty_like(selected)
    shard_packed = torch.empty(
        (request_count, 2, capacity),
        dtype=torch.int32,
        device="npu",
    )
    shard_mapping = torch.empty_like(shard_packed)
    shard_counts = torch.empty(
        (request_count, 2, 16),
        dtype=torch.int32,
        device="npu",
    )
    block_table = torch.arange(
        request_count * table_width,
        dtype=torch.int32,
        device="npu",
    ).reshape(request_count, table_width)
    addresses = tuple(
        tensor.data_ptr()
        for tensor in (
            values,
            selected,
            counts,
            targets,
            mapping,
            shard_packed,
            shard_mapping,
            shard_counts,
        )
    )

    def invoke():
        op = (
            torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_sharded_
            if use_sharded_sort
            else torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_staged_
        )
        args = [
            values,
            boundaries_npu,
            row_requests,
            block_table,
            selected,
            counts,
            targets,
            mapping,
        ]
        if use_sharded_sort:
            args.extend((shard_packed, shard_mapping, shard_counts))
        op(*args, block_size, mtp, True, True)

    if capture_graph:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            invoke()
        values.copy_(source.npu())
        boundaries_npu.copy_(boundaries.npu())
        graph.replay()
    else:
        invoke()
    torch.npu.synchronize()
    assert addresses == tuple(
        tensor.data_ptr()
        for tensor in (
            values,
            selected,
            counts,
            targets,
            mapping,
            shard_packed,
            shard_mapping,
            shard_counts,
        )
    )
    return (
        values.cpu(),
        selected.cpu(),
        counts[:, 0].cpu(),
        targets.cpu(),
        block_table.cpu(),
    )


def _run_vector_sharded_union(
    source: torch.Tensor,
    boundaries: torch.Tensor,
    request_count: int,
    *,
    capture_graph: bool = False,
    use_position_map: bool = False,
):
    topk = source.shape[-1]
    row_count = source.shape[0]
    mtp = row_count // request_count
    capacity = mtp * topk
    shard_count = 1 << (mtp - 1).bit_length()
    block_size = 128
    max_tokens = 131072
    values = source.npu()
    boundaries_npu = boundaries.npu()
    selected, counts, targets = _buffers(request_count, capacity)
    local_to_union = torch.empty((row_count, topk), dtype=torch.int32, device="npu")
    block_table = torch.arange(
        request_count * max_tokens // block_size,
        dtype=torch.int32,
        device="npu",
    ).reshape(request_count, max_tokens // block_size)
    shard_packed = torch.empty(
        (request_count, shard_count, topk),
        dtype=torch.int32,
        device="npu",
    )
    shard_mapping = torch.empty(
        (request_count, shard_count, capacity),
        dtype=torch.int32,
        device="npu",
    )
    shard_counts = torch.empty(
        (request_count, shard_count, 16),
        dtype=torch.int32,
        device="npu",
    )
    shard_pairs = None
    if use_position_map:
        op = torch.ops._C_ascend.npu_dsa_staged_sharded_vector_dedup_
    else:
        shard_pairs = torch.empty(
            (request_count, shard_count, 2 * topk),
            dtype=torch.int32,
            device="npu",
        )
        op = torch.ops._C_ascend.npu_dsa_staged_sharded_vector_union_

    def invoke():
        common_args = (
            values,
            boundaries_npu,
            selected,
            local_to_union,
            counts,
            block_table,
            targets,
            shard_packed,
            shard_mapping,
            shard_counts,
        )
        if use_position_map:
            op(*common_args, block_size)
        else:
            op(*common_args, shard_pairs, block_size)

    if capture_graph:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            invoke()
        # Replay more than once with the original absolute indices restored.
        # This catches missing cross-pipeline dependencies that otherwise let
        # one replay consume mapping or row data left by the previous replay.
        source_npu = source.npu()
        for _ in range(3):
            values.copy_(source_npu)
            graph.replay()
    else:
        invoke()
    torch.npu.synchronize()
    return {
        "values": values.cpu(),
        "selected": selected.cpu(),
        "counts": counts.cpu(),
        "targets": targets.cpu(),
        "local_to_union": local_to_union.cpu(),
        "shard_packed": shard_packed.cpu(),
        "shard_counts": shard_counts.cpu(),
    }


def _assert_vector_sharded_result(
    result,
    source: torch.Tensor,
    boundaries: torch.Tensor,
    request_count: int,
):
    row_count, _, topk = source.shape
    mtp = row_count // request_count
    capacity = mtp * topk
    remapped = result["values"].reshape(row_count, topk)
    local_to_union = result["local_to_union"].reshape(row_count, topk)
    for request in range(request_count):
        first_row = request * mtp
        expected_tokens = set()
        for local_row in range(mtp):
            row = first_row + local_row
            boundary = int(boundaries[row])
            for token in source[row].flatten().tolist():
                if 0 <= token < boundary:
                    expected_tokens.add(token)
        count = result["counts"][request, 0].item()
        assert count == len(expected_tokens)
        actual_tokens = result["selected"][request, :count].tolist()
        assert len(actual_tokens) == len(set(actual_tokens))
        assert set(actual_tokens) == expected_tokens
        assert torch.equal(
            result["targets"][request, :count],
            torch.arange(count, dtype=torch.long) + request * 131072,
        )

        for local_row in range(mtp):
            row = first_row + local_row
            boundary = int(boundaries[row])
            original = source[row].flatten()
            selected_mask = (original >= 0) & (original < boundary)
            ranks = remapped[row]
            assert torch.equal(
                ranks[selected_mask],
                local_to_union[row][selected_mask],
            )
            assert torch.equal(
                remapped[row][~selected_mask],
                original[~selected_mask],
            )
            if selected_mask.any():
                reconstructed = result["selected"][request, ranks[selected_mask].to(torch.long)]
                assert torch.equal(
                    reconstructed,
                    original[selected_mask],
                )
                assert torch.all(ranks[selected_mask] >= 0)
                assert torch.all(ranks[selected_mask] < count)
            assert torch.all(local_to_union[row][~selected_mask] < 0)
        assert count <= capacity


def test_pre_union_and_union_ops_with_half_overlapping_mtp_rows():
    topk = 2048
    shared = torch.arange(topk // 2, dtype=torch.int32)
    unique = torch.arange(topk // 2, topk + topk // 2, dtype=torch.int32)
    topk_indices = torch.stack(
        (
            torch.cat((shared, unique[: topk // 2])),
            torch.cat((shared, unique[topk // 2 :])),
        )
    ).unsqueeze(1)
    assert len(set(topk_indices[0].flatten().tolist()).intersection(topk_indices[1].flatten().tolist())) / topk == 0.5

    boundaries = torch.full((2,), 131072, dtype=torch.int32)
    row_requests = torch.zeros(2, dtype=torch.int32)
    block_size = 128
    tables = torch.arange(2 * topk // block_size, dtype=torch.int32).unsqueeze(0)

    legacy_values = topk_indices.npu()
    legacy_packed = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_legacy_(
        legacy_values,
        boundaries.npu(),
        torch.arange(2, dtype=torch.int32, device="npu"),
        torch.tensor([0, topk], dtype=torch.int32, device="npu"),
        True,
        row_requests.npu(),
    )
    union_buffers = _buffers(1, 2 * topk)
    union_values = topk_indices.npu()
    union_result = prepare_sparse_indices(
        union_values,
        boundaries.npu(),
        row_req_indices=row_requests.npu(),
        request_block_table=tables.npu(),
        selected_packed=union_buffers[0],
        selected_counts=union_buffers[1],
        target_slot_mapping=union_buffers[2],
        block_size=block_size,
    )
    torch.npu.synchronize()

    assert torch.equal(legacy_packed.cpu(), topk_indices.squeeze(1))
    assert union_result[2].cpu().tolist() == [3 * topk // 2]
    assert torch.equal(
        union_result[0][0, :, : topk // 2].cpu(),
        union_result[0][1, :, : topk // 2].cpu(),
    )


def test_four_stage_native_unique_with_batched_mtp_rows():
    topk = 2048
    request_count = 4
    row_count = 2 * request_count
    max_tokens = 131072
    shared = torch.arange(topk // 2, dtype=torch.int32)
    unique = torch.arange(topk // 2, topk + topk // 2, dtype=torch.int32)
    request_rows = torch.stack(
        (
            torch.cat((shared, unique[: topk // 2])),
            torch.cat((shared, unique[topk // 2 :])),
        )
    )
    source = request_rows.repeat(request_count, 1).unsqueeze(1).npu()
    values = source.clone()
    row_requests = torch.arange(request_count, dtype=torch.int32, device="npu").repeat_interleave(2)
    boundaries = torch.full((row_count,), max_tokens, dtype=torch.int32, device="npu")
    valid_rows = torch.arange(row_count, dtype=torch.int32, device="npu")
    scratch_base = torch.zeros(row_count, dtype=torch.int32, device="npu")
    block_size = 128
    capacity = 2 * topk
    block_table = torch.arange(
        request_count * capacity // block_size,
        dtype=torch.int32,
        device="npu",
    ).reshape(request_count, capacity // block_size)
    selected, counts, targets = _buffers(request_count, capacity)
    local_to_union = torch.empty((row_count, topk), dtype=torch.int32, device="npu")

    packed_keys = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_legacy_(
        values,
        boundaries,
        valid_rows,
        scratch_base,
        True,
        row_requests,
        max_tokens,
    )
    unique_keys, inverse = torch.unique(
        packed_keys.reshape(-1),
        sorted=True,
        return_inverse=True,
    )
    torch.ops._C_ascend.npu_dsa_staged_unique_finalize_(
        unique_keys,
        inverse,
        row_requests,
        selected,
        local_to_union,
        counts,
        block_table,
        targets,
        block_size,
        max_tokens,
    )
    torch.ops._C_ascend.npu_dsa_staged_remap_rows_(values, local_to_union)
    torch.npu.synchronize()

    expected_count = 3 * topk // 2
    assert counts[:, 0].cpu().tolist() == [expected_count] * request_count
    expected_selected = torch.arange(expected_count, dtype=torch.int32)
    for request in range(request_count):
        assert torch.equal(
            selected[request, :expected_count].cpu(),
            expected_selected,
        )
        assert torch.equal(
            targets[request, :expected_count].cpu(),
            torch.arange(expected_count, dtype=torch.long) + request * capacity,
        )
    reconstructed = torch.gather(
        selected.repeat_interleave(2, dim=0),
        1,
        values.reshape(row_count, topk).to(torch.long),
    )
    assert torch.equal(reconstructed.cpu(), source.reshape(row_count, topk).cpu())


@pytest.mark.parametrize("mtp", [1, 2, 3, 4])
def test_sharded_union_tracks_mtp_depth(mtp):
    topk = 2048
    request_count = 2
    row_count = request_count * mtp
    shared = torch.arange(topk // 2, dtype=torch.int32)
    request_rows = torch.stack(
        tuple(
            torch.cat(
                (
                    shared,
                    torch.arange(
                        topk // 2 + row * topk // 2,
                        topk // 2 + (row + 1) * topk // 2,
                        dtype=torch.int32,
                    ),
                )
            )
            for row in range(mtp)
        )
    )
    source = request_rows.repeat(request_count, 1).unsqueeze(1).npu()
    values = source.clone()
    source_max = (mtp + 1) * topk // 2 - 1
    boundary = source_max - 100
    boundaries = torch.full((row_count,), boundary, dtype=torch.int32, device="npu")
    capacity = mtp * topk
    block_size = 128
    block_table = torch.arange(
        request_count * 131072 // block_size,
        dtype=torch.int32,
        device="npu",
    ).reshape(request_count, 131072 // block_size)
    selected, counts, targets = _buffers(request_count, capacity)
    local_to_union = torch.empty((row_count, topk), dtype=torch.int32, device="npu")
    shard_count = 1 << (mtp - 1).bit_length()
    shard_packed = torch.empty(
        (request_count, shard_count, topk),
        dtype=torch.int32,
        device="npu",
    )
    shard_mapping = torch.empty(
        (request_count, shard_count, capacity),
        dtype=torch.int32,
        device="npu",
    )
    shard_counts = torch.empty(
        (request_count, shard_count, 16),
        dtype=torch.int32,
        device="npu",
    )

    torch.ops._C_ascend.npu_dsa_staged_sharded_union_(
        values,
        boundaries,
        selected,
        local_to_union,
        counts,
        block_table,
        targets,
        shard_packed,
        shard_mapping,
        shard_counts,
        block_size,
    )
    torch.npu.synchronize()

    expected_count = boundary
    assert counts[:, 0].cpu().tolist() == [expected_count] * request_count
    selected_mask = source.reshape(row_count, topk) < boundary
    remapped = values.reshape(row_count, topk)
    safe_indices = torch.where(selected_mask, remapped, torch.zeros_like(remapped))
    selected_reconstructed = torch.gather(
        selected.repeat_interleave(mtp, dim=0),
        1,
        safe_indices.to(torch.long),
    )
    reconstructed = torch.where(
        selected_mask,
        selected_reconstructed,
        remapped,
    )
    assert torch.equal(reconstructed.cpu(), source.reshape(row_count, topk).cpu())
    expected_tokens = set(range(expected_count))
    for request in range(request_count):
        assert set(selected[request, :expected_count].cpu().tolist()) == expected_tokens
    if mtp == 3:
        values.copy_(source)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            torch.ops._C_ascend.npu_dsa_staged_sharded_union_(
                values,
                boundaries,
                selected,
                local_to_union,
                counts,
                block_table,
                targets,
                shard_packed,
                shard_mapping,
                shard_counts,
                block_size,
            )
        values.copy_(source)
        graph.replay()
        torch.npu.synchronize()
        assert counts[:, 0].cpu().tolist() == [expected_count] * request_count


@pytest.mark.parametrize("mtp", range(1, 9))
@pytest.mark.parametrize(
    "use_position_map",
    [False, True],
    ids=["pair-map", "position-map"],
)
def test_vector_sharded_union_all_supported_mtp_depths(mtp, use_position_map):
    topk = 2048
    request_count = 2
    shared = torch.arange(topk // 2, dtype=torch.int32)
    request_rows = torch.stack(
        tuple(
            torch.cat(
                (
                    shared,
                    torch.arange(
                        topk // 2 + row * topk // 2,
                        topk // 2 + (row + 1) * topk // 2,
                        dtype=torch.int32,
                    ),
                )
            )
            for row in range(mtp)
        )
    )
    source = request_rows.repeat(request_count, 1).unsqueeze(1)
    source_max = (mtp + 1) * topk // 2 - 1
    boundary = source_max - 100
    boundaries = torch.full((request_count * mtp,), boundary, dtype=torch.int32)

    result = _run_vector_sharded_union(
        source,
        boundaries,
        request_count,
        use_position_map=use_position_map,
    )
    _assert_vector_sharded_result(result, source, boundaries, request_count)

    shard_count = 1 << (mtp - 1).bit_length()
    for request in range(request_count):
        occurrence_total = 0
        unique_total = 0
        for shard in range(shard_count):
            unique_count = result["shard_counts"][request, shard, 0].item()
            occurrence_count = result["shard_counts"][request, shard, 1].item()
            assert 0 <= unique_count <= occurrence_count <= topk
            shard_tokens = result["shard_packed"][request, shard, :unique_count]
            assert torch.equal(
                shard_tokens,
                torch.sort(shard_tokens).values,
            )
            assert all(token % shard_count == shard for token in shard_tokens.tolist())
            occurrence_total += occurrence_count
            unique_total += unique_count
        assert unique_total == boundary
        assert occurrence_total == mtp * topk - 101
        if use_position_map and mtp > 1:
            expected_offsets = []
            offset = 0
            for shard in range(shard_count):
                expected_offsets.append(offset)
                offset += result["shard_counts"][request, shard, 0].item()
            assert result["counts"][request, 1 : 1 + shard_count].tolist() == expected_offsets


@pytest.mark.parametrize(
    "use_position_map",
    [False, True],
    ids=["pair-map", "position-map"],
)
def test_vector_sharded_union_uses_each_rows_split_boundary(
    use_position_map,
):
    topk = 2048
    mtp = 4
    request_count = 2
    rows = []
    boundaries = []
    for request in range(request_count):
        for local_row in range(mtp):
            row = torch.arange(
                local_row * 512,
                local_row * 512 + topk,
                dtype=torch.int32,
            )
            row = torch.roll(row, 137 * (local_row + 1))
            row[0] = -1 - request * mtp - local_row
            rows.append(row)
            boundaries.append(650 + request * 200 + local_row * 425)
    source = torch.stack(rows).unsqueeze(1)
    boundary_tensor = torch.tensor(boundaries, dtype=torch.int32)

    result = _run_vector_sharded_union(
        source,
        boundary_tensor,
        request_count,
        use_position_map=use_position_map,
    )
    _assert_vector_sharded_result(result, source, boundary_tensor, request_count)


@pytest.mark.parametrize(
    "use_position_map",
    [False, True],
    ids=["pair-map", "position-map"],
)
def test_vector_sharded_union_mtp1_preserves_compacted_topk_order(
    use_position_map,
):
    topk = 2048
    request_count = 2
    rows = []
    boundaries = torch.tensor([1400, 1900], dtype=torch.int32)
    for request in range(request_count):
        row = torch.arange(topk, dtype=torch.int32)
        row = torch.roll(row, 317 * (request + 1))
        rows.append(row)
    source = torch.stack(rows).unsqueeze(1)

    result = _run_vector_sharded_union(
        source,
        boundaries,
        request_count,
        use_position_map=use_position_map,
    )
    _assert_vector_sharded_result(result, source, boundaries, request_count)
    for request in range(request_count):
        original = source[request].flatten()
        expected = original[original < boundaries[request]]
        count = result["counts"][request, 0].item()
        assert count == expected.numel()
        assert torch.equal(
            result["selected"][request, :count],
            expected,
        )
        assert result["shard_counts"][request, 0, 0].item() == count
        assert result["shard_counts"][request, 0, 1].item() == count


@pytest.mark.parametrize(
    "use_position_map",
    [False, True],
    ids=["pair-map", "position-map"],
)
def test_vector_sharded_union_copies_unaligned_tails_exactly(
    use_position_map,
):
    topk = 2048
    request_count = 8
    source = torch.stack(
        tuple(
            torch.roll(
                torch.arange(topk, dtype=torch.int32),
                137 * request,
            )
            for request in range(request_count)
        )
    ).unsqueeze(1)
    # Exercise every int32 count residue modulo one 32-byte data block.
    boundaries = torch.arange(193, 201, dtype=torch.int32)

    result = _run_vector_sharded_union(
        source,
        boundaries,
        request_count,
        use_position_map=use_position_map,
    )
    _assert_vector_sharded_result(
        result,
        source,
        boundaries,
        request_count,
    )
    for request, boundary in enumerate(boundaries.tolist()):
        original = source[request].flatten()
        expected = original[original < boundary]
        count = result["counts"][request, 0].item()
        assert count == boundary
        assert torch.equal(
            result["selected"][request, :count],
            expected,
        )


@pytest.mark.parametrize(
    "use_position_map",
    [False, True],
    ids=["pair-map", "position-map"],
)
def test_vector_sharded_union_preserves_all_ignored_positions(
    use_position_map,
):
    topk = 2048
    mtp = 3
    request_count = 2
    rows = torch.stack(
        tuple(
            torch.roll(
                torch.arange(topk, dtype=torch.int32) + row * topk,
                211 * (row + 1),
            )
            for row in range(mtp)
        )
    )
    source = rows.repeat(request_count, 1).unsqueeze(1)
    boundaries = torch.tensor(
        [0, -1, 0, -7, 0, -3],
        dtype=torch.int32,
    )

    result = _run_vector_sharded_union(
        source,
        boundaries,
        request_count,
        use_position_map=use_position_map,
    )
    _assert_vector_sharded_result(result, source, boundaries, request_count)
    assert result["counts"][:, 0].tolist() == [0, 0]
    assert torch.equal(
        result["values"].reshape_as(source),
        source,
    )
    assert torch.all(result["local_to_union"] < 0)
    assert torch.all(result["shard_counts"][:, :, :2] == 0)


@pytest.mark.parametrize("mtp", [1, 3, 8])
@pytest.mark.parametrize(
    "use_position_map",
    [False, True],
    ids=["pair-map", "position-map"],
)
def test_vector_sharded_union_supports_graph_replay(mtp, use_position_map):
    topk = 2048
    request_count = 2
    rows = torch.stack(
        tuple(
            torch.roll(
                torch.arange(topk, dtype=torch.int32) + row * 256,
                97 * (row + 1),
            )
            for row in range(mtp)
        )
    )
    source = rows.repeat(request_count, 1).unsqueeze(1)
    boundaries = torch.tensor(
        [1200 + 100 * (row % mtp) for row in range(request_count * mtp)],
        dtype=torch.int32,
    )

    result = _run_vector_sharded_union(
        source,
        boundaries,
        request_count,
        capture_graph=True,
        use_position_map=use_position_map,
    )
    _assert_vector_sharded_result(result, source, boundaries, request_count)


def test_mtp_rows_build_one_sorted_union_per_request():
    topk = _aligned([[3, 1, 8], [2, 3, 9], [4, 1, 10]])
    row_requests = torch.tensor([0, 0, 1], dtype=torch.int32)
    boundaries = torch.tensor([5, 5, 5], dtype=torch.int32)
    tables = _aligned([[20, 21, 22, 23, 24, 25], [30, 31, 32, 33, 34, 35]])
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


@pytest.mark.parametrize("capture_graph", [False, True])
@pytest.mark.parametrize(
    "use_sharded_sort",
    [True, False],
    ids=["sharded-sort", "legacy-staged-sort"],
)
def test_production_mtp2_backends_respect_per_row_boundary_and_graph(
    capture_graph,
    use_sharded_sort,
):
    request_count = 2
    mtp = 2
    topk = 2048
    rows = []
    boundaries = []
    for request in range(request_count):
        base = request * 8192
        first = torch.roll(
            torch.arange(base, base + topk, dtype=torch.int32),
            137,
        )
        second = torch.roll(
            torch.arange(base + 1024, base + 1024 + topk, dtype=torch.int32),
            293,
        )
        first[0] = -1
        second[1] = -1
        rows.extend((first, second))
        boundaries.extend(
            (
                0 if request == 0 else int(first.max()) - 100,
                0 if request == 0 else int(second.max()) - 100,
            )
        )
    source = torch.stack(rows).unsqueeze(1)
    boundary_tensor = torch.tensor(boundaries, dtype=torch.int32)
    row_requests = torch.arange(
        request_count,
        dtype=torch.int32,
    ).repeat_interleave(mtp)
    table_width = mtp * topk // 128
    block_table = torch.arange(
        request_count * table_width,
        dtype=torch.int32,
    ).reshape(request_count, table_width)
    expected = _prepare_sparse_indices_torch(
        source,
        boundary_tensor,
        row_req_indices=row_requests,
        request_block_table=block_table,
        block_size=128,
    )

    actual = _run_production_staged(
        source,
        boundary_tensor,
        request_count,
        mtp,
        capture_graph=capture_graph,
        use_sharded_sort=use_sharded_sort,
    )

    assert torch.equal(actual[2], expected[2])
    if use_sharded_sort:
        source_2d = source.reshape(request_count * mtp, topk)
        remapped = actual[0].reshape(request_count * mtp, topk)
        selected_mask = (
            (source_2d >= 0)
            & (source_2d < boundary_tensor.reshape(-1, 1))
        )
        safe_ranks = torch.where(
            selected_mask,
            remapped,
            torch.zeros_like(remapped),
        )
        reconstructed_selected = torch.gather(
            actual[1].repeat_interleave(mtp, dim=0),
            1,
            safe_ranks.to(torch.long),
        )
        reconstructed = torch.where(
            selected_mask,
            reconstructed_selected,
            remapped,
        )
        assert torch.equal(reconstructed, source_2d)
    else:
        assert torch.equal(actual[0], expected[0])
    for request, count in enumerate(expected[2].tolist()):
        actual_selected = actual[1][request, :count]
        expected_selected = expected[1][request, :count]
        if use_sharded_sort:
            assert torch.equal(
                torch.sort(actual_selected).values,
                expected_selected,
            )
        else:
            assert torch.equal(actual_selected, expected_selected)
        assert torch.equal(
            actual[3][request, :count],
            expected[3][request, :count],
        )


def test_production_sharded_mtp2_handles_one_full_value_shard():
    topk = 2048
    source = torch.stack(
        (
            2 * torch.arange(topk, dtype=torch.int32),
            2 * torch.arange(topk, 2 * topk, dtype=torch.int32),
        )
    ).unsqueeze(1)
    boundaries = torch.full(
        (2,),
        4 * topk,
        dtype=torch.int32,
    )
    row_requests = torch.zeros(2, dtype=torch.int32)
    block_table = torch.arange(32, dtype=torch.int32).reshape(1, 32)
    expected = _prepare_sparse_indices_torch(
        source,
        boundaries,
        row_req_indices=row_requests,
        request_block_table=block_table,
        block_size=128,
    )

    actual = _run_production_staged(
        source,
        boundaries,
        request_count=1,
        mtp=2,
        use_sharded_sort=True,
    )

    assert torch.equal(actual[0], expected[0])
    assert actual[2].tolist() == [2 * topk]
    assert torch.equal(actual[1][0, : 2 * topk], expected[1][0])
    assert torch.equal(actual[3][0, : 2 * topk], expected[3][0])


@pytest.mark.parametrize("capture_graph", [False, True])
@pytest.mark.parametrize(
    "use_sharded_sort",
    [True, False],
    ids=["sharded-operator", "legacy-staged-operator"],
)
def test_production_mtp1_backends_skip_union_and_preserve_source_order(
    capture_graph,
    use_sharded_sort,
):
    request_count = 2
    topk = 2048
    rows = torch.stack(
        (
            torch.roll(torch.arange(topk, dtype=torch.int32), 197),
            torch.roll(
                torch.arange(4096, 4096 + topk, dtype=torch.int32),
                331,
            ),
        )
    ).unsqueeze(1)
    boundaries = torch.tensor(
        [int(row.max()) - 100 for row in rows.reshape(request_count, topk)],
        dtype=torch.int32,
    )
    actual = _run_production_staged(
        rows,
        boundaries,
        request_count,
        1,
        capture_graph=capture_graph,
        use_sharded_sort=use_sharded_sort,
    )

    expected_remapped = rows.reshape(request_count, topk).clone()
    expected_selected = []
    for request in range(request_count):
        row = rows[request].reshape(-1)
        mask = (row >= 0) & (row < boundaries[request])
        selected = row[mask]
        expected_selected.append(selected)
        expected_remapped[request, mask] = torch.arange(
            selected.numel(),
            dtype=torch.int32,
        )
        count = selected.numel()
        assert actual[2][request].item() == count
        assert torch.equal(actual[1][request, :count], selected)
        logical = torch.arange(count, dtype=torch.long)
        physical = actual[4][request].to(torch.long)
        expected_targets = (
            physical[logical // 128] * 128 + logical % 128
        )
        assert torch.equal(actual[3][request, :count], expected_targets)
    assert torch.equal(
        actual[0].reshape(request_count, topk),
        expected_remapped,
    )


def test_production_staged_graph_padding_rows_are_zeroed():
    topk = 2048
    source = torch.stack(
        (
            torch.arange(topk, dtype=torch.int32),
            torch.arange(topk, dtype=torch.int32) + 4096,
        )
    ).unsqueeze(1)
    values = source.npu()
    boundaries = torch.tensor(
        [topk - 100, 0],
        dtype=torch.int32,
        device="npu",
    )
    row_requests = torch.tensor([0, -1], dtype=torch.int32, device="npu")
    selected, counts, targets = _buffers(2, topk)
    mapping = torch.empty_like(selected)
    block_table = torch.arange(
        2 * (topk // 128),
        dtype=torch.int32,
        device="npu",
    ).reshape(2, topk // 128)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_staged_(
            values,
            boundaries,
            row_requests,
            block_table,
            selected,
            counts,
            targets,
            mapping,
            128,
            1,
            True,
            True,
        )
    values.copy_(source.npu())
    graph.replay()
    torch.npu.synchronize()

    assert torch.count_nonzero(values[1]).item() == 0
    assert counts[1, 0].item() == 0


def test_production_staged_remaps_without_building_optional_payload():
    topk = 2048
    source = torch.stack(
        (
            torch.arange(topk, dtype=torch.int32),
            torch.arange(1024, 1024 + topk, dtype=torch.int32),
        )
    ).unsqueeze(1)
    boundaries = torch.tensor(
        [topk - 100, 1024 + topk - 100],
        dtype=torch.int32,
    )
    selected, counts, targets = _buffers(1, 2 * topk)
    targets.fill_(-7)
    workspace = torch.empty_like(selected)
    shard_packed = torch.empty(
        (1, 2, 2 * topk),
        dtype=torch.int32,
        device="npu",
    )
    shard_mapping = torch.empty_like(shard_packed)
    shard_counts = torch.empty(
        (1, 2, 16),
        dtype=torch.int32,
        device="npu",
    )
    actual = prepare_sparse_indices(
        source.npu(),
        boundaries.npu(),
        row_req_indices=torch.tensor(
            [0, 0],
            dtype=torch.int32,
            device="npu",
        ),
        request_block_table=torch.arange(
            32,
            dtype=torch.int32,
            device="npu",
        ).reshape(1, 32),
        selected_packed=selected,
        selected_counts=counts,
        target_slot_mapping=targets,
        block_size=128,
        need_packed=False,
        local_to_union_workspace=workspace,
        shard_packed_workspace=shard_packed,
        shard_mapping_workspace=shard_mapping,
        shard_counts_workspace=shard_counts,
        staged_mtp=2,
    )
    torch.npu.synchronize()

    remapped = actual[0].cpu().reshape(2, topk)
    source_rows = source.reshape(2, topk)
    selected_mask = (
        (source_rows >= 0)
        & (source_rows < boundaries.reshape(-1, 1))
    )
    assert torch.equal(remapped[~selected_mask], source_rows[~selected_mask])

    # The sharded operator does not promise a global union order. Validate the
    # actual contract without relying on either shard ordering: equal tokens
    # share one rank, unequal tokens never share a rank, and all union ranks
    # form a dense [0, unique_count) range.
    token_to_rank = {}
    rank_to_token = {}
    for token, rank in zip(
        source_rows[selected_mask].tolist(),
        remapped[selected_mask].tolist(),
    ):
        if token in token_to_rank:
            assert token_to_rank[token] == rank
        else:
            token_to_rank[token] = rank
        if rank in rank_to_token:
            assert rank_to_token[rank] == token
        else:
            rank_to_token[rank] = token
    assert sorted(rank_to_token) == list(range(len(token_to_rank)))
    assert actual[1:] == (None, None, None)
    assert counts[0, 0].item() == 0
    assert torch.all(targets == -7)


@pytest.mark.parametrize("mtp", [1, 2])
@pytest.mark.parametrize(
    "parallel_map",
    [False, True],
    ids=["all-torch", "hybrid-aiv"],
)
def test_resident_sparse_cache_native_ops_and_graph_replay(
    mtp,
    parallel_map,
):
    requests = 2
    # Match the production kernel shape. In particular, this makes the
    # parallel-map graph test exercise the full 2048-element source row.
    topk = 2048
    capacity = mtp * topk
    union_count = capacity - 8
    token_stride = 8192
    block_size = 16
    state_rows = 2 * requests

    union_seed = torch.stack(
        [
            torch.arange(
                request * 128,
                request * 128 + capacity,
                dtype=torch.int32,
            )
            for request in range(requests)
        ]
    ).npu()
    selected = union_seed.clone()
    counts = torch.zeros(
        (requests, 16), dtype=torch.int32, device="npu"
    )
    counts[:, 0] = union_count
    count_seed = counts.clone()
    mapping = torch.full(
        (requests, capacity), -1, dtype=torch.int32, device="npu"
    )
    mapping[:, :union_count] = torch.arange(
        union_count, dtype=torch.int32, device="npu"
    )
    topk_seed = torch.arange(
        requests * capacity, dtype=torch.int32, device="npu"
    ).reshape(requests * mtp, 1, topk)
    topk_values = topk_seed.clone()
    targets = torch.empty(
        (requests, capacity), dtype=torch.long, device="npu"
    )
    block_table = torch.arange(
        requests * (capacity // block_size),
        dtype=torch.int32,
        device="npu",
    ).reshape(requests, capacity // block_size)
    states = torch.arange(requests, dtype=torch.int32, device="npu")
    request_generations = torch.ones(
        requests, dtype=torch.int64, device="npu"
    )
    token_to_slot = torch.full(
        (state_rows, token_stride),
        -1,
        dtype=torch.int16,
        device="npu",
    )
    slot_stride = ((capacity + 1 + 15) // 16) * 16
    slot_to_token = torch.full(
        (state_rows, slot_stride),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    state_generations = torch.full(
        (state_rows, 8), -1, dtype=torch.int64, device="npu"
    )
    workspace = allocate_resident_workspace(
        requests, capacity, device=torch.device("npu")
    )

    def invoke():
        prepare_resident_sparse_cache_(
            topk_values,
            mapping,
            selected,
            counts,
            targets,
            block_table,
            states,
            request_generations,
            token_to_slot,
            slot_to_token,
            state_generations,
            workspace,
            block_size=block_size,
            scratch_capacity=capacity,
            parallel_map=parallel_map,
        )

    persistent_addresses = (
        token_to_slot.data_ptr(),
        slot_to_token.data_ptr(),
        state_generations.data_ptr(),
    )
    workspace_addresses = tuple(
        value.data_ptr()
        for value in vars(workspace).values()
        if isinstance(value, torch.Tensor)
    )
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        invoke()

    # Cold replay: every union token is a miss.
    token_to_slot.fill_(-1)
    slot_to_token.fill_(-1)
    state_generations.fill_(-1)
    selected.copy_(union_seed)
    counts.copy_(count_seed)
    topk_values.copy_(topk_seed)
    graph.replay()
    torch.npu.synchronize()
    assert counts[:, 0].cpu().tolist() == [union_count] * requests
    assert torch.equal(
        selected[:, :union_count].cpu(),
        union_seed[:, :union_count].cpu(),
    )

    # Warm replay: all tokens hit, so LMCache receives an empty payload while
    # the original positions still map to their resident slots.
    selected.copy_(union_seed)
    counts.copy_(count_seed)
    topk_values.copy_(topk_seed)
    graph.replay()
    torch.npu.synchronize()
    assert counts[:, 0].cpu().tolist() == [0] * requests
    assert torch.equal(
        topk_values.reshape(requests, capacity)[:, :union_count].cpu(),
        torch.arange(union_count, dtype=torch.int32)
        .expand(requests, -1),
    )
    assert persistent_addresses == (
        token_to_slot.data_ptr(),
        slot_to_token.data_ptr(),
        state_generations.data_ptr(),
    )
    assert workspace_addresses == tuple(
        value.data_ptr()
        for value in vars(workspace).values()
        if isinstance(value, torch.Tensor)
    )


@pytest.mark.parametrize("mtp", [1, 2])
@pytest.mark.parametrize("request_count", [1, 4])
@pytest.mark.parametrize("capture_graph", [False, True])
def test_resident_parallel_map_matches_tensor_reference(
    mtp,
    request_count,
    capture_graph,
):
    topk = 2048
    capacity = mtp * topk
    row_count = request_count * mtp
    source = (
        torch.arange(row_count * topk, dtype=torch.int32)
        .reshape(row_count, 1, topk)
        + 50000
    )
    positions = torch.arange(capacity, dtype=torch.int32).expand(
        request_count, -1
    )
    selected_mask = positions.remainder(3) != 0
    # Multiplication by an odd number is a permutation modulo a power-of-two
    # capacity. This exercises non-monotonic union ranks.
    ranks = positions.mul(17).remainder(capacity)
    position_to_union = torch.where(
        selected_mask,
        ranks,
        torch.full_like(ranks, -1),
    )
    union_to_slot = torch.arange(
        capacity - 1,
        -1,
        -1,
        dtype=torch.int32,
    ).expand(request_count, -1)
    union_to_slot = (
        union_to_slot
        + torch.arange(request_count, dtype=torch.int32).reshape(-1, 1)
        * capacity
    )
    expected = source.reshape(request_count, capacity).clone()
    expected.copy_(
        torch.where(
            selected_mask,
            torch.gather(union_to_slot, 1, ranks.to(torch.long)),
            expected,
        )
    )

    values = source.npu()
    mapping_npu = position_to_union.npu()
    slots_npu = union_to_slot.npu()

    def invoke():
        torch.ops._C_ascend.npu_dsa_resident_remap_rows_(
            values,
            mapping_npu,
            slots_npu,
        )

    if capture_graph:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            invoke()
        values.copy_(source.npu())
        mapping_npu.copy_(position_to_union.npu())
        slots_npu.copy_(union_to_slot.npu())
        graph.replay()
    else:
        invoke()
    torch.npu.synchronize()
    assert torch.equal(
        values.cpu().reshape(request_count, capacity),
        expected,
    )


def test_resident_hybrid_lookup_rows_uses_private_padding_sentinels():
    requests = 2
    capacity = 256
    token_stride = 512
    dummy_state_base = requests
    selected = torch.arange(
        requests * capacity, dtype=torch.int32, device="npu"
    ).reshape(requests, capacity)
    counts = torch.zeros(
        (requests, 16), dtype=torch.int32, device="npu"
    )
    counts[:, 0] = torch.tensor([3, 1], dtype=torch.int32, device="npu")
    # The second request is graph padding and must use its own dummy row.
    states = torch.tensor([1, -1], dtype=torch.int32, device="npu")
    indices = torch.empty(
        (requests, capacity), dtype=torch.int64, device="npu"
    )

    torch.ops._C_ascend.npu_dsa_resident_lookup_rows_(
        selected,
        counts,
        states,
        indices,
        token_stride,
        dummy_state_base,
    )
    torch.npu.synchronize()

    expected = torch.empty((requests, capacity), dtype=torch.int64)
    expected[0].fill_(2 * token_stride - 1)
    expected[0, :3] = (
        selected[0, :3].cpu().to(torch.int64) + token_stride
    )
    dummy_state = dummy_state_base + 1
    expected[1].fill_((dummy_state + 1) * token_stride - 1)
    expected[1, 0] = (
        selected[1, 0].cpu().to(torch.int64)
        + dummy_state * token_stride
    )
    assert torch.equal(indices.cpu(), expected)


def _resident_production_fixture(mtp, request_count=2):
    topk = 2048
    capacity = mtp * topk
    row_count = request_count * mtp
    shared = torch.arange(topk // 2, dtype=torch.int32)
    request_rows = torch.stack(
        [
            torch.cat(
                (
                    shared,
                    torch.arange(
                        topk // 2 + row * topk // 2,
                        topk // 2 + (row + 1) * topk // 2,
                        dtype=torch.int32,
                    ),
                )
            )
            for row in range(mtp)
        ]
    )
    source = request_rows.repeat(request_count, 1).unsqueeze(1)
    split_boundary = int(source.max()) - 100
    boundaries = torch.full(
        (row_count,), split_boundary, dtype=torch.int32, device="npu"
    )
    row_requests = torch.arange(
        request_count, dtype=torch.int32, device="npu"
    ).repeat_interleave(mtp)
    block_size = 128
    max_tokens = 131072
    block_table = torch.arange(
        request_count * max_tokens // block_size,
        dtype=torch.int32,
        device="npu",
    ).reshape(request_count, max_tokens // block_size)
    values = source.npu()
    selected, counts, targets = _buffers(request_count, capacity)
    mapping = torch.empty_like(selected)
    shard_packed = torch.empty(
        (request_count, 2, capacity),
        dtype=torch.int32,
        device="npu",
    )
    shard_mapping = torch.empty_like(shard_packed)
    shard_counts = torch.empty(
        (request_count, 2, 16),
        dtype=torch.int32,
        device="npu",
    )
    token_stride = 131104
    slot_stride = ((capacity + 16) // 16) * 16
    token_to_slot = torch.full(
        (2 * request_count, token_stride),
        -1,
        dtype=torch.int16,
        device="npu",
    )
    slot_to_token = torch.full(
        (2 * request_count, slot_stride),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    state_generations = torch.ones(
        (2 * request_count, 8),
        dtype=torch.int64,
        device="npu",
    )
    states = torch.arange(
        request_count, dtype=torch.int32, device="npu"
    )
    request_generations = torch.ones(
        request_count, dtype=torch.int64, device="npu"
    )
    workspace = allocate_resident_workspace(
        request_count, capacity, device=torch.device("npu")
    )

    def union():
        values.copy_(source.npu())
        prepare_sparse_indices(
            values,
            boundaries,
            row_req_indices=row_requests,
            request_block_table=block_table,
            selected_packed=selected,
            selected_counts=counts,
            target_slot_mapping=targets,
            block_size=block_size,
            need_packed=True,
            clear_invalid_rows=True,
            local_to_union_workspace=mapping,
            shard_packed_workspace=shard_packed,
            shard_mapping_workspace=shard_mapping,
            shard_counts_workspace=shard_counts,
            staged_mtp=mtp,
        )

    def resident(use_hybrid_aiv=True):
        prepare_resident_sparse_cache_(
            values,
            mapping,
            selected,
            counts,
            targets,
            block_table,
            states,
            request_generations,
            token_to_slot,
            slot_to_token,
            state_generations,
            workspace,
            block_size=block_size,
            scratch_capacity=capacity,
            parallel_map=use_hybrid_aiv,
        )

    return {
        "topk": topk,
        "capacity": capacity,
        "source": source,
        "boundaries": boundaries,
        "values": values,
        "selected": selected,
        "counts": counts,
        "targets": targets,
        "mapping": mapping,
        "block_table": block_table,
        "token_to_slot": token_to_slot,
        "slot_to_token": slot_to_token,
        "state_generations": state_generations,
        "request_generations": request_generations,
        "union": union,
        "resident": resident,
    }


@pytest.mark.parametrize("mtp", [1, 2])
def test_resident_hybrid_matches_all_torch_for_partial_hits(mtp):
    reference = _resident_production_fixture(mtp)
    hybrid = _resident_production_fixture(mtp)
    fixtures = (reference, hybrid)
    for fixture in fixtures:
        fixture["union"]()
    torch.npu.synchronize()

    union_count = int(reference["counts"][0, 0].item())
    hit_count = union_count // 2
    request_count = reference["selected"].shape[0]
    capacity = reference["capacity"]
    hit_slots = torch.arange(
        capacity - hit_count,
        capacity,
        dtype=torch.int16,
        device="npu",
    ).expand(request_count, -1)
    request_rows = torch.arange(
        request_count, dtype=torch.long, device="npu"
    ).reshape(-1, 1)
    for fixture in fixtures:
        hit_tokens = fixture["selected"][:, :hit_count].to(torch.long)
        fixture["token_to_slot"][
            request_rows, hit_tokens
        ] = hit_slots
        fixture["slot_to_token"][
            :request_count, capacity - hit_count : capacity
        ].copy_(fixture["selected"][:, :hit_count])

    reference["resident"](False)
    hybrid["resident"](True)
    torch.npu.synchronize()
    miss_count = union_count - hit_count
    for name in (
        "values",
        "token_to_slot",
        "slot_to_token",
        "state_generations",
    ):
        assert torch.equal(reference[name].cpu(), hybrid[name].cpu()), name
    assert torch.equal(
        reference["counts"][:, 0].cpu(),
        hybrid["counts"][:, 0].cpu(),
    )
    assert torch.equal(
        reference["selected"][:, :miss_count].cpu(),
        hybrid["selected"][:, :miss_count].cpu(),
    )
    assert torch.equal(
        reference["targets"][:, :miss_count].cpu(),
        hybrid["targets"][:, :miss_count].cpu(),
    )


@pytest.mark.parametrize("mtp", [1, 2])
def test_resident_production_pipeline_partial_hits_all_hits_and_generation(
    mtp,
):
    fixture = _resident_production_fixture(mtp)
    request_count = fixture["selected"].shape[0]
    capacity = fixture["capacity"]
    fixture["union"]()
    torch.npu.synchronize()
    union_seed = fixture["selected"].clone()
    count_seed = fixture["counts"].clone()
    union_count = int(count_seed[0, 0].item())
    assert fixture["counts"][:, 0].cpu().tolist() == [
        union_count
    ] * request_count

    # Put half the union in high-numbered scratch slots. Misses must use the
    # lowest unprotected slots, while hits retain their existing slots.
    hit_count = union_count // 2
    hit_slots = torch.arange(
        capacity - hit_count,
        capacity,
        dtype=torch.int16,
        device="npu",
    ).expand(request_count, -1)
    hit_tokens = union_seed[:, :hit_count].to(torch.long)
    request_rows = torch.arange(
        request_count, dtype=torch.long, device="npu"
    ).reshape(-1, 1)
    fixture["token_to_slot"][request_rows, hit_tokens] = hit_slots
    fixture["slot_to_token"][
        :request_count, capacity - hit_count : capacity
    ].copy_(union_seed[:, :hit_count])

    fixture["resident"]()
    torch.npu.synchronize()
    miss_count = union_count - hit_count
    assert fixture["counts"][:, 0].cpu().tolist() == [
        miss_count
    ] * request_count
    assert torch.equal(
        fixture["selected"][:, :miss_count].cpu(),
        union_seed[:, hit_count:union_count].cpu(),
    )
    for request in range(request_count):
        assert torch.equal(
            fixture["targets"][request, :miss_count].cpu(),
            torch.arange(miss_count, dtype=torch.long)
            + request * 131072,
        )

    source_rows = fixture["source"].reshape(request_count, capacity)
    remapped = fixture["values"].reshape(request_count, capacity)
    boundary_rows = fixture["boundaries"].cpu().reshape(
        request_count, mtp, 1
    )
    selected_mask = (
        fixture["source"].reshape(request_count, mtp, -1)
        < boundary_rows
    ).reshape(request_count, capacity)
    safe_slots = torch.where(
        selected_mask.npu(),
        remapped,
        torch.zeros_like(remapped),
    )
    reconstructed = torch.gather(
        fixture["slot_to_token"][:request_count, :capacity],
        1,
        safe_slots.to(torch.long),
    )
    assert torch.equal(
        reconstructed[selected_mask.npu()].cpu(),
        source_rows[selected_mask].cpu(),
    )
    assert torch.equal(
        remapped[~selected_mask.npu()].cpu(),
        source_rows[~selected_mask].cpu(),
    )

    # The next identical step is entirely resident.
    fixture["union"]()
    fixture["resident"]()
    torch.npu.synchronize()
    assert fixture["counts"][:, 0].cpu().tolist() == [0] * request_count

    # Reusing the state row with a new generation invalidates every old hit.
    fixture["request_generations"].fill_(2)
    fixture["union"]()
    fixture["resident"]()
    torch.npu.synchronize()
    assert fixture["counts"][:, 0].cpu().tolist() == [
        union_count
    ] * request_count


@pytest.mark.parametrize("mtp", [1, 2])
def test_resident_production_pipeline_zero_split_boundary_is_noop(mtp):
    fixture = _resident_production_fixture(mtp)
    fixture["boundaries"].zero_()
    fixture["union"]()
    fixture["resident"]()
    torch.npu.synchronize()

    assert fixture["counts"][:, 0].cpu().tolist() == [
        0
    ] * fixture["selected"].shape[0]
    assert torch.equal(fixture["values"].cpu(), fixture["source"])
