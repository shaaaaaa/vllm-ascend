# SPDX-License-Identifier: Apache-2.0
"""CPU regression for actual MTP position reduction and diagnostic row labels.

Extract the production proposer and padding functions without importing the
NPU runtime. Only the collective is a CPU stand-in: identical rank inputs are
summed, then partitioned, just as reduce_scatter's default SUM requires.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]


def source_function(path, name, namespace, *, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if class_name is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setitem(sys.modules, "layerwise_prefill_file_store", NS(install=lambda: None))
    path = ROOT / "tools" / "layerwise_prefill_file_worker.py"
    spec = importlib.util.spec_from_file_location("file_positions_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_production_position_reduction(source, world_size, rank, *, flash_comm=True):
    extra = NS(flash_comm_v1_enabled=flash_comm, pad_size=(-len(source)) % world_size)
    calls = []

    def reduce_scatter(value, dim):
        assert dim == 0
        calls.append(value.clone())
        # All TP ranks enter with the same position IDs. This is SUM, not
        # an all-gather, a mean, or a simple local slice of the input.
        summed = torch.stack([value] * world_size).sum(dim=0, dtype=value.dtype)
        return summed.chunk(world_size, dim=0)[rank].clone()

    pad_reduce = source_function(
        ROOT / "vllm_ascend" / "ops" / "register_custom_ops.py",
        "_maybe_pad_and_reduce_impl",
        {
            "torch": torch,
            "F": F,
            "_EXTRA_CTX": extra,
            "get_forward_context": lambda: NS(flash_comm_v1_enabled=True, is_draft_model=True, dp_metadata=None),
            "enable_sp_by_pass": lambda: False,
            "is_vl_model": lambda: False,
            "tensor_model_parallel_reduce_scatter": reduce_scatter,
        },
    )
    proposer_reduce = source_function(
        ROOT / "vllm_ascend" / "spec_decode" / "eagle_proposer.py",
        "maybe_pad_and_reduce",
        {
            "torch": NS(Tensor=torch.Tensor, ops=NS(vllm=NS(maybe_pad_and_reduce=pad_reduce))),
            "_EXTRA_CTX": extra,
        },
        class_name="SpecDecodeBaseProposer",
    )
    hidden = torch.arange(len(source) * 3, dtype=torch.float32).reshape(len(source), 3)
    _, positions = proposer_reduce(NS(is_multimodal_model=False, method="mtp"), hidden, source)
    assert len(calls) == (2 if flash_comm else 0)
    return positions


def metadata(actual, world_size, rank):
    capacity = (actual + world_size - 1) // world_size
    start = rank * capacity
    return NS(
        num_actual_tokens=actual,
        # Deliberately different from the observed proposer buffer. CPU length
        # arithmetic is not the authority for what the model actually received.
        seq_lens_cpu=[actual + 999],
        query_start_loc_cpu=[0, actual],
        dsa_cp_context=NS(
            local_start=start,
            local_end=min(start + capacity, actual),
            local_end_with_pad=start + capacity,
        ),
    )


@pytest.mark.parametrize("world_size", [4, 8])
@pytest.mark.parametrize("actual", [1, 5, 9])
def test_actual_mtp_sum_shards_keep_global_labels_and_source_buffer(worker, world_size, actual):
    source = torch.arange(100, 100 + actual, dtype=torch.int64)
    unchanged_source = source.clone()
    token_ids = list(range(20, 20 + actual))
    covered = []
    for rank in range(world_size):
        raw = run_production_position_reduction(source, world_size, rank)
        unchanged_raw = raw.clone()
        meta = metadata(actual, world_size, rank)
        start, end = worker.row_window(len(raw), actual, meta.dsa_cp_context, prefer_local=True)
        logical, inputs = worker.query_context(
            meta,
            raw.tolist(),
            token_ids,
            positions_local=True,
            source_positions=source.tolist(),
            position_scale=world_size,
        )
        assert logical == source.tolist()
        assert inputs == token_ids
        assert raw[: end - start].tolist() == [position * world_size for position in logical[start:end]]
        assert torch.equal(raw, unchanged_raw)
        assert torch.equal(source, unchanged_source)
        covered.extend(logical[start:end])
    assert covered == source.tolist()


@pytest.mark.parametrize("world_size", [4, 8])
def test_real_sum_positions_reject_wrong_scale_and_corrupt_valid_rows(worker, world_size):
    source = torch.arange(100, 105, dtype=torch.int64)
    raw = run_production_position_reduction(source, world_size, 0).tolist()
    meta = metadata(len(source), world_size, 0)
    kwargs = {"positions_local": True, "source_positions": source.tolist()}
    with pytest.raises(RuntimeError, match="disagree"):
        worker.query_context(meta, raw, [1] * len(source), position_scale=1, **kwargs)
    raw[0] += 1
    with pytest.raises(RuntimeError, match="disagree"):
        worker.query_context(meta, raw, [1] * len(source), position_scale=world_size, **kwargs)


def test_padding_only_rank_has_no_logical_position_and_source_is_required(worker):
    source = torch.tensor([100])
    raw = run_production_position_reduction(source, 8, 7)
    assert raw.tolist() == [0]
    meta = metadata(1, 8, 7)
    assert worker.query_context(
        meta, raw.tolist(), [999], positions_local=True, source_positions=[100], position_scale=8
    ) == ([100], [999])
    # Padding has no logical owner, even if its raw bytes happen to be nonzero.
    assert worker.query_context(
        meta, [54321], [999], positions_local=True, source_positions=[100], position_scale=8
    ) == ([100], [999])
    with pytest.raises(RuntimeError, match="require the observed"):
        worker.query_context(meta, raw.tolist(), [999], positions_local=True, position_scale=8)
    with pytest.raises(RuntimeError, match="require the observed"):
        worker.query_context(meta, raw.tolist(), [999], positions_local=True, source_positions=[], position_scale=8)


def test_flash_comm_off_retains_global_raw_positions(worker):
    source = torch.arange(100, 105, dtype=torch.int64)
    raw = run_production_position_reduction(source, 8, 0, flash_comm=False)
    assert raw.data_ptr() == source.data_ptr()
    assert worker.query_context(metadata(5, 8, 0), raw.tolist(), [1, 2, 3, 4, 5], source_positions=source.tolist()) == (
        source.tolist(),
        [1, 2, 3, 4, 5],
    )
    with pytest.raises(RuntimeError, match="disagree"):
        worker.query_context(metadata(5, 8, 0), raw.tolist(), [1, 2, 3, 4, 5], source_positions=[9] * 5)
