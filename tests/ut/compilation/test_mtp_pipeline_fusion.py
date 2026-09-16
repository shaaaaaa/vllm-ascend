# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise actual specialized and general MTP preparation without NPU imports."""

import ast
from copy import copy
from types import MethodType
from types import SimpleNamespace as NS

import pytest
import torch
from sfa_test_support import ROOT, HostTL, Pointer, extract, load_module

PROPOSER = ROOT / "vllm_ascend/spec_decode/eagle_proposer.py"
KERNELS = ROOT / "vllm_ascend/ops/triton/spec_decode/utils.py"


class Kernel:
    def __init__(self, name, calls):
        self.tl = HostTL()
        load = self.tl.load
        self.tl.load = lambda p, mask, other=0: load(p, mask, other)
        self.body = extract(KERNELS, name, {"tl": self.tl})
        self.name, self.calls = name, calls

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append(self.name)
            convert = lambda x: Pointer(x) if isinstance(x, torch.Tensor) else x
            self.tl.programs = grid[0]
            for pid in range(grid[0]):
                self.tl.pid = pid
                self.body(*(convert(x) for x in args), **{k: convert(v) for k, v in kwargs.items()})

        return launch


def owner(n, width=2, position_dtype=torch.int64):
    calls = []
    ns = dict(
        torch=torch,
        HAS_TRITON=True,
        AscendCommonAttentionMetadata=NS,
        triton=NS(cdiv=lambda n, d: (n + d - 1) // d),
        get_vectorcore_num=lambda: 4,
        _PREPARE_INPUTS_BLOCK_SIZE=4,
    )
    for name in ("prepare_next_mtp_tokens_kernel", "prepare_inputs_padded_kernel", "pack_mtp_tokens_positions_kernel"):
        ns[name] = Kernel(name, calls)
    runner = NS(
        _fixed_mtp_metadata=(torch.arange(1, n + 1, dtype=torch.int32),),
        actual_seq_lengths_q=list(range(2, 2 * n + 1, 2)),
        attn_state="spec",
        decode_token_per_req=2,
    )
    obj = NS(
        runner=runner,
        method="mtp",
        num_speculative_tokens=1,
        use_async_scheduling=True,
        _fixed_mtp_pipeline_ready=True,
        needs_extra_input_slots=False,
        pcp_size=1,
        dcp_size=1,
        uses_mrope=False,
        uses_xdrope_dim=0,
        vllm_config=NS(model_config=NS(uses_mrope=False)),
        token_indices_to_sample=torch.full((n + 5,), -77, dtype=torch.int32),
        arange=torch.arange(2 * n + 10, dtype=torch.int32),
        input_ids=torch.full((2 * n + 5,), -99, dtype=torch.int32),
        positions=torch.full((2 * n + 5,), -99, dtype=torch.int32),
        hidden_states=torch.full((2 * n + 5, 8), -99.0),
    )
    cpu = torch.zeros(n + 3, dtype=torch.int32)
    obj.backup_next_token_ids = NS(np=cpu.numpy(), gpu=torch.zeros_like(cpu))
    obj.backup_next_token_ids.copy_to_gpu = lambda count: obj.backup_next_token_ids.gpu[:count].copy_(cpu[:count])
    for name in (
        "prepare_next_token_ids_padded",
        "prepare_inputs_padded",
        "set_inputs_first_pass",
        "warmup_next_mtp_tokens",
    ):
        setattr(obj, name, MethodType(extract(PROPOSER, name, ns), obj))
    obj._set_positions = MethodType(
        extract(ROOT.parent / "vllm/vllm/v1/spec_decode/eagle.py", "_set_positions", {}), obj
    )
    common = NS(
        num_reqs=n,
        num_actual_tokens=2 * n,
        max_query_len=2,
        num_input_tokens=2 * n,
        query_start_loc=torch.arange(n + 1, dtype=torch.int32) * 2,
        query_start_loc_cpu=torch.arange(n + 1, dtype=torch.int32) * 2,
        seq_lens=torch.arange(n, dtype=torch.int32) + 40,
        seq_lens_cpu=torch.arange(n, dtype=torch.int32) + 40,
        num_computed_tokens_cpu=torch.arange(n, dtype=torch.int32) + 38,
        block_table_tensor=torch.arange(n * 4, dtype=torch.int32).view(n, 4),
        slot_mapping=torch.arange(2 * n, dtype=torch.int32),
        indexer_block_table_tensor=torch.arange(n * 4, dtype=torch.int32).view(n, 4) + 100,
        indexer_slot_mapping=torch.arange(2 * n, dtype=torch.int32) + 200,
        positions=torch.arange(2 * n, dtype=position_dtype) + 38,
    )
    spec = NS(cu_num_draft_tokens=runner._fixed_mtp_metadata[0][:n])
    batch = NS(num_reqs=n, req_ids=[str(i) for i in range(n)], vocab_size=100)
    requests = {rid: NS(get_token_id=lambda index: 13) for rid in batch.req_ids}
    patterns = torch.tensor([[7, 8], [7, -1], [-1, -1], [-1, 8], [100, 8], [-2, 8], [99, 99]], dtype=torch.int32)
    sampled = patterns[torch.arange(n) % len(patterns), :width].clone()
    return obj, common, spec, batch, requests, sampled, calls, ns


def prepare(case, fused=True):
    obj, common, spec, batch, requests, sampled, calls, ns = case
    next_ids, counts = obj.prepare_next_token_ids_padded(
        common, sampled, requests, batch, torch.empty(0, dtype=torch.int64), 0, spec if fused else None
    )
    metadata, indices, sample_indices, rejected = obj.prepare_inputs_padded(common, spec, counts)
    return next_ids, counts, metadata, indices, sample_indices, rejected


@pytest.mark.parametrize("n", [1, 3, 16, 33])
@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_fused_preparation_and_packing_match_general_path(n, width, position_dtype):
    new, old = owner(n, width, position_dtype), owner(n, width, position_dtype)
    actual, expected = prepare(new), prepare(old, False)
    for i in (0, 1, 3, 4, 5):
        torch.testing.assert_close(actual[i], expected[i])
    for field, a in vars(actual[2]).items():
        b = getattr(expected[2], field)
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b)
        else:
            assert a == b
    assert "prepare_inputs_padded_kernel" not in new[6]
    assert actual[4].data_ptr() == new[0].token_indices_to_sample.data_ptr()
    assert new[0]._prepared_mtp_inputs is None
    for case, values in ((new, actual), (old, expected)):
        obj = case[0]
        obj.set_inputs_first_pass(
            target_token_ids=torch.arange(2 * n, dtype=torch.int32) + 5,
            next_token_ids=values[0],
            target_positions=case[1].positions,
            target_hidden_states=torch.arange(16 * n).float().view(2 * n, 8),
            token_indices_to_sample=values[4],
            cad=values[2],
            num_rejected_tokens_gpu=values[5],
        )
    for field in ("input_ids", "positions", "hidden_states"):
        torch.testing.assert_close(getattr(new[0], field), getattr(old[0], field))
    assert new[6].count("pack_mtp_tokens_positions_kernel") == 1
    assert "pack_mtp_tokens_positions_kernel" not in old[6]


@pytest.mark.parametrize(
    "change",
    ["not_warmed", "async", "cp", "extra_slots", "width", "padded_rows", "foreign_drafts", "strided_sample", "discard"],
)
def test_unproven_layout_does_not_fuse(change):
    obj, common, spec, batch, requests, sampled, calls, ns = owner(3)
    discarded = 0
    if change == "not_warmed":
        obj._fixed_mtp_pipeline_ready = False
    elif change == "async":
        obj.use_async_scheduling = False
    elif change == "cp":
        obj.pcp_size = 2
    elif change == "extra_slots":
        obj.needs_extra_input_slots = True
    elif change == "width":
        common.max_query_len = 1
    elif change == "padded_rows":
        common.query_start_loc_cpu = torch.arange(5) * 2
    elif change == "foreign_drafts":
        spec.cu_num_draft_tokens = spec.cu_num_draft_tokens.clone()
    elif change == "strided_sample":
        storage = torch.empty(3, 4, dtype=sampled.dtype)
        storage[:, ::2] = sampled
        sampled = storage[:, ::2]
    else:
        discarded = 1
    obj.prepare_next_token_ids_padded(common, sampled, requests, batch, torch.tensor([0]), discarded, spec)
    assert obj._prepared_mtp_inputs is None


@pytest.mark.parametrize("replaced", ["counts", "common", "spec"])
def test_fused_handoff_is_one_use_and_checks_all_owners(replaced):
    case = owner(3)
    obj, common, spec, batch, requests, sampled, calls, _ = case
    _, counts = obj.prepare_next_token_ids_padded(
        common, sampled, requests, batch, torch.empty(0, dtype=torch.int64), 0, spec
    )
    obj.prepare_inputs_padded(
        copy(common) if replaced == "common" else common,
        copy(spec) if replaced == "spec" else spec,
        counts.clone() if replaced == "counts" else counts,
    )
    assert calls[-1] == "prepare_inputs_padded_kernel" and obj._prepared_mtp_inputs is None


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("kind", ["identity", "copy", "strided", "offset", "padded", "noncontiguous_hidden", "pp"])
def test_hidden_state_view_is_only_used_for_owned_identity_layout(dtype, kind):
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    arrays = extract(path, "_fixed_mtp_metadata_arrays", {"torch": torch})(8, "cpu")
    obj = NS(_fixed_mtp_metadata=arrays, parallel_config=NS(pipeline_parallel_size=1))
    select = extract(path, "_select_sample_hidden_states", {"torch": torch})
    identity = arrays[5 if dtype == torch.int64 else 4]
    indices = identity[:6]
    hidden = torch.arange(16 * 12).float().view(16, 12)
    if kind == "copy":
        indices = indices.clone()
    elif kind == "strided":
        indices = identity[:12:2]
    elif kind == "offset":
        indices = identity[1:7]
    elif kind == "padded":
        indices = torch.nn.functional.pad(indices, (0, 2))
    elif kind == "noncontiguous_hidden":
        hidden = hidden[:, ::2]
    elif kind == "pp":
        obj.parallel_config.pipeline_parallel_size = 2
    actual = select(obj, hidden, indices)
    torch.testing.assert_close(actual, hidden[indices])
    assert (actual.data_ptr() == hidden.data_ptr()) == (kind == "identity")


def test_warmup_compiles_fixed_and_general_variants_without_touching_serving_buffers():
    obj, _, _, batch, _, _, calls, ns = owner(3)
    obj.runner.max_num_reqs, obj.runner.input_batch, obj.device = 3, batch, "cpu"
    before = [t.clone() for t in (obj.input_ids, obj.positions, obj.token_indices_to_sample)]
    obj.warmup_next_mtp_tokens()
    assert obj._fixed_mtp_pipeline_ready
    assert calls.count("prepare_next_mtp_tokens_kernel") == 4
    assert calls.count("pack_mtp_tokens_positions_kernel") == 2
    for actual, expected in zip((obj.input_ids, obj.positions, obj.token_indices_to_sample), before):
        torch.testing.assert_close(actual, expected)


def test_fixed_cpu_layout_avoids_scalar_readback_and_preserves_handoff_failure(monkeypatch):
    obj, common, spec, batch, requests, sampled, calls, _ = owner(3)
    next_ids, counts = obj.prepare_next_token_ids_padded(common, sampled, requests, batch, torch.empty(0), 0, spec)

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed CPU layout must not compute scalar lengths")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden)
        meta, _, _, _ = obj.prepare_inputs_padded(common, spec, counts)
    assert meta.max_query_len == 2 and meta.num_actual_tokens == 6
    assert calls == ["prepare_next_mtp_tokens_kernel"]
    obj._prepared_mtp_inputs = (common, spec, counts, object(), object())
    common.seq_lens_cpu = None
    with pytest.raises(TypeError):
        obj.prepare_next_token_ids_padded(common, sampled, requests, batch, torch.empty(0), 0, spec)
    assert obj._prepared_mtp_inputs is None


@pytest.mark.parametrize("kind", ["same", "copy", "strided", "offset", "dtype"])
def test_sampling_index_staging_skips_only_exact_prefix_alias(kind):
    # Execute the staging block from the real _propose method.
    tree = ast.parse(PROPOSER.read_text(encoding="utf8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_propose")
    start = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "token_indices_to_sample_len" for t in n.targets)
    )
    code = ast.parse("def stage(self, token_indices_to_sample): pass")
    code.body[0].body = method.body[start : start + 2]
    ns = {}
    exec(compile(ast.fix_missing_locations(code), str(PROPOSER), "exec"), ns)
    obj = NS(token_indices_to_sample=torch.arange(16, dtype=torch.int32))
    source = obj.token_indices_to_sample[:4]
    if kind == "copy":
        source = source.clone() + 10
    elif kind == "strided":
        source = obj.token_indices_to_sample[:8:2]
    elif kind == "offset":
        source = obj.token_indices_to_sample[4:8]
    elif kind == "dtype":
        source = obj.token_indices_to_sample.view(torch.int64)[:4]
    expected = obj.token_indices_to_sample.clone()
    expected[:4].copy_(source.clone())
    version = obj.token_indices_to_sample._version
    if kind == "dtype":
        # The original copy rejects this overlapping dtype reinterpretation.
        with pytest.raises(RuntimeError, match="single memory location"):
            ns["stage"](obj, source)
        return
    ns["stage"](obj, source)
    torch.testing.assert_close(obj.token_indices_to_sample, expected)
    assert (obj.token_indices_to_sample._version == version) == (kind == "same")


def test_npu_pipeline_matches_cpu_reference_and_preserves_queued_outputs(monkeypatch):
    pytest.importorskip("torch_npu")
    pytest.importorskip("triton")
    if not torch.npu.is_available():
        pytest.skip("Requires NPU")
    module = load_module(KERNELS, "mtp_pipeline_npu", monkeypatch)
    held = []
    for n in (1, 3, 17):
        for width in (1, 2):
            case, reference = owner(n, width), owner(n, width)
            obj, common, spec, batch, requests, sampled, _, ns = case
            for name in (
                "prepare_next_mtp_tokens_kernel",
                "prepare_inputs_padded_kernel",
                "pack_mtp_tokens_positions_kernel",
            ):
                ns[name] = getattr(module, name)
            for field in ("token_indices_to_sample", "arange", "input_ids", "positions", "hidden_states"):
                setattr(obj, field, getattr(obj, field).to("npu"))
            obj.backup_next_token_ids.gpu = obj.backup_next_token_ids.gpu.to("npu")
            for field in (
                "query_start_loc",
                "seq_lens",
                "block_table_tensor",
                "slot_mapping",
                "indexer_block_table_tensor",
                "indexer_slot_mapping",
                "positions",
            ):
                setattr(common, field, getattr(common, field).to("npu"))
            obj.runner._fixed_mtp_metadata = (obj.runner._fixed_mtp_metadata[0].to("npu"),)
            spec.cu_num_draft_tokens = obj.runner._fixed_mtp_metadata[0]
            sampled = sampled.to("npu")
            obj.runner.max_num_reqs, obj.runner.input_batch, obj.device = n, batch, torch.device("npu")
            obj.warmup_next_mtp_tokens()
            for _ in range(2):
                actual = prepare((obj, common, spec, batch, requests, sampled, [], ns))
                expected = prepare(reference, False)
                for i in (0, 1, 3, 4, 5):
                    held.append((actual[i].clone() if i in (3, 4) else actual[i], expected[i].clone()))
                for current, values, device in ((case, actual, "npu"), (reference, expected, "cpu")):
                    current[0].set_inputs_first_pass(
                        target_token_ids=torch.arange(2 * n, dtype=torch.int32, device=device) + 5,
                        next_token_ids=values[0],
                        target_positions=current[1].positions,
                        target_hidden_states=torch.arange(16 * n, device=device).float().view(2 * n, 8),
                        token_indices_to_sample=values[4],
                        cad=values[2],
                        num_rejected_tokens_gpu=values[5],
                    )
                for field in ("input_ids", "positions", "hidden_states"):
                    held.append((getattr(obj, field).clone(), getattr(reference[0], field).clone()))
                sampled.fill_(9)
                reference[5].fill_(9)
    torch.npu.synchronize()
    for actual, expected in held:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_failed_warmup_never_enables_fixed_pipeline():
    obj, _, _, batch, _, _, _, ns = owner(3)
    obj.runner.max_num_reqs, obj.runner.input_batch, obj.device = 3, batch, "cpu"

    class Fail:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                raise RuntimeError("compile failed")

            return launch

    ns["prepare_next_mtp_tokens_kernel"] = Fail()
    with pytest.raises(RuntimeError, match="compile failed"):
        obj.warmup_next_mtp_tokens()
    assert not obj._fixed_mtp_pipeline_ready


def test_actual_runner_passes_fixed_layout_through_count_readback_and_draft_packing():
    obj, common, spec, batch, requests, sampled, calls, ns = owner(3)
    runner = obj.runner
    config = NS(disable_padded_drafter_batch=False, use_eagle=lambda: True, uses_draft_model=lambda: False)
    runner.drafter, runner.speculative_config = obj, config
    runner.vllm_config = NS(speculative_config=config)
    runner.input_batch, runner.requests = batch, requests
    runner.discard_request_indices = NS(gpu=torch.empty(0, dtype=torch.int64))
    runner.num_discarded_requests, runner.use_cp, runner.pcp_size = 0, False, 1
    runner.use_aux_hidden_state_outputs, runner.num_spec_tokens = False, 1
    runner.input_ids = NS(gpu=torch.arange(6, dtype=torch.int32))
    runner._get_positions = lambda index: common.positions[index]
    runner._copy_valid_sampled_token_count = lambda *args: calls.append("count_readback")

    def propose(**kwargs):
        assert kwargs["token_indices_to_sample"].data_ptr() == obj.token_indices_to_sample.data_ptr()
        assert kwargs["common_attn_metadata"].max_query_len == 2
        obj.set_inputs_first_pass(
            target_token_ids=kwargs["target_token_ids"],
            next_token_ids=kwargs["next_token_ids"],
            target_positions=kwargs["target_positions"],
            target_hidden_states=kwargs["target_hidden_states"],
            token_indices_to_sample=kwargs["token_indices_to_sample"],
            cad=kwargs["common_attn_metadata"],
            num_rejected_tokens_gpu=kwargs["num_rejected_tokens_gpu"],
        )
        return obj.input_ids[:6].clone()

    obj._propose = propose
    other = type("OtherProposer", (), {})
    dispatch = extract(
        ROOT / "vllm_ascend/worker/model_runner_v1.py",
        "propose_draft_token_ids",
        dict(torch=torch, AscendNgramProposer=other, AscendSuffixDecodingProposer=other, AscendMedusaProposer=other),
    )
    result = dispatch(
        runner,
        sampled,
        None,
        NS(num_scheduled_tokens={rid: 2 for rid in batch.req_ids}),
        spec,
        common,
        common.positions,
        6,
        torch.arange(48).float().view(6, 8),
    )
    assert result.shape == (6,)
    assert calls == ["prepare_next_mtp_tokens_kernel", "count_readback", "pack_mtp_tokens_positions_kernel"]


def test_failed_fused_launch_does_not_publish_or_reuse_a_handoff():
    obj, common, spec, batch, requests, sampled, _, ns = owner(3)
    obj._prepared_mtp_inputs = (common, spec, object(), object(), object())

    class Failure:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                raise RuntimeError("submission failed")

            return launch

    ns["prepare_next_mtp_tokens_kernel"] = Failure()
    with pytest.raises(RuntimeError, match="submission failed"):
        obj.prepare_next_token_ids_padded(common, sampled, requests, batch, torch.empty(0), 0, spec)
    assert obj._prepared_mtp_inputs is None


def test_draft_count_view_with_shared_pointer_but_wrong_stride_falls_back():
    obj, common, spec, batch, requests, sampled, _, _ = owner(3)
    counts = torch.arange(1, 7, dtype=torch.int32)
    obj.runner._fixed_mtp_metadata = (counts,)
    spec.cu_num_draft_tokens = counts[::2]
    obj.prepare_next_token_ids_padded(common, sampled, requests, batch, torch.empty(0), 0, spec)
    assert obj._prepared_mtp_inputs is None


def test_npu_hidden_state_view_consumers_finish_before_buffer_reuse():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires NPU")
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    arrays = extract(path, "_fixed_mtp_metadata_arrays", {"torch": torch})(16, "npu")
    runner = NS(_fixed_mtp_metadata=arrays, parallel_config=NS(pipeline_parallel_size=1))
    select = extract(path, "_select_sample_hidden_states", {"torch": torch})
    hidden = torch.zeros((40, 64), device="npu")
    weight = torch.eye(64, device="npu")
    held = []
    for n in (1, 4, 16, 3):
        hidden.fill_(n)
        indices = arrays[5][: 2 * n]
        # Compare queued consumers, then overwrite the persistent graph output.
        reference = hidden[indices] @ weight
        actual = select(runner, hidden, indices) @ weight
        held.append((actual, reference))
    torch.npu.synchronize()
    for actual, reference in held:
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
