# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts; run without importing the NPU plugin/test bootstrap."""

import ast
import runpy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]
API = runpy.run_path(str(ROOT / "vllm_ascend/core/mc2_recovery.py"))


def config(*, sizes=(8, 16, 24, 32), tp=4, seqs=16, drafts=1, consumer=True):
    return NS(
        compilation_config=NS(cudagraph_capture_sizes=sizes, max_cudagraph_capture_size=max(sizes, default=0)),
        parallel_config=NS(
            tensor_parallel_size=tp,
            enable_expert_parallel=True,
            data_parallel_size=4,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_dbo=False,
        ),
        kv_transfer_config=NS(is_kv_consumer=consumer),
        speculative_config=NS(num_speculative_tokens=drafts, method="mtp"),
        scheduler_config=NS(
            max_num_seqs=seqs,
            max_num_batched_tokens=4096,
            enable_chunked_prefill=True,
            scheduler_cls="vllm_ascend.core.recompute_scheduler.AsyncRecomputeScheduler",
        ),
    )


@pytest.mark.parametrize(
    "sizes,tp,seqs,width,expected",
    [
        ((8, 16, 24, 32), 4, 16, 2, 32),
        ((17,), 8, 16, 2, 24),
        ((), 4, 16, 2, 32),
        ((), 4, 1000, 2, 512),
        ((), 8, 7, 1, 8),
    ],
)
def test_capacity_matches_mc2_allocation(sizes, tp, seqs, width, expected):
    cfg = config(sizes=sizes, tp=tp, seqs=seqs)
    assert API["calculate_mc2_tokens_capacity"](cfg, seqs, width) == expected


def test_only_consumer_budget_changes_and_original_limit_is_retained():
    cfg = config()
    assert API["decoder_recovery_budget"](cfg) == 32
    assert cfg.scheduler_config.max_num_batched_tokens == 4096
    cfg.scheduler_config.max_num_batched_tokens = 12
    assert API["decoder_recovery_budget"](cfg) == 12
    cfg.kv_transfer_config.is_kv_consumer = False
    assert API["decoder_recovery_budget"](cfg) is None


def test_non_chunkable_recovery_is_rejected():
    cfg = config()
    cfg.scheduler_config.enable_chunked_prefill = False
    with pytest.raises(ValueError, match="chunked_prefill"):
        API["decoder_recovery_budget"](cfg)


def test_sync_gate_does_not_trust_recompute_flag_alone():
    tree = ast.parse((ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_can_skip_dp_metadata")
    method.decorator_list = []
    calls = []
    ns = dict(
        is_moe_model=lambda _: True,
        is_drafter_moe_model=lambda _: True,
        MoECommType=NS(MC2=1, FUSED_MC2=2),
        get_mc2_tokens_capacity=lambda: 32,
    )

    def select(tokens, cfg, draft):
        calls.append(draft)
        return 1 if tokens <= 32 else 0

    ns["select_moe_comm_method"] = select
    exec(compile(ast.Module(body=[method], type_ignores=[]), "gate", "exec"), ns)
    cfg = config()
    runner = NS(
        vllm_config=cfg,
        is_kv_consumer=True,
        compilation_config=cfg.compilation_config,
        max_num_reqs=16,
        uniform_decode_query_len=2,
        ascend_config=NS(recompute_scheduler_enable=True),
        pcp_size=1,
        dcp_size=1,
        parallel_config=NS(enable_dbo=False, pipeline_parallel_size=1),
    )
    gate = ns[method.name]
    assert not gate(runner)
    cfg.scheduler_config.mc2_recovery_token_budget = 32
    assert gate(runner)
    assert gate(runner, True)
    assert calls[-1] is True
    runner.drafter = NS(needs_extra_input_slots=True)
    assert not gate(runner)
    runner.drafter = None
    runner.pcp_size = 2
    assert not gate(runner)
    runner.pcp_size = 1
    cfg.scheduler_config.scheduler_cls = "different.Scheduler"
    assert not gate(runner)


@pytest.mark.parametrize("case", ["parallel_drafting", "draft_model", "unpadded", "pipeline"])
def test_unqualified_recovery_keeps_metadata_agreement(case):
    cfg = config()
    if case == "parallel_drafting":
        cfg.speculative_config.parallel_drafting = True
    elif case == "draft_model":
        cfg.speculative_config.method = "draft_model"
    elif case == "unpadded":
        cfg.speculative_config.disable_padded_drafter_batch = True
    else:
        cfg.parallel_config.pipeline_parallel_size = 2
    assert API["decoder_recovery_budget"](cfg) is None


def test_production_draft_expansion_can_exceed_target_capacity():
    import torch

    source = ast.parse((ROOT / "vllm_ascend/spec_decode/eagle_proposer.py").read_text(encoding="utf-8"))
    function = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == "set_inputs_first_pass")
    seen = []

    class StopAtNative(RuntimeError):
        pass

    def native(*args):
        seen.append(args[-1])
        raise StopAtNative()

    ns = {"torch": NS(int32=torch.int32, ops=NS(_C_ascend=NS(npu_copy_and_expand_eagle_inputs=native)))}
    prefix = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix, function], type_ignores=[])), "draft", "exec"), ns)
    drafter = NS(
        needs_extra_input_slots=True,
        is_rejected_token_mask=object(),
        is_masked_token_mask=object(),
        net_num_new_slots_per_request=1,
        parallel_drafting_token_id=0,
        extra_slots_per_request=2,
        pass_hidden_states_to_model=True,
    )
    with pytest.raises(StopAtNative):
        ns[function.name](
            drafter,
            torch.arange(32),
            torch.tensor([1]),
            torch.arange(32),
            torch.zeros(32, 4),
            None,
            NS(batch_size=lambda: 1, query_start_loc=torch.tensor([0, 32])),
            None,
        )
    assert seen == [33]


def test_per_step_gate_only_reads_startup_decisions():
    tree = ast.parse((ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_skip_all_reduce_across_dp_group"
    )
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "gate", "exec"), ns)
    runner = NS(_dp_metadata_skip=(True, False))
    assert ns[fn.name](runner)
    assert not ns[fn.name](runner, True)


def test_async_scheduler_inherits_the_bounded_ascend_schedule():
    sources = [
        (ROOT.parent / "vllm/vllm/v1/core/sched/async_scheduler.py", {"AsyncScheduler"}),
        (ROOT / "vllm_ascend/core/recompute_scheduler.py", {"RecomputeScheduler", "AsyncRecomputeScheduler"}),
    ]
    nodes = []
    for path, names in sources:
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef) and node.name in names:
                # Retain any schedule override and the actual declared bases.
                node.body = [n for n in node.body if getattr(n, "name", None) == "schedule"] or [ast.Pass()]
                nodes.append(node)
    ns = {"Scheduler": type("Scheduler", (), {})}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), "scheduler_mro", "exec"), ns)
    assert ns["AsyncRecomputeScheduler"].schedule is ns["RecomputeScheduler"].schedule
