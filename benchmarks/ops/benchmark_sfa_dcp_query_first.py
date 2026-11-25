# SPDX-License-Identifier: Apache-2.0
"""Compare KV-first and Q-first consumers through a complete DCP layer group.

Includes BF16 projections, real KV normalization/RoPE/cache writes, resident
planning, registered-host sparse H2D, native attention and DCP exchange/merge.
Both arms already use early gathering on the producer. No synthetic delay.
"""

import argparse
import json
import os
import statistics
import sys
from datetime import timedelta
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
from benchmark_sfa_copy_reduction import AttentionPipeline, measure_case
from benchmark_sfa_dcp_tuning import format_case_summary
from benchmark_sfa_target_preparation import BLOCK, LATENT, ROOT, ROPE, make_inputs


def current_slots(positions, rows, table, origin, world, rank):
    """Global sequence positions -> physical slots on exactly one DCP rank."""
    local = origin + positions // (BLOCK * world) * BLOCK + positions % BLOCK
    slots = table[rows.clamp_min(0).long(), local // BLOCK].long() * BLOCK + local % BLOCK
    return torch.where((rows >= 0) & (positions // BLOCK % world == rank), slots, -1)


class Projection:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x):
        return torch.nn.functional.linear(x, self.weight), None


class QueryFirstPipeline(AttentionPipeline):
    def __init__(self, args, inputs, sources, group, candidate):
        import torch_npu

        from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

        super().__init__(args, inputs, sources, group, direct=True, early_gather=True)
        self.query_first = candidate
        self.projectors, self.frame_inputs = [], {}
        rng = torch.Generator().manual_seed(args.seed + 731)
        head_rng = torch.Generator().manual_seed(args.seed + 887 + args.rank)

        def tensor(*shape, scale=1.0, generator=rng):
            return (torch.randn(shape, generator=generator) * scale).to(device=self.device, dtype=torch.bfloat16)

        self.norm_weight = torch.ones(args.q_rank, device=self.device, dtype=torch.bfloat16)
        for _ in range(args.layers):
            impl = SimpleNamespace(
                enable_dsa_cp=False,
                num_kv_heads=1,
                kv_lora_rank=LATENT,
                qk_rope_head_dim=ROPE,
                local_num_heads=args.heads,
                qk_nope_head_dim=args.qk_nope,
                qk_head_dim=args.qk_nope + ROPE,
                kv_a_layernorm=SimpleNamespace(
                    weight=torch.ones(LATENT, device=self.device, dtype=torch.bfloat16), variance_epsilon=1e-6
                ),
                qkv=Projection(tensor(args.q_rank + LATENT + ROPE, args.hidden_size, scale=args.hidden_size**-0.5)),
                q_proj=Projection(
                    tensor(
                        args.heads * (args.qk_nope + ROPE),
                        args.q_rank,
                        scale=args.q_rank**-0.5,
                        generator=head_rng,
                    )
                ),
                W_UK_T=tensor(args.heads, args.qk_nope, LATENT, scale=args.qk_nope**-0.5),
                W_UV=tensor(args.heads, LATENT, args.value_dim, scale=LATENT**-0.5),
                o_proj=Projection(
                    tensor(args.hidden_size, args.heads * args.value_dim, scale=(args.heads * args.value_dim) ** -0.5)
                ),
            )
            for name in ("exec_kv", "_q_proj_and_k_up_proj", "rope_single"):
                setattr(impl, name, MethodType(getattr(AscendSFAImpl, name), impl))
            self.projectors.append(impl)
        rows, table = self.rows.cpu(), self.table.cpu()
        for raw, positions in self.frames:
            pos = positions.cpu()
            slots = current_slots(pos, rows, table, self.origin, args.dcp_size, args.rank)
            # Include current KV in attention, so stale/missing current writes
            # cannot pass parity merely because only historical tokens are read.
            active = torch.nonzero(rows >= 0).flatten().to(self.device)
            raw[active, 0, -1] = positions.index_select(0, active).int()
            angles = pos.float()[:, None] / (10000 ** (torch.arange(ROPE // 2).float() / (ROPE // 2)))
            angles = angles.repeat_interleave(2, dim=-1).reshape(self.tokens, 1, 1, ROPE)
            self.frame_inputs[raw.data_ptr()] = (
                tensor(self.tokens, args.hidden_size),
                angles.cos().to(self.device, torch.bfloat16),
                angles.sin().to(self.device, torch.bfloat16),
                slots.to(self.device),
            )
        self.rms_norm = torch_npu.npu_rms_norm

    def prepare_producer(self, frame, hidden, q_c, cos, sin):
        return self.prepare(frame)

    def step(self, frame):
        from vllm_ascend.ops.triton.sfa_dcp import execute_sfa_dcp, start_sfa_dcp_query_gather

        hidden, cos, sin, slots = self.frame_inputs[frame[0].data_ptr()]
        self.attention_outputs = []
        self.current_writes = []
        for layer, impl in enumerate(self.projectors):
            edge = self.edges[layer - 1] if layer and self.edges else None
            query_first = self.query_first and layer > 0
            if edge is not None and not query_first:
                edge.join()
            pending = None

            def query_ready(q, r, edge=edge):
                nonlocal pending
                pending = start_sfa_dcp_query_gather(q, r, self.group)
                if edge is not None and edge.pending_capture_launch:
                    edge.join()

            try:
                q_c, kv = impl.qkv(hidden)[0].split([self.args.q_rank, LATENT + ROPE], dim=-1)
                q_c = self.rms_norm(q_c, self.norm_weight, epsilon=1e-6)[0]
                if query_first:
                    q, r = impl._q_proj_and_k_up_proj(q_c)
                    r = impl.rope_single(r, cos, sin)
                    query_ready(q, r)
                current = impl.exec_kv(kv, cos, sin, self.layer_caches[layer], slots, None)
                # Validation retains the real op's returned current KV, outside
                # timed verification; both arms have the same storage lifetime.
                self.current_writes.append((slots, current))
                if not query_first:
                    q, r = impl._q_proj_and_k_up_proj(q_c)
                    r = impl.rope_single(r, cos, sin)
                    if not layer or not self.edges:
                        query_ready(q, r)
                if not layer:
                    prepared = self.prepare_producer(frame, hidden, q_c, cos, sin)
                self._attend_layer(
                    layer,
                    (q, r),
                    self.layer_caches[layer],
                    self.transfers[layer],
                    self.bridges[layer],
                    prepared,
                    pending,
                    execute_sfa_dcp,
                )
                # Include local output projection in the incoming-prefetch slack.
                values = torch.bmm(self.attention_outputs[-1].transpose(0, 1), impl.W_UV)
                projected = impl.o_proj(values.transpose(0, 1).reshape(self.tokens, -1))[0]
                hidden = hidden + projected
            finally:
                try:
                    if edge is not None and edge.pending_capture_launch:
                        edge.join()
                finally:
                    if pending is not None:
                        pending.wait()
        self.projected_output = hidden

    def check_transfer(self, *, caches=None):
        misses = super().check_transfer(caches=caches)
        if caches is None:
            # Also check current writes, including ranks with no owned tokens.
            for dest, (slots, (rope, latent)) in zip(self.layer_caches, self.current_writes):
                ids = slots.cpu()
                owned = ids >= 0
                if not owned.any():
                    continue
                for cache, values in zip(dest, (latent, rope)):
                    actual = cache.view(-1, cache.shape[-1]).index_select(0, ids[owned].to(cache.device)).cpu()
                    expected = values.reshape(self.tokens, -1).cpu()[owned]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return misses

    def snapshot(self, *, include_cache=True):
        return super().snapshot(include_cache=include_cache) + [self.projected_output.cpu()]


@torch.inference_mode()
def main(*, comparison="query-first", pipeline_factory=QueryFirstPipeline, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    if comparison == "indexer-sharding":
        parser.add_argument("--owner-baseline", choices=("replicated", "restore"), default="replicated",
                            help="use restore to time the former sharded chain against direct planning")
        parser.add_argument("--owner-layout", choices=("balanced", "skewed"), default="balanced")
        parser.add_argument("--skew-history", action="store_true")
        parser.add_argument("--replicate-owner-storage", action="store_true",
                            help="isolate owner compute while retaining replicated history in the candidate")
    if comparison == "prefetch-after-pack":
        parser.add_argument(
            "--query-first",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="hold consumer Q-first scheduling fixed in both arms",
        )
    parser.add_argument("--lmcache-ascend-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--layouts", nargs="+", choices=("uniform", "bounded", "mixed"), default=["uniform"])
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--heads", type=int, default=64, help="global attention heads")
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--q-rank", type=int, default=1536)
    parser.add_argument("--qk-nope", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4, help="one producer plus shared-index consumers")
    parser.add_argument("--overlap-retrieval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.9)
    parser.add_argument("--history", type=int, default=16384)
    parser.add_argument("--live-tokens", type=int, default=128)
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--json", type=Path, default=Path(f"sfa-dcp-{comparison}.json"))
    args = parser.parse_args()
    args.dcp_size, args.rank, args.device = (int(os.environ[name]) for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    if (
        args.dcp_size not in (2, 4, 8)
        or args.heads % args.dcp_size
        or args.heads % 8
        or min(*args.requests, args.heads, args.layers, args.steps, args.trials, args.iterations, args.history) < 1
        or args.warmup_steps < 0
        or any(n < 16 or n % 16 for n in (args.hidden_size, args.q_rank, args.qk_nope, args.value_dim))
    ):
        parser.error("use DCP=2/4/8, divisible heads, positive sizes and projection dimensions aligned to 16")
    for layout in args.layouts:
        try:
            make_inputs(1, 1, args.history, args.live_tokens, args.overlap, layout, args.padding)
        except ValueError as exc:
            parser.error(str(exc))
    if not (args.lmcache_ascend_dir / "benchmark/v1/kv_transfer/load_benchmark_utils.py").is_file():
        parser.error("--lmcache-ascend-dir must be the companion LMCache-Ascend checkout")
    sys.path.insert(0, str(ROOT))
    import torch_npu

    from vllm_ascend import envs
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(args.device)
    if not enable_custom_op():
        raise RuntimeError("vLLM-Ascend native operators are unavailable")
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    results = {
        "scope": (
            "QKV/Q projections, RMSNorm/RoPE/current KV, planner/H2D, DCP SFA exchange/merge, local output projection"
        ),
        "excludes": (
            "quantized projections, actual indexer compute, TP output reduction, MoE, sampling/server scheduling"
        ),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "HCCL_OP_EXPANSION_MODE": os.getenv("HCCL_OP_EXPANSION_MODE"),
        "dcp_hccl_host": bool(envs.VLLM_ASCEND_DCP_HCCL_HOST),
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cases": [],
    }
    if comparison == "indexer-sharding":
        results["scope"] += ", real C8 Lightning Indexer, key/scale writes, DCP owner gather and resident planning"
        results["baseline_indexer"] = args.owner_baseline
        results["excludes"] = "quantized projection weights, TP output reduction, MoE, sampling and server scheduling"
    try:
        for layout in args.layouts:
            for requests in args.requests:
                case_args = SimpleNamespace(
                    **{
                        **vars(args),
                        "layout": layout,
                        "requests": requests,
                        "shared_plan": True,
                        "heads": args.heads // args.dcp_size,
                    }
                )
                if not args.rank:
                    print(f"Starting {comparison} {layout} batch={requests} DCP={args.dcp_size}", flush=True)
                case = measure_case(case_args, control, pipeline_factory=pipeline_factory)
                case.update(
                    variant=comparison,
                    slowest_rank_mean={
                        name: statistics.mean(
                            max(rank[name][i] for rank in case["per_rank_samples"])
                            for i in range(len(case["per_rank_samples"][0][name]))
                        )
                        for name in ("baseline", "direct")
                    },
                    median_reduction_percent=100
                    * (1 - case["slowest_rank_median"]["direct"] / case["slowest_rank_median"]["baseline"]),
                )
                if not args.rank:
                    results["cases"].append(case)
                    args.json.parent.mkdir(parents=True, exist_ok=True)
                    args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
                    print(format_case_summary(case), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
