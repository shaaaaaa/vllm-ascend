# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup contract and TP-local routing for independent full-SFA DP execution."""

import ctypes
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _mapped_file_identity(fd: int) -> tuple[int, int, int]:
    """Read the opened file's backing identity without relying on overlay stat IDs."""
    # ACCESS_COPY provides an addressable private buffer; no file bytes are written.
    with mmap.mmap(fd, 1, access=mmap.ACCESS_COPY) as probe:
        address = ctypes.addressof(ctypes.c_char.from_buffer(probe))
        for line in Path("/proc/self/maps").read_text().splitlines():
            columns = line.split(maxsplit=5)
            start, end = (int(value, 16) for value in columns[0].split("-"))
            if start <= address < end:
                major, minor = (int(value, 16) for value in columns[3].split(":"))
                return major, minor, int(columns[4])
    raise ValueError("Cannot identify the opened native file's backing mapping")


def _native_identity(maps: Path = Path("/proc/self/maps")) -> dict[str, str]:
    """Fingerprint mapped runtime files once; package labels are not build IDs."""
    libraries = {}
    for line in maps.read_text().splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) != 6 or "x" not in columns[1]:
            continue
        path = Path(columns[5])
        name = path.name
        if not name.startswith(
            ("libtorch_npu.", "libhccl", "libruntime.", "libopapi.", "libcust_opapi.", "vllm_ascend_C.")
        ):
            continue
        with path.open("rb") as file:
            before = os.fstat(file.fileno())
            major, minor = (int(value, 16) for value in columns[3].split(":"))
            expected = (major, minor, int(columns[4]))
            actual = (os.major(before.st_dev), os.minor(before.st_dev), before.st_ino)
            if actual != expected and _mapped_file_identity(file.fileno()) != expected:
                raise ValueError(f"Mapped native file was replaced: {path}")
            digest = hashlib.sha256()
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(file.fileno())
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Native file changed while identifying it")
        fingerprint = digest.hexdigest()
        if name in libraries and libraries[name] != fingerprint:
            raise ValueError("Different native builds share one library name")
        libraries[name] = fingerprint
    for required in ("libtorch_npu.", "libhccl.", "libruntime.", "vllm_ascend_C."):
        if not any(name.startswith(required) for name in libraries):
            raise ValueError(f"Cannot identify loaded native dependency {required}")
    return libraries


def topology_reason(config: Any) -> str:
    """Describe ineligibility without restricting the established execution path."""
    parallel, spec = config.parallel_config, config.speculative_config
    if parallel.data_parallel_size <= 1 or not parallel.enable_expert_parallel:
        return "requires MoE expert parallel across DP"
    if parallel.distributed_executor_backend != "mp" or parallel.nnodes_within_dp != 1:
        return "requires one local MP executor per TP group"
    if (
        any(
            getattr(parallel, name) != 1
            for name in (
                "pipeline_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        )
        or parallel.enable_dbo
        or getattr(parallel, "enable_elastic_ep", False)
    ):
        return "requires fixed PP=PCP=DCP=1 without DBO"
    if spec is not None and (
        spec.num_speculative_tokens != 1
        or spec.method not in ("mtp", "deepseek_mtp")
        or getattr(spec, "parallel_drafting", False)
        or getattr(spec, "disable_padded_drafter_batch", False)
    ):
        return "requires the existing one-token MTP or non-speculative schedule"
    if (
        spec is not None
        and getattr(spec, "draft_tensor_parallel_size", parallel.tensor_parallel_size) != parallel.tensor_parallel_size
    ):
        return "draft TP subgroup differs from the qualified target TP group"
    return ""


def execution_contract(runner: Any) -> dict[str, Any]:
    """Qualify the actual loaded runner after capture, before any live bypass."""
    import torch_npu
    from vllm.distributed import get_ep_group, get_tp_group

    from vllm_ascend.ascend_forward_context import MoECommType, get_mc2_tokens_capacity, select_moe_comm_method
    from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method
    from vllm_ascend.utils import (
        AscendDeviceType,
        enable_dsa_cp,
        enable_sp,
        get_ascend_device_type,
        is_drafter_moe_model,
        is_hierarchical_communication_enabled,
        is_moe_model,
        sfa_full_graph_enabled,
        shared_expert_dp_enabled,
    )

    config = runner.vllm_config
    reason = topology_reason(config)
    if reason:
        return dict(reason=reason)
    graph = getattr(runner, "_sfa_full_graph", None)
    if not sfa_full_graph_enabled(config) or graph is None or not graph.sealed:
        return dict(reason="requires sealed full-SFA capture")
    if not is_moe_model(config) or is_hierarchical_communication_enabled():
        return dict(reason="requires non-hierarchical MoE MC2")
    if not all(
        hasattr(torch_npu, name)
        for name in (
            "npu_moe_distribute_dispatch_v2",
            "npu_moe_distribute_combine_v2",
        )
    ):
        return dict(reason="paired MC2 V2 bindings are unavailable")
    device = get_ascend_device_type()
    if device not in (AscendDeviceType.A2, AscendDeviceType.A3):
        return dict(reason="device MC2 ABI has not been qualified for this mode")
    capacity = get_mc2_tokens_capacity()
    tp, ep = get_tp_group(), get_ep_group()
    dispatcher = getattr(get_moe_comm_method(MoECommType.MC2), "token_dispatcher", None)
    global_bs = math.ceil(capacity / tp.world_size) * ep.world_size
    if (
        dispatcher is None
        or not dispatcher.enable_dispatch_v2
        or dispatcher.need_comm_alg
        or dispatcher.global_bs != global_bs
        or dispatcher.ep_world_size != ep.world_size
    ):
        return dict(reason="loaded MC2 dispatcher differs from the qualified capacity/ABI")
    if math.ceil(capacity / tp.world_size) > (256 if device == AscendDeviceType.A2 else 512):
        return dict(reason="MC2 per-rank capacity exceeds the installed ABI bound")
    phases = (False, True) if config.speculative_config else (False,)
    draft_moe = bool(config.speculative_config and is_drafter_moe_model(config))
    for draft in phases:
        if not runner._can_skip_dp_metadata(draft):
            return dict(reason="native recovery/draft is not bounded to MC2")
        if not draft or draft_moe:
            # Check all legal local extents, not just two endpoints. The result
            # is immutable; no selectors or per-layer scans are added per step.
            if any(select_moe_comm_method(n, config, draft) != MoECommType.MC2 for n in range(1, capacity + 1)):
                return dict(reason="a legal local extent changes MC2 operator family")
    keys = sorted(
        (key.token_capacity, key.request_capacity, key.query_profile.value, key.max_query_len) for key in graph.entries
    )
    if not keys or any(key[0] > capacity for key in keys):
        return dict(reason="captured graph capacity exceeds MC2 allocation")
    if getattr(getattr(runner, "drafter", None), "needs_extra_input_slots", False):
        return dict(reason="draft expands the agreed MC2 input extent")
    try:
        native = _native_identity()
    except (OSError, ValueError) as error:
        return dict(reason=f"native runtime identity unavailable: {error}")
    models = [config.model_config]
    if draft_config := getattr(config.speculative_config, "draft_model_config", None):
        models.append(draft_config)
    # Include source identity without retaining source text or rebuilding it on
    # admission/decode. Virtual addresses and per-request source IDs are local.
    sources = {}
    for file in (
        Path(__file__),
        Path(__file__).with_name("model_runner_v1.py"),
        Path(__file__).with_name("sfa_async_mtp.py"),
        Path(__file__).parents[1] / "ascend_forward_context.py",
        Path(__file__).parents[1] / "ascend_config.py",
        Path(__file__).parents[1] / "utils.py",
        Path(__file__).parents[1] / "attention/sfa_v1.py",
        Path(__file__).parents[1] / "ops/register_custom_ops.py",
        Path(__file__).parents[1] / "spec_decode/eagle_proposer.py",
        Path(__file__).parents[1] / "ops/fused_moe/fused_moe.py",
        Path(__file__).parents[1] / "ops/fused_moe/token_dispatcher.py",
        Path(__file__).parents[1] / "ops/fused_moe/prepare_finalize.py",
    ):
        sources[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    contract = dict(
        schema=1,
        tp=tp.world_size,
        ep=ep.world_size,
        ranks=list(ep.ranks),
        models=[
            dict(
                model=model.hf_text_config.to_dict(),
                dtype=str(model.dtype),
                quantization=getattr(model, "quantization", None),
            )
            for model in models
        ],
        keys=keys,
        capacity=capacity,
        global_bs=dispatcher.global_bs,
        dispatcher=f"{type(dispatcher).__module__}.{type(dispatcher).__qualname__}",
        device=str(device),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        sources=sources,
        native=native,
        runner=f"{type(runner).__module__}.{type(runner).__qualname__}",
        draft_moe=draft_moe,
        draft_rounds=int(bool(config.speculative_config)),
        recovery=getattr(config.scheduler_config, "mc2_recovery_token_budget", None),
        async_scheduling=config.scheduler_config.async_scheduling,
        additional_config=config.additional_config,
        token_layout=(enable_sp(config), enable_dsa_cp(), shared_expert_dp_enabled()),
        hccl_buffer_mb=int(os.environ.get("HCCL_BUFFSIZE", "200")),
        mc2_abi="dispatch/combine_v2;common_global_bs;no_active_mask;default_comm_alg",
    )
    return dict(
        digest=hashlib.sha256(json.dumps(contract, sort_keys=True, default=str).encode()).hexdigest(),
        capacity=capacity,
        reason="",
    )


class IndependentDP:
    """One preallocated TP-only proposal check; no healthy cross-DP host vote."""

    def __init__(self, tp: Any, dp_size: int, capacity: int):
        self.rank, self.width = tp.rank_in_group, tp.world_size
        self.capacity, self.sequence = capacity, 0
        self.packet = torch.zeros((7, self.width), dtype=torch.int64, device="cpu")
        # Stable CPU views avoid scalar Torch dispatch on every target/draft vote.
        self.proposal = self.packet[:, self.rank].numpy()
        self.reference = self.packet[:, :1].expand_as(self.packet)
        self.counts = [torch.empty(dp_size, dtype=torch.int32, device="cpu") for _ in range(2)]
        # LMCache uses tp.cpu_group from background threads. Routing needs its
        # own sequence domain, created once before serving, with only TP peers.
        # WorkerProc's distributed teardown destroys this registered group too.
        self.group = dist.new_group(
            ranks=tp.ranks, backend="gloo", use_local_synchronization=True,
            group_desc="sfa_independent_tp",
        ) if self.width > 1 else None

    def agree(
        self, tokens: int, mode: int, route: int, bounded: bool, *, draft: bool = False, requests: int = 0
    ) -> torch.Tensor:
        """Keep TP shape/route/phase consistency while permitting unequal DP buckets."""
        if not 0 < tokens <= self.capacity:
            raise ValueError("Local SFA extent exceeds the fixed MC2 contract")
        self.sequence += 1
        self.packet.zero_()
        self.proposal[:] = (tokens, mode, route, int(bounded), int(draft), self.sequence, requests)
        if self.width > 1:
            dist.all_reduce(self.packet, group=self.group)
            if not torch.equal(self.packet, self.reference):
                raise RuntimeError("Independent SFA proposal/phase differs within TP")
        counts = self.counts[int(draft)]
        counts.fill_(tokens)  # Local API compatibility, not observed peer counts.
        return counts
