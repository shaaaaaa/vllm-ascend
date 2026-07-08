from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar
import hashlib
import io
import json
import os
import re
import time
import scipy  # type: ignore
import numpy as np
import torch
import torch_npu
import vllm.envs as envs_vllm
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.sparse_offload import _prof as _dsa_prof
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadataBuilder
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import (
    AttentionBackend,  # type: ignore
    AttentionCGSupport,
    MLAAttentionImpl,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.common_cp import AscendPCPMetadata
from vllm_ascend.attention.mla_v1 import MAX_O_PROJ_PREFETCH_SIZE, MLAPO_MAX_SUPPORTED_TOKENS
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    ascend_chunked_prefill_workspace_size,
    enable_cp,
    maybe_save_kv_layer_to_connector,
    trans_rope_weight,
    transdata,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.kv_transfer.sparse_offload.scratch_remap import scratch_remap
from vllm_ascend.distributed.utils import all_gather_async
from vllm_ascend.ops.layer_shard_linear import (
    is_hidden_layer,
    post_process_after_loading_for_shard_weight_series,
    reach_layer_for_shard_weight_series,
    register_all_layers_to_shard_weight_series,
)
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.ops.triton.rope import rope_forward_triton_siso
from vllm_ascend.quantization.methods import AscendW8A8LinearMethod
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_ND,
    _round_up,
    dispose_layer,
    enable_dsa_cp,
    enable_dsa_cp_with_layer_shard,
    enable_dsa_cp_with_o_proj_tp,
    get_weight_prefetch_method,
    maybe_trans_nz,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

# token count limits within bmm_transpose operator
BMM_TRANS_MAX_SUPPORTED_TOKENS = 1024
_DSA_TARGET_SLOT_GUARD = envs.VLLM_ASCEND_DSA_TARGET_SLOT_GUARD
_DSA_INDEXER_DIAG = os.getenv("VLLM_ASCEND_DSA_INDEXER_DIAG", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_DSA_INDEXER_DIAG_DUMP = os.getenv(
    "VLLM_ASCEND_DSA_INDEXER_DIAG_DUMP", "0"
).lower() in ("1", "true", "yes", "on")
_DSA_INDEXER_DIAG_DUMP_FULL_KEY = os.getenv(
    "VLLM_ASCEND_DSA_INDEXER_DIAG_DUMP_FULL_KEY", "0"
).lower() in ("1", "true", "yes", "on")
_DSA_INDEXER_DIAG_DUMP_DIR = os.getenv(
    "VLLM_ASCEND_DSA_INDEXER_DIAG_DUMP_DIR",
    "/tmp/vllm_dsa_indexer_diag",
)
_DSA_INDEXER_DIAG_SESSION_ID = os.getenv(
    "VLLM_ASCEND_DSA_INDEXER_DIAG_SESSION_ID"
) or f"pid{os.getpid()}_{int(time.time() * 1000)}"
_DSA_INDEXER_DIAG_RUNS: dict[str, int] = {}
_DSA_INDEXER_DIAG_REQ_RUNS: dict[str, tuple[str, int]] = {}


def _dsa_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        logger.warning(
            "Invalid integer env %s=%r; using %s",
            name,
            os.getenv(name),
            default,
        )
        return default


_DSA_INDEXER_DIAG_LAYERS = _dsa_env_int(
    "VLLM_ASCEND_DSA_INDEXER_DIAG_LAYERS", 8
)
_DSA_INDEXER_DIAG_SAMPLE_VALUES = _dsa_env_int(
    "VLLM_ASCEND_DSA_INDEXER_DIAG_VALUES", 8
)
_DSA_INDEXER_DIAG_EXACT_VALUES = os.getenv(
    "VLLM_ASCEND_DSA_INDEXER_DIAG_EXACT_VALUES", "0"
).lower() in ("1", "true", "yes", "on")
_DSA_INDEXER_DIAG_MAX_DIGEST_NUMEL = _dsa_env_int(
    "VLLM_ASCEND_DSA_INDEXER_DIAG_MAX_DIGEST_NUMEL", 65536
)
_DSA_INDEXER_DIAG_KEY_SAMPLE_TOKENS = _dsa_env_int(
    "VLLM_ASCEND_DSA_INDEXER_DIAG_KEY_SAMPLE_TOKENS", 128
)
_DSA_INDEXER_DIAG_COMPONENTS_ENV = os.getenv(
    "VLLM_ASCEND_DSA_INDEXER_DIAG_COMPONENTS", "all"
)
_DSA_INDEXER_DIAG_COMPONENTS = {
    component.strip().lower()
    for component in _DSA_INDEXER_DIAG_COMPONENTS_ENV.split(",")
    if component.strip() and component.strip().lower() not in ("0", "none", "off")
}
_DSA_INDEXER_DIAG_SLEEP_MS = _dsa_env_int(
    "VLLM_ASCEND_DSA_INDEXER_DIAG_SLEEP_MS", 0
)
_DSA_SYNC_PROBES = {
    probe.strip().lower()
    for probe in os.getenv("VLLM_ASCEND_DSA_SYNC_PROBE", "").split(",")
    if probe.strip()
}
_DSA_SYNC_PROBE_ONCE: set[tuple[str, str | None]] = set()


def _dsa_debug_sample(value, limit: int = 8) -> list:
    if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
        return []
    if value.numel() > _DSA_INDEXER_DIAG_MAX_DIGEST_NUMEL and not value.is_contiguous():
        return ["sample_skipped_non_contiguous_large_tensor"]
    return value.detach().reshape(-1)[:limit].to(device="cpu").tolist()


def _dsa_debug_minmax_count(value) -> tuple[object, object, int] | None:
    if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    flat = value.detach().reshape(-1)
    return (
        flat.min().to(device="cpu").item(),
        flat.max().to(device="cpu").item(),
        int(flat.numel()),
    )


def _dsa_diag_safe_name(value: Any) -> str:
    raw = str(value if value is not None else "none")
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in raw
    )[:160]


def _dsa_diag_layer_id(layer_name: str | None) -> int:
    match = re.search(r"layers\.(\d+)", str(layer_name))
    return int(match.group(1)) if match else -1


def _dsa_diag_tensor_digest(
    value: Any,
    *,
    include_stride: bool = True,
    max_numel: int | None = None,
) -> str | None:
    if not isinstance(value, torch.Tensor):
        return None
    h = hashlib.blake2b(digest_size=16)
    h.update(str(tuple(int(dim) for dim in value.shape)).encode("utf-8"))
    h.update(str(value.dtype).encode("utf-8"))
    if include_stride:
        h.update(
            str(tuple(int(stride) for stride in value.stride())).encode(
                "utf-8"
            )
        )
    try:
        tensor = value.detach()
        if max_numel is not None and tensor.numel() > max_numel:
            h.update(f"sampled:{max_numel}/{int(tensor.numel())}".encode("utf-8"))
            if tensor.numel() > max_numel and not tensor.is_contiguous():
                h.update(b"sample_skipped_non_contiguous_large_tensor")
                return h.hexdigest()
            flat = tensor.reshape(-1)
            # Use a deterministic prefix sample. It is cheap and avoids the
            # full NPU->CPU copy that made this diagnostic unusably slow.
            tensor = flat[:max_numel]
        cpu = tensor.to(device="cpu").contiguous()
        try:
            raw = cpu.view(torch.uint8).numpy().tobytes()
        except Exception:
            buf = io.BytesIO()
            torch.save(cpu, buf)
            raw = buf.getvalue()
        h.update(raw)
    except Exception as exc:
        h.update(f"digest_error:{type(exc).__name__}:{exc}".encode("utf-8"))
    return h.hexdigest()


def _dsa_diag_tensor_summary(
    value: Any,
    *,
    digest_values: bool = True,
    max_digest_numel: int | None = None,
) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__, "value": value}
    summary = {
        "type": "Tensor",
        "shape": tuple(int(dim) for dim in value.shape),
        "stride": tuple(int(stride) for stride in value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": int(value.numel()),
        "head": _dsa_debug_sample(value, _DSA_INDEXER_DIAG_SAMPLE_VALUES),
    }
    if digest_values:
        digest = _dsa_diag_tensor_digest(
            value,
            include_stride=False,
            max_numel=max_digest_numel,
        )
        digest_numel = int(value.numel())
        if max_digest_numel is not None:
            digest_numel = min(digest_numel, int(max_digest_numel))
        summary.update(
            {
                "digest": digest,
                "value_digest": digest,
                "value_digest_exact": (
                    max_digest_numel is None
                    or int(value.numel()) <= int(max_digest_numel)
                ),
                "value_digest_numel": digest_numel,
            }
        )
    else:
        summary.update(
            {
                "digest": None,
                "value_digest": None,
                "value_digest_exact": False,
                "value_digest_numel": 0,
                "digest_skipped": "metadata_only",
            }
        )
    return summary


def _dsa_diag_summary_path(
    *,
    session_id: str,
    signature: str,
    run: int,
    layer_name: str | None,
    tp_rank: int,
) -> str:
    return os.path.join(
        _DSA_INDEXER_DIAG_DUMP_DIR,
        (
            f"{_dsa_diag_safe_name(session_id)}"
            f"_{_dsa_diag_safe_name(signature)}"
            f"_run{run}"
            f"_{_dsa_diag_safe_name(layer_name)}"
            f"_rank{tp_rank}_summary.json"
        ),
    )


def _dsa_diag_write_json(path: str, payload: dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2)
    os.replace(tmp_path, path)


def _dsa_indexer_diag_component(name: str) -> bool:
    return (
        "all" in _DSA_INDEXER_DIAG_COMPONENTS
        or name in _DSA_INDEXER_DIAG_COMPONENTS
    )


def _dsa_sync_probe(name: str, layer_name: str | None = None) -> None:
    if name not in _DSA_SYNC_PROBES and "all" not in _DSA_SYNC_PROBES:
        return
    try:
        torch.npu.synchronize()
    except Exception:
        logger.warning(
            "[DSA_SYNC_PROBE_ERROR] probe=%s layer=%s failed",
            name,
            layer_name,
            exc_info=True,
        )
        return
    log_key = (name, layer_name)
    if log_key not in _DSA_SYNC_PROBE_ONCE:
        _DSA_SYNC_PROBE_ONCE.add(log_key)
        logger.warning("[DSA_SYNC_PROBE] probe=%s layer=%s", name, layer_name)


def _dsa_topk_to_2d_indices(topk_indices: torch.Tensor) -> torch.Tensor:
    if topk_indices.dim() == 3 and topk_indices.shape[1] == 1:
        return topk_indices[:, 0, :]
    if topk_indices.dim() == 2:
        return topk_indices
    return topk_indices.reshape(topk_indices.shape[0], -1)


def _dsa_mask_padding_sparse_rows(
    topk_indices: torch.Tensor,
    row_req_indices: torch.Tensor | None,
    num_actual_rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep graph padding rows from referencing freed DSA logical blocks."""
    topk_2d = _dsa_topk_to_2d_indices(topk_indices)
    num_rows = int(topk_2d.shape[0])
    if row_req_indices is None:
        return topk_indices, topk_2d
    if num_actual_rows is not None and num_rows <= int(num_actual_rows):
        return topk_indices, topk_2d

    row_req_indices = row_req_indices[:num_rows].to(
        device=topk_indices.device, dtype=torch.long
    )
    if int(row_req_indices.numel()) < num_rows:
        pad = torch.full(
            (num_rows - int(row_req_indices.numel()),),
            -1,
            dtype=torch.long,
            device=topk_indices.device,
        )
        row_req_indices = torch.cat((row_req_indices, pad), dim=0)
    padding_mask = row_req_indices < 0
    if topk_indices.dim() == 3 and topk_indices.shape[1] == 1:
        topk_indices = topk_indices.masked_fill(
            padding_mask.reshape(-1, 1, 1), 0
        )
    elif topk_indices.dim() == 2:
        topk_indices = topk_indices.masked_fill(padding_mask.reshape(-1, 1), 0)
    else:
        topk_indices = topk_indices.clone()
        topk_indices.reshape(num_rows, -1).masked_fill_(
            padding_mask.reshape(-1, 1), 0
        )
    return topk_indices, _dsa_topk_to_2d_indices(topk_indices)


def _dsa_sparse_fa_bad_block_hit(
    topk_2d: torch.Tensor,
    block_table: torch.Tensor,
    row_req_indices: torch.Tensor | None,
    block_size: int,
) -> dict[str, object] | None:
    if row_req_indices is None or topk_2d.numel() == 0 or block_table.numel() == 0:
        return None
    num_rows = int(topk_2d.shape[0])
    width = int(topk_2d.reshape(num_rows, -1).shape[1])
    if num_rows <= 0 or width <= 0 or block_size <= 0:
        return None

    topk_2d = topk_2d.reshape(num_rows, width).to(
        device=block_table.device, dtype=torch.long
    )
    row_req_indices = row_req_indices[:num_rows].to(
        device=block_table.device, dtype=torch.long
    )
    if int(row_req_indices.numel()) < num_rows:
        pad = torch.full(
            (num_rows - int(row_req_indices.numel()),),
            -1,
            dtype=torch.long,
            device=block_table.device,
        )
        row_req_indices = torch.cat((row_req_indices, pad), dim=0)

    real_row_mask = row_req_indices >= 0
    if not bool(real_row_mask.any().to(device="cpu").item()):
        return None

    real_rows = real_row_mask.nonzero(as_tuple=False).flatten()
    real_req_indices = row_req_indices.index_select(0, real_rows)
    req_in_range = real_req_indices < int(block_table.shape[0])
    if not bool(req_in_range.all().to(device="cpu").item()):
        bad_idx = int((~req_in_range).nonzero(as_tuple=False)[0].item())
        sparse_row = int(real_rows[bad_idx].to(device="cpu").item())
        req_idx = int(real_req_indices[bad_idx].to(device="cpu").item())
        return {
            "reason": "req_index_out_of_range",
            "sparse_row": sparse_row,
            "req_idx": req_idx,
            "batch_size": int(block_table.shape[0]),
        }

    real_topk = topk_2d.index_select(0, real_rows)
    block_table_rows = block_table.index_select(0, real_req_indices).to(torch.long)
    num_logical_blocks = int(block_table_rows.shape[1])
    if num_logical_blocks <= 0:
        return {
            "reason": "empty_block_table",
            "batch_size": int(block_table.shape[0]),
        }

    safe_indices = torch.clamp(real_topk, min=0)
    logical_blocks = safe_indices // block_size
    logical_oob = (real_topk >= 0) & (logical_blocks >= num_logical_blocks)
    safe_logical_blocks = torch.clamp(
        logical_blocks, min=0, max=num_logical_blocks - 1
    )
    physical_blocks = block_table_rows.gather(1, safe_logical_blocks)
    bad_hits = (real_topk >= 0) & ((physical_blocks == 0) | logical_oob)
    if not bool(bad_hits.any().to(device="cpu").item()):
        return None

    flat_idx = int(bad_hits.reshape(-1).nonzero(as_tuple=False)[0].item())
    row_in_real = flat_idx // width
    col = flat_idx % width
    sparse_row = int(real_rows[row_in_real].to(device="cpu").item())
    req_idx = int(real_req_indices[row_in_real].to(device="cpu").item())
    first_topk = int(real_topk[row_in_real, col].to(device="cpu").item())
    first_logical_block = int(
        logical_blocks[row_in_real, col].to(device="cpu").item()
    )
    first_physical_block = int(
        physical_blocks[row_in_real, col].to(device="cpu").item()
    )
    return {
        "reason": "null_or_oob_block",
        "sparse_row": sparse_row,
        "req_idx": req_idx,
        "col": col,
        "topk": first_topk,
        "logical_block": first_logical_block,
        "physical_block": first_physical_block,
        "num_logical_blocks": num_logical_blocks,
        "row_req_indices_sample": _dsa_debug_sample(row_req_indices),
        "topk_row_sample": _dsa_debug_sample(real_topk[row_in_real]),
        "logical_blocks_sample": _dsa_debug_sample(logical_blocks[row_in_real]),
        "physical_blocks_sample": _dsa_debug_sample(physical_blocks[row_in_real]),
        "block_table_row_sample": _dsa_debug_sample(block_table_rows[row_in_real]),
    }


def _dsa_build_target_slot_mapping(
    block_table: torch.Tensor,
    row_req_indices: torch.Tensor,
    scratch_base: torch.Tensor,
    width: int,
    block_size: int,
) -> torch.Tensor:
    """Build per-row target slots for compact DSA scratch loads."""
    if width <= 0 or row_req_indices.numel() == 0:
        return torch.empty(
            (int(row_req_indices.numel()), max(width, 0)),
            dtype=torch.long,
            device=block_table.device,
        )

    row_req_indices = row_req_indices.to(device=block_table.device, dtype=torch.long)
    scratch_base = scratch_base.to(device=block_table.device, dtype=torch.long)
    block_table_rows = block_table.index_select(0, row_req_indices).to(torch.long)
    positions = scratch_base.reshape(-1, 1) + torch.arange(
        width, dtype=torch.long, device=block_table.device
    ).reshape(1, -1)
    logical_blocks = positions // block_size
    offsets = positions % block_size
    max_logical_block = max(int(block_table_rows.shape[1]) - 1, 0)
    safe_logical_blocks = torch.clamp(logical_blocks, min=0, max=max_logical_block)
    physical_blocks = block_table_rows.gather(1, safe_logical_blocks)
    return physical_blocks * block_size + offsets


def _dsa_validate_target_slot_mapping(
    *,
    layer_name: str | None,
    selected_tokens: torch.Tensor,
    target_slot_mapping: torch.Tensor,
    kv_cache_layer: torch.Tensor,
    row_req_indices: torch.Tensor | None,
    row_scratch_base: torch.Tensor | None,
    block_table: torch.Tensor | None,
) -> None:
    if target_slot_mapping.numel() == 0:
        return
    if kv_cache_layer.dim() < 2:
        raise RuntimeError(
            "DSA target_slot_mapping guard requires paged KV cache: "
            f"layer={layer_name} kv_shape={tuple(kv_cache_layer.shape)}"
        )
    if selected_tokens.shape != target_slot_mapping.shape:
        raise RuntimeError(
            "DSA target_slot_mapping shape mismatch: "
            f"layer={layer_name} selected_shape={tuple(selected_tokens.shape)} "
            f"target_slot_shape={tuple(target_slot_mapping.shape)}"
        )

    block_size = int(kv_cache_layer.shape[1])
    max_slots = int(kv_cache_layer.shape[0]) * block_size
    target_slots = target_slot_mapping.to(device=kv_cache_layer.device, dtype=torch.long)
    min_slot = int(target_slots.min().to(device="cpu").item())
    max_slot = int(target_slots.max().to(device="cpu").item())
    physical_blocks = target_slots // block_size
    bad_slots = (target_slots < 0) | (target_slots >= max_slots) | (physical_blocks == 0)
    if not bool(bad_slots.any().to(device="cpu").item()):
        return

    width = int(target_slots.reshape(target_slots.shape[0], -1).shape[1])
    flat_idx = int(bad_slots.reshape(-1).nonzero(as_tuple=False)[0].item())
    row = flat_idx // width
    col = flat_idx % width
    first_slot = int(target_slots.reshape(target_slots.shape[0], -1)[row, col]
                     .to(device="cpu").item())
    first_physical_block = first_slot // block_size
    raise RuntimeError(
        "DSA target_slot_mapping contains invalid write slot: "
        f"layer={layer_name} selected_shape={tuple(selected_tokens.shape)} "
        f"target_slot_shape={tuple(target_slot_mapping.shape)} "
        f"kv_shape={tuple(kv_cache_layer.shape)} block_size={block_size} "
        f"max_slots={max_slots} min_slot={min_slot} max_slot={max_slot} "
        f"first_bad_row={row} first_bad_col={col} "
        f"first_bad_slot={first_slot} "
        f"first_bad_physical_block={first_physical_block} "
        f"selected_sample={_dsa_debug_sample(selected_tokens)} "
        f"target_slot_sample={_dsa_debug_sample(target_slot_mapping)} "
        f"row_req_indices_sample={_dsa_debug_sample(row_req_indices)} "
        f"row_scratch_base_sample={_dsa_debug_sample(row_scratch_base)} "
        f"block_table_shape={tuple(block_table.shape) if block_table is not None else None}"
    )


def _dsa_indexer_layer_name(layer_name: str) -> str:
    return layer_name.rsplit(".", 1)[0] + ".indexer.k_cache"


def _dsa_index_lmcache_enabled() -> bool:
    if envs.VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE:
        return False
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return False
    connector = get_kv_transfer_group()
    return bool(getattr(connector, "supports_dsa_index_lmcache", False))


class AscendSFABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "ASCEND_SFA" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_builder_cls():
        if enable_cp():
            from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFACPMetadataBuilder

            return AscendSFACPMetadataBuilder
        return AscendSFAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_impl_cls() -> type["AscendSFAImpl"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFACPImpl

            return AscendSFACPImpl
        return AscendSFAImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


@dataclass
class DSACPContext:
    num_tokens: int
    num_tokens_pad: int
    local_start: int
    local_end: int
    local_end_with_pad: int
    slot_mapping_cp: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_key: torch.Tensor


@dataclass
class AscendSFAMetadata:
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|
    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    cum_query_lens: torch.Tensor
    block_table: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.
    # The dimension of the attention heads
    head_dim: int | None = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill
    dsa_cp_context: DSACPContext | None = None
    # DSA two-group mode: the indexer KV group's own block table / slot mapping.
    # None in single-group mode (indexer shares the latent's block ids).
    indexer_block_table: torch.Tensor | None = None
    indexer_slot_mapping: torch.Tensor | None = None
    reshape_cache_event: torch.npu.Event = None
    sfa_cp_metadata: AscendPCPMetadata | None = None
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0

    # DSA latent offload (GLM5.1): request ids and prompt lengths per request, used to
    # key LMCache and to split the indexer top-k into prefill (LMCache) vs decode
    # (resident) sources. None unless latent offload is enabled.
    # HW-VERIFY: confirm the source — req_ids/prompt_lens live on the runner's
    # input_batch, not on CommonAttentionMetadata; the runner may need to thread them
    # in (see sparse_offload/INTEGRATION.md section B).
    req_ids: list[str] | None = None
    prompt_lens: torch.Tensor | None = None
    decode_req_indices: torch.Tensor | None = None
    decode_req_indices_cpu: Any = None
    decode_valid_row_indices: torch.Tensor | None = None
    decode_valid_rows_all: bool = False
    decode_req_indices_compact: torch.Tensor | None = None
    decode_req_indices_compact_cpu: Any = None
    decode_request_ids_compact: list[str] | None = None
    decode_row_offsets: torch.Tensor | None = None
    decode_scratch_base: torch.Tensor | None = None
    decode_scratch_base_compact: torch.Tensor | None = None
    decode_target_slot_mapping: torch.Tensor | None = None
    need_sparse_lmcache_payload: bool = False


M = TypeVar("M", bound=AscendSFAMetadata)


class AscendSFAMetadataBuilder(MLACommonMetadataBuilder[AscendSFAMetadata]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls if metadata_cls is not None else AscendSFAMetadata,
            supports_dcp_with_varlen,
        )

        self.block_size = vllm_config.cache_config.block_size
        self.max_blocks = (vllm_config.model_config.max_model_len + self.block_size - 1) // self.block_size

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )
        self.reorder_batch_threshold = self.decode_threshold
        self.attn_mask_builder = AttentionMaskBuilder(self.device)
        self.rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
        self.enable_dsa_cp = enable_dsa_cp()
        self.dsa_shrink_latent = (
            int(envs.VLLM_ASCEND_DSA_SHRINK_LATENT)
            if envs.VLLM_ASCEND_DSA_UNBUNDLE
            else 0
        )
        hf_config = self.model_config.hf_config
        hf_text_config = self.model_config.hf_text_config
        self.index_topk = int(
            getattr(
                hf_text_config or hf_config,
                "topk_tokens",
                getattr(hf_text_config or hf_config, "index_topk", 2048),
            )
        )

        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.actual_seq_lengths_query = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)
        self.actual_seq_lengths_key = torch.empty_like(self.actual_seq_lengths_query)

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int:
        return ascend_chunked_prefill_workspace_size(vllm_config)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendSFAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def reorder_batch(self, input_batch: "NPUInputBatch", scheduler_output: "SchedulerOutput") -> bool:
        # No need to reorder for Ascend SFA
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendSFAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_input_tokens = common_attn_metadata.num_input_tokens

        block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        input_positions = common_attn_metadata.positions[:num_input_tokens].long()

        # DSA two-group mode: mirror the indexer group's table/slots (same
        # slicing as the latent's) so the impl can address the indexer cache.
        indexer_block_table = None
        indexer_slot_mapping = None
        if common_attn_metadata.indexer_block_table_tensor is not None:
            indexer_block_table = common_attn_metadata.indexer_block_table_tensor[:num_reqs]
            indexer_slot_mapping = common_attn_metadata.indexer_slot_mapping[:num_input_tokens]

        # DSA shrink-latent: expand per-request prompt lengths to per-ROW values
        # for scratch_remap — decode rows (position >= prompt_len) get plen,
        # prefill and padding rows get 0 (= left untouched by the remap). Works
        # for both pure-decode and mixed chunked-prefill+decode steps.
        prompt_lens_rows = None
        decode_req_indices_rows = None
        decode_valid_row_indices = None
        decode_valid_rows_all = False
        decode_req_indices_compact = None
        decode_req_indices_compact_cpu = None
        decode_request_ids_compact = None
        decode_row_offsets_rows = None
        decode_scratch_base_rows = None
        decode_scratch_base_compact = None
        decode_target_slot_mapping = None
        need_sparse_lmcache_payload = False
        num_decode_rows = 0
        plens_cpu = common_attn_metadata.prompt_lens_cpu
        if plens_cpu is not None:
            rows = np.zeros(num_input_tokens, dtype=np.int32)
            req_rows = np.full(num_input_tokens, -1, dtype=np.int32)
            row_offsets = np.zeros(num_input_tokens, dtype=np.int32)
            n_real = min(len(plens_cpu), num_reqs)
            qsl = common_attn_metadata.query_start_loc_cpu[: n_real + 1].numpy()
            computed = common_attn_metadata.num_computed_tokens_cpu[:n_real].numpy()
            for r in range(n_real):
                s, e = int(qsl[r]), int(qsl[r + 1])
                plen = int(plens_cpu[r])
                first_decode = max(s, s + plen - int(computed[r]))
                if first_decode < e:
                    count = e - first_decode
                    offsets = np.arange(count, dtype=np.int32)
                    rows[first_decode:e] = plen
                    req_rows[first_decode:e] = r
                    row_offsets[first_decode:e] = offsets
            num_decode_rows = int((rows > 0).sum())
            prompt_lens_rows = torch.from_numpy(rows).to(block_table.device)
            decode_req_indices_rows = torch.from_numpy(req_rows).to(block_table.device)
            scratch_base_np = row_offsets.astype(np.int64) * self.index_topk
            # Plain decode has one row per request and uses the legacy per-request
            # sparse slot mapping. Only MTP/spec rows need disjoint scratch bases
            # and explicit target-slot tensors.
            needs_row_scratch_base = bool(np.any(scratch_base_np))
            if needs_row_scratch_base:
                decode_row_offsets_rows = torch.from_numpy(row_offsets).to(
                    block_table.device
                )
                decode_scratch_base_rows = torch.from_numpy(scratch_base_np).to(
                    block_table.device
                )
            need_sparse_lmcache_payload = (
                self.dsa_shrink_latent != 3
                and has_kv_transfer_group()
                and is_v1_kv_transfer_group()
            )
            valid_row_indices_np = (
                np.flatnonzero(req_rows >= 0).astype(np.int64)
                if need_sparse_lmcache_payload
                else np.empty(0, dtype=np.int64)
            )
            if valid_row_indices_np.size:
                decode_valid_rows_all = int(valid_row_indices_np.size) == int(
                    num_input_tokens
                )
                valid_req_indices_np = req_rows[valid_row_indices_np].astype(np.int64)
                valid_scratch_base_np = scratch_base_np[valid_row_indices_np]
                decode_req_indices_compact_cpu = valid_req_indices_np
                req_ids = common_attn_metadata.request_ids
                if req_ids is not None:
                    decode_request_ids_compact = [
                        req_ids[int(req_idx)] for req_idx in valid_req_indices_np
                    ]
                if not decode_valid_rows_all:
                    decode_valid_row_indices = torch.from_numpy(
                        valid_row_indices_np
                    ).to(block_table.device)
                decode_req_indices_compact = torch.from_numpy(
                    valid_req_indices_np
                ).to(block_table.device)
                if needs_row_scratch_base:
                    decode_scratch_base_compact = torch.from_numpy(
                        valid_scratch_base_np
                    ).to(block_table.device)
                    decode_target_slot_mapping = _dsa_build_target_slot_mapping(
                        block_table,
                        decode_req_indices_compact,
                        decode_scratch_base_compact,
                        self.index_topk,
                        self.block_size,
                    )

        cum_query_lens = common_attn_metadata.query_start_loc[1 : num_reqs + 1]
        seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_reqs]

        cos, sin = get_cos_and_sin_mla(input_positions, True)

        dsa_cp_context = None
        if self.enable_dsa_cp:
            global_tp_size = get_tp_group().world_size
            num_tokens = num_input_tokens
            num_tokens_pad = _round_up(num_tokens, global_tp_size)
            num_tokens_per_device = num_tokens_pad // global_tp_size
            local_start = get_tp_group().rank_in_group * num_tokens_per_device
            local_end_with_pad = local_start + num_tokens_per_device
            local_end = min(local_end_with_pad, num_actual_tokens)

            pad_size = num_tokens_pad - cos.shape[0]
            assert cos.shape == sin.shape, f"cos.shape must be equal to sin.shape, got {cos.shape} and {sin.shape}"

            if pad_size > 0:
                cos = nn.functional.pad(cos, (0, 0, 0, 0, 0, 0, 0, pad_size))
                sin = nn.functional.pad(sin, (0, 0, 0, 0, 0, 0, 0, pad_size))

            pad_size_slot = num_tokens_pad - slot_mapping.shape[0]
            if pad_size_slot > 0:
                slot_mapping = nn.functional.pad(slot_mapping, (0, pad_size_slot), value=-1)
            else:
                slot_mapping = slot_mapping[:num_tokens_pad]
            slot_mapping_cp = slot_mapping[local_start:local_end_with_pad]

            cos = cos[local_start:local_end_with_pad]
            sin = sin[local_start:local_end_with_pad]

            assert cos.shape[0] == num_tokens_per_device, (
                f"cos.shape[0] must be equal to num_tokens_per_device, \
                    got {cos.shape[0]} and {num_tokens_per_device}"
            )
            assert slot_mapping_cp.shape[0] == num_tokens_per_device, (
                f"slot_mapping_cp.shape[0] must be equal to num_tokens_per_device, \
                    got {slot_mapping_cp.shape[0]} and {num_tokens_per_device}"
            )
            assert slot_mapping.shape[0] == num_tokens_pad, (
                f"slot_mapping.shape[0] must be equal to num_tokens_pad, \
                    got {slot_mapping.shape[0]} and {num_tokens_pad}"
            )

            actual_seq_lengths_query = self.actual_seq_lengths_query
            actual_seq_lengths_key = self.actual_seq_lengths_key

            num_segs = cum_query_lens.shape[0]
            last_token = 0
            cum = 0
            for i in range(0, num_segs):
                global_start = last_token
                global_end = cum_query_lens[i].item()
                last_token = global_end

                req_local_start = max(global_start, local_start)
                req_local_end = min(global_end, local_end_with_pad)
                num_local_tokens = req_local_end - req_local_start

                if num_local_tokens > 0:
                    cum += num_local_tokens
                    actual_seq_lengths_query[i] = cum

                    offset = global_end - req_local_end
                    actual_seq_lengths_key[i] = seq_lens[i].item() - offset
                else:
                    actual_seq_lengths_query[i] = cum
                    actual_seq_lengths_key[i] = 0

            actual_seq_lengths_query = actual_seq_lengths_query[:num_reqs]
            actual_seq_lengths_key = actual_seq_lengths_key[:num_reqs]

            dsa_cp_context = DSACPContext(
                num_tokens=num_tokens,
                num_tokens_pad=num_tokens_pad,
                local_start=local_start,
                local_end=local_end,
                local_end_with_pad=local_end_with_pad,
                slot_mapping_cp=slot_mapping_cp,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
            )

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=num_actual_tokens,
            cum_query_lens=cum_query_lens,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            slot_mapping=slot_mapping,
            head_dim=self.model_config.get_head_size(),
            attn_mask=self.attn_mask_builder.get_attention_mask(self.model_config),
            attn_state=common_attn_metadata.attn_state,
            block_table=block_table,
            sin=sin[:num_input_tokens],
            cos=cos[:num_input_tokens],
            dsa_cp_context=dsa_cp_context,
            indexer_block_table=indexer_block_table,
            indexer_slot_mapping=indexer_slot_mapping,
            # DSA latent offload: best-effort; getattr -> None when not threaded in yet
            # (harmless unless the feature is enabled). HW-VERIFY the real source.
            req_ids=getattr(common_attn_metadata, "request_ids", None),
            prompt_lens=prompt_lens_rows,
            decode_req_indices=decode_req_indices_rows,
            decode_req_indices_cpu=req_rows if decode_req_indices_rows is not None else None,
            decode_valid_row_indices=decode_valid_row_indices,
            decode_valid_rows_all=decode_valid_rows_all,
            decode_req_indices_compact=decode_req_indices_compact,
            decode_req_indices_compact_cpu=decode_req_indices_compact_cpu,
            decode_request_ids_compact=decode_request_ids_compact,
            decode_row_offsets=decode_row_offsets_rows,
            decode_scratch_base=decode_scratch_base_rows,
            decode_scratch_base_compact=decode_scratch_base_compact,
            decode_target_slot_mapping=decode_target_slot_mapping,
            need_sparse_lmcache_payload=need_sparse_lmcache_payload,
            num_decode_tokens=num_decode_rows,
        )

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in {AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding}:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError("Currently we only support building dummy metadata for DecodeOnly state")

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendSFAImpl(MLAAttentionImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # Supports forward using the all-gather o_proj weight for decode requests when Sharded CP is enabled.
    o_proj_full_pool: torch.Tensor | None = None

    # q_hadamard and k_hadamard tensor shared when dsa c8 enabled
    q_hadamard: torch.Tensor | None = None
    k_hadamard: torch.Tensor | None = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # MLA Args
        self.q_lora_rank = kwargs["q_lora_rank"]
        self.kv_lora_rank = kwargs["kv_lora_rank"]
        self.qk_nope_head_dim = kwargs["qk_nope_head_dim"]
        self.qk_rope_head_dim = kwargs["qk_rope_head_dim"]
        self.qk_head_dim = kwargs["qk_head_dim"]
        self.v_head_dim = kwargs["v_head_dim"]
        self.rotary_emb = kwargs["rotary_emb"]
        self.q_proj = kwargs["q_proj"] if self.q_lora_rank is None else kwargs["q_b_proj"]
        self.fused_qkv_a_proj = kwargs.get("fused_qkv_a_proj")
        self.kv_b_proj = kwargs["kv_b_proj"]
        self.o_proj = kwargs["o_proj"]
        self.indexer = kwargs["indexer"]
        self.kv_a_proj_with_mqa = kwargs.get("kv_a_proj_with_mqa")
        self.kv_a_layernorm = kwargs.get("kv_a_layernorm")
        self.q_a_layernorm = kwargs.get("q_a_layernorm")
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tp_group().rank_in_group
        self.q_b_proj = kwargs["q_b_proj"]
        self._dsa_indexer_diag_seen: set[tuple[str, str]] = set()

        ascend_config = get_ascend_config()
        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp

        # The MLAPO operator fuses the pre-processing steps on Q/K/V in MLA into a single operator
        # NOTE: it imposes a limit on the number of input tokens and conflicts with FlashComm
        self.enable_mlapo = envs.VLLM_ASCEND_ENABLE_MLAPO

        assert self.indexer is not None, "Indexer is required for DSA."

        self.local_num_heads = self.num_heads
        self.vllm_config = get_current_vllm_config()
        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )

        # indexer param
        self.n_head: int = self.indexer.n_head  # 64
        self.head_dim: int = self.indexer.head_dim  # 128
        hf_config = self.vllm_config.model_config.hf_config
        hf_text_config = getattr(self.vllm_config.model_config, "hf_text_config", None)
        self.index_topk = int(
            getattr(
                self.indexer,
                "topk_tokens",
                getattr(hf_text_config or hf_config, "index_topk", 2048),
            )
        )
        self.wq_b = self.indexer.wq_b
        self.wk = self.indexer.wk
        self.weights_proj = self.indexer.weights_proj
        self.k_norm = self.indexer.k_norm
        self.cp_size = 1
        self.is_rope_neox_style = True
        self.use_torch_npu_lightning_indexer = False
        if self.vllm_config.model_config.hf_config.model_type in ["glm_moe_dsa"]:
            self.is_rope_neox_style = False
            self.use_torch_npu_lightning_indexer = True

        # DSA latent offload Route-1 pragmatic (M-B): latent written to the
        # PagedLatentPool instead of the (shrunk) vLLM paged latent cache.
        self.dsa_offload_free_paged = bool(
            envs.VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD
            and envs.VLLM_ASCEND_DSA_OFFLOAD_FREE_PAGED
        )
        self.dsa_offload_unbundle = bool(envs.VLLM_ASCEND_DSA_UNBUNDLE)
        # Step B staging (1 = B2 compact-scratch read; 2 = +B1 freeing).
        self.dsa_shrink_latent = (
            int(envs.VLLM_ASCEND_DSA_SHRINK_LATENT) if self.dsa_offload_unbundle else 0
        )
        # dsa c8
        self.use_sparse_c8_indexer = ascend_config.enable_sparse_c8
        if self.use_sparse_c8_indexer:
            self.c8_k_cache_dtype = torch.int8
            self.c8_k_scale_cache_dtype = torch.float16

        # Effective in SFA when FlashComm is enabled.
        self.enable_dsa_cp = enable_dsa_cp()

        # Enable layer sharding via DSA-CP on the P node in the PD-disaggregated setup.
        self.enable_dsa_cp_with_layer_shard = enable_dsa_cp_with_layer_shard()

        # Improves glm5 accuracy after enabling dsa-cp in scenarios with strict accuracy requirements,
        # especially for customized cases, at the cost of performance degradation due to extra communication.
        self.enable_dsa_cp_strict_accuracy = (
            self.enable_dsa_cp_with_layer_shard
            and self.vllm_config.model_config.hf_config.model_type in ["glm_moe_dsa"]
        )

        # use original TP o_proj weight in PD mix stage, and full gather
        # for o_proj weight for prefill stage.
        self.enable_dsa_cp_with_o_proj_tp = enable_dsa_cp_with_o_proj_tp()

        if self.enable_dsa_cp:
            self.local_num_heads = self.num_heads * self.tp_size
            if self.enable_dsa_cp_with_layer_shard:
                self.layer_sharding_kwargs = []
                for layer_name in get_ascend_config().layer_sharding or []:
                    if layer_name in kwargs:
                        self.layer_sharding_kwargs.append(kwargs[layer_name])
                    else:
                        logger.warning_once(
                            f"[SFAImpl init] Layer '{layer_name}' not found in kwargs for layer sharding, "
                            "skipping sharding configuration"
                        )
                register_all_layers_to_shard_weight_series(self.layer_sharding_kwargs)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # NOTE: We currently do not support quant kv_b_proj.
        assert isinstance(self.kv_b_proj.quant_method, UnquantizedLinearMethod)
        # NOTE: Weight will be reshaped next, we need to revert and transpose it.
        kv_b_proj_weight = torch_npu.npu_format_cast(self.kv_b_proj.weight.data, ACL_FORMAT_FRACTAL_ND).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.local_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.local_num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}"
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.local_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # NOTE: When we make a incontiguous weight contiguous, a new address will be allocated for the weight,
        # in graph + RL scenario, we only capture the graph once, and the weight address is expected to be the same
        # across iterations, so we need to copy the weight to the original address after making it contiguous.
        if not hasattr(self, "W_UV"):
            # Convert from (L, N, V) to (N, L, V)
            self.W_UV = W_UV.transpose(0, 1).contiguous()
            # Convert from (L, N, P) to (N, P, L)
            self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()
        else:
            self.W_UV.copy_(W_UV.transpose(0, 1).contiguous())
            self.W_UK_T.copy_(W_UK.permute(1, 2, 0).contiguous())

        # TODO(zzzzwwjj): Currently, torch.ops._C_ascend.batch_matmul_transpose cannot support weight nz
        # self.W_UV = maybe_trans_nz(self.W_UV)

        # Dispose kv_b_proj since it is replaced by W_UV and W_UK_T to save memory
        dispose_layer(self.kv_b_proj)
        if self.enable_dsa_cp:
            if self.enable_dsa_cp_with_layer_shard:
                for layer in self.layer_sharding_kwargs or []:
                    if is_hidden_layer(layer):
                        post_process_after_loading_for_shard_weight_series(layer)
            else:
                self._init_o_proj_tp_full_params()

        if self.enable_mlapo:
            quant_method = getattr(
                getattr(self.fused_qkv_a_proj, "quant_method", None),
                "quant_method",
                None,
            )
            reasons = []
            if self.fused_qkv_a_proj is None or not isinstance(quant_method, AscendW8A8LinearMethod):
                reasons.append(
                    "Currently mlapo only supports W8A8 quantization in SFA scenario."
                    "Some layers in your model are not quantized with W8A8,"
                    "thus mlapo is disabled for these layers."
                )
            if self.enable_dsa_cp:
                reasons.append("Currently mlapo does not support SFA with CP,thus mlapo is disabled for these layers.")
            if reasons:
                self.enable_mlapo = False
                for msg in reasons:
                    logger.warning_once(msg)
            else:
                self._process_weights_for_fused_mlapo(act_dtype)
        if not self.enable_mlapo:
            # if mlapo, W_UK_T can't trans nz
            self.W_UK_T = maybe_trans_nz(self.W_UK_T)

        if self.use_sparse_c8_indexer and AscendSFAImpl.q_hadamard is None:
            AscendSFAImpl.q_hadamard = torch.tensor(scipy.linalg.hadamard(128), dtype=torch.bfloat16, device="npu") / (
                128**0.5
            )
        if self.use_sparse_c8_indexer and AscendSFAImpl.k_hadamard is None:
            AscendSFAImpl.k_hadamard = torch.tensor(scipy.linalg.hadamard(128), dtype=torch.bfloat16, device="npu") / (
                128**0.5
            )

    # Processing the input parameters for MLAPO by reordering and transposing
    # QKV(and part of Q) weight, applying RoPE-related dimension transformations,
    # and handling quantization parameters.
    def _process_weights_for_fused_mlapo(self, act_dtype: torch.dtype):
        assert self.kv_a_proj_with_mqa is None
        assert self.fused_qkv_a_proj is not None

        kv_a_proj_wt = self.fused_qkv_a_proj.weight.data[..., self.q_lora_rank :].contiguous()
        q_a_proj_wt = self.fused_qkv_a_proj.weight.data[..., : self.q_lora_rank].contiguous()

        kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
        kv_a_proj_wt = trans_rope_weight(kv_a_proj_wt, self.qk_rope_head_dim)
        kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
        wd_qkv = torch.cat((kv_a_proj_wt, q_a_proj_wt), dim=-1)
        wd_qkv = wd_qkv.t().contiguous()
        wd_qkv = transdata(wd_qkv, block_size=(16, 32)).unsqueeze(0).contiguous()
        self.wd_qkv = torch_npu.npu_format_cast(wd_qkv, 29)

        kv_a_proj_deq_scl = self.fused_qkv_a_proj.deq_scale[self.q_lora_rank :].contiguous()
        q_a_proj_deq_scl = self.fused_qkv_a_proj.deq_scale[: self.q_lora_rank].contiguous()
        kv_a_proj_deq_scl = kv_a_proj_deq_scl.reshape(self.kv_lora_rank + self.qk_rope_head_dim, -1).contiguous()
        kv_a_proj_deq_scl = trans_rope_weight(kv_a_proj_deq_scl, self.qk_rope_head_dim)
        kv_a_proj_deq_scl = kv_a_proj_deq_scl.view(self.kv_lora_rank + self.qk_rope_head_dim).contiguous()
        self.deq_scale_qkv = torch.cat((kv_a_proj_deq_scl, q_a_proj_deq_scl), dim=-1).contiguous()

        kv_a_proj_qt_bias = self.fused_qkv_a_proj.quant_bias[self.q_lora_rank :].contiguous()
        q_a_proj_qt_bias = self.fused_qkv_a_proj.quant_bias[: self.q_lora_rank].contiguous()

        kv_a_proj_qt_bias = kv_a_proj_qt_bias.reshape(self.kv_lora_rank + self.qk_rope_head_dim, -1).contiguous()
        kv_a_proj_qt_bias = trans_rope_weight(kv_a_proj_qt_bias, self.qk_rope_head_dim)
        kv_a_proj_qt_bias = kv_a_proj_qt_bias.view(self.kv_lora_rank + self.qk_rope_head_dim).contiguous()
        self.quant_bias_qkv = torch.cat((kv_a_proj_qt_bias, q_a_proj_qt_bias), dim=-1).contiguous()

        wu_q = self.q_proj.weight.data
        wu_q = wu_q.t().reshape(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        wu_q = trans_rope_weight(wu_q, self.qk_rope_head_dim)
        wu_q = wu_q.reshape(self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), -1)
        wu_q = transdata(wu_q, block_size=(16, 32)).unsqueeze(0).contiguous()
        self.wu_q = torch_npu.npu_format_cast(wu_q, 29)

        qb_deq_scl = self.q_proj.deq_scale.data
        qb_deq_scl = qb_deq_scl.reshape(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        qb_deq_scl = trans_rope_weight(qb_deq_scl, self.qk_rope_head_dim)
        self.qb_deq_scl = qb_deq_scl.reshape(self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim))

        qb_qt_bias = self.q_proj.quant_bias.data
        qb_qt_bias = qb_qt_bias.reshape(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        qb_qt_bias = trans_rope_weight(qb_qt_bias, self.qk_rope_head_dim)
        self.qb_qt_bias = qb_qt_bias.reshape(self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim))

        device = self.q_proj.weight.device
        self.gamma1 = self.q_a_layernorm.weight.data  # type: ignore[union-attr]
        self.beta1 = self.q_a_layernorm.bias.data  # type: ignore[union-attr]
        self.gamma2 = self.kv_a_layernorm.weight.data  # type: ignore[union-attr]
        self.quant_scale0 = self.fused_qkv_a_proj.input_scale.data
        self.quant_offset0 = self.fused_qkv_a_proj.input_offset.data
        self.quant_scale1 = self.q_proj.input_scale.data
        self.quant_offset1 = self.q_proj.input_offset.data
        self.ctkv_scale = torch.tensor([1], dtype=act_dtype, device=device)
        self.q_nope_scale = torch.tensor([1], dtype=act_dtype, device=device)

        # On KV consumers (decode-only) MLAPO uses the transformed weights built above;
        # the original fused_qkv_a_proj/q_proj weights and quant params are no longer
        # referenced, so drop them to save memory.
        if (
            self.vllm_config.kv_transfer_config is not None
            and self.vllm_config.kv_transfer_config.is_kv_consumer
            and self.vllm_config.scheduler_config.max_num_batched_tokens <= MLAPO_MAX_SUPPORTED_TOKENS
        ):
            self.fused_qkv_a_proj.weight = None
            self.fused_qkv_a_proj.deq_scale = None
            self.fused_qkv_a_proj.quant_bias = None
            self.q_proj.weight = None
            self.q_proj.deq_scale = None
            self.q_proj.quant_bias = None
            torch.npu.empty_cache()

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        raise NotImplementedError("forward_mha is not supported for SFA attention. Use forward() instead.")

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError("forward_mqa is not supported for SFA attention. Use forward() instead.")

    def rope_single(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape
        S = 1
        x = x.view(B, N, S, D)
        x = torch_npu.npu_interleave_rope(x, cos, sin)
        return x.view(B, N, D)

    def _init_o_proj_tp_full_params(self):
        """
        Initialize TP-mode and Full-mode parameters for o_proj weight,
        preparing for weight switching in PD mix stage.

        For PD mix stage:
        - Use original TP o_proj weight for decode phase
        - Need full-gather o_proj weight from all TP ranks for prefill phase
        """
        if AscendSFAImpl.o_proj_full_pool is None:
            sample = self.o_proj.weight
            AscendSFAImpl.o_proj_full_pool = torch.empty(
                (sample.shape[0] * self.tp_size, sample.shape[1]), dtype=sample.dtype, device=sample.device
            )

        # Save TP-mode parameters (original sharded weights)
        self.o_proj_tp_weight = self.o_proj.weight.clone().detach()
        self.o_proj_tp_aclnn_input_scale = self.o_proj.aclnn_input_scale.clone().detach()
        self.o_proj_tp_aclnn_input_scale_reciprocal = self.o_proj.aclnn_input_scale_reciprocal.clone().detach()
        self.o_proj_tp_aclnn_input_offset = self.o_proj.aclnn_input_offset.clone().detach()

        # Initially switch to TP mode for graph capture
        self.o_proj.weight.set_(self.o_proj_tp_weight)
        self.o_proj.aclnn_input_scale.set_(self.o_proj_tp_aclnn_input_scale)
        self.o_proj.aclnn_input_scale_reciprocal.set_(self.o_proj_tp_aclnn_input_scale_reciprocal)
        self.o_proj.aclnn_input_offset.set_(self.o_proj_tp_aclnn_input_offset)

        # Precompute Full-mode quantization parameters by repeating TP parameters across all TP ranks
        self.o_proj_full_aclnn_input_scale = self.o_proj.aclnn_input_scale.repeat(self.tp_size)
        self.o_proj_full_aclnn_input_scale_reciprocal = self.o_proj.aclnn_input_scale_reciprocal.repeat(self.tp_size)
        self.o_proj_full_aclnn_input_offset = self.o_proj.aclnn_input_offset.repeat(self.tp_size)

    def _handle_o_proj_weight_switch_and_forward(
        self,
        attn_output: torch.Tensor,
        output: torch.Tensor,
        o_proj_full_handle: torch.distributed.Work | None,
        should_shard_weight: bool,
    ) -> tuple[torch.Tensor, bool]:
        """
        Handle o_proj weight switching between TP-mode and Full-mode, and execute forward computation.
        """
        # Gather o_proj weight from all TP ranks for Full-mode computation
        if should_shard_weight:
            # Wait for the completion of o_proj weight all-gather operation
            if o_proj_full_handle is not None:
                o_proj_full_handle.wait()

            # Switch o_proj to Full-mode (gathered weight from all TP ranks)
            self.o_proj.weight.set_(AscendSFAImpl.o_proj_full_pool)
            self.o_proj.aclnn_input_scale.set_(self.o_proj_full_aclnn_input_scale)
            self.o_proj.aclnn_input_scale_reciprocal.set_(self.o_proj_full_aclnn_input_scale_reciprocal)
            self.o_proj.aclnn_input_offset.set_(self.o_proj_full_aclnn_input_offset)

            # Apply quantization method and execute forward computation
            output[...] = self.o_proj.quant_method.quant_method.apply(self.o_proj, attn_output)

            # Switch o_proj back to TP-mode for subsequent decode operations
            self.o_proj.weight.set_(self.o_proj_tp_weight)
            self.o_proj.aclnn_input_scale.set_(self.o_proj_tp_aclnn_input_scale)
            self.o_proj.aclnn_input_scale_reciprocal.set_(self.o_proj_tp_aclnn_input_scale_reciprocal)
            self.o_proj.aclnn_input_offset.set_(self.o_proj_tp_aclnn_input_offset)

            return output, False
        else:
            # For decode scenario: perform all-to-all communication on o_proj input activations
            # Reshape for all-to-all: [batch * seq, tp_size, head_dim] -> [tp_size, batch * seq, head_dim]
            send = (
                attn_output.view(-1, self.tp_size, self.num_heads * self.v_head_dim)
                .permute(1, 0, 2)
                .reshape(-1, self.num_heads * self.v_head_dim)
            )

            attn_output = torch.empty_like(send)
            torch.distributed.all_to_all_single(attn_output, send, group=get_tp_group().device_group)

            return attn_output, True

    def _get_full_kv(self, k, attn_metadata):
        return k

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata: M,
    ):
        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv_no_split = kv_no_split.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA"

        if self.enable_dsa_cp:
            _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv_no_split,
                self.kv_a_layernorm.weight,  # type: ignore[union-attr]
                cos,
                sin,
                slots.to(torch.int64),
                kv_cache[1],
                kv_cache[0],
                epsilon=self.kv_a_layernorm.variance_epsilon,  # type: ignore[union-attr]
                cache_mode=cache_mode,
                is_output_kv=True,
            )
            return k_pe, k_nope
        else:
            # is_output_kv=True returns the freshly-computed latent (k_pe, k_nope) in
            # addition to caching it, so the DSA-offload forward can store/gather it
            # without a paged read-back. The op still caches to kv_cache[0]/[1] here;
            # suppressing that paged write is Stage2-B step 10b. Returning the latent is
            # harmless to the non-offload path (it ignores the returns).
            _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv_no_split,
                self.kv_a_layernorm.weight,  # type: ignore[union-attr]
                cos,
                sin,
                slots.to(torch.int64),
                kv_cache[1],
                kv_cache[0],
                epsilon=self.kv_a_layernorm.variance_epsilon,  # type: ignore[union-attr]
                cache_mode=cache_mode,
                is_output_kv=True,
            )
            return k_pe, k_nope

    # Return `ql_nope`, `q_pe`
    def _q_proj_and_k_up_proj(self, x):
        q_nope, q_pe = (
            self.q_proj(x)[0]
            .view(-1, self.local_num_heads, self.qk_head_dim)
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        )

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        return ql_nope.transpose(0, 1), q_pe

    def _v_up_proj(self, x):
        num_input_tokens, _, _ = x.shape
        if (
            x.dtype in [torch.float16, torch.bfloat16]
            and hasattr(torch.ops._C_ascend, "batch_matmul_transpose")
            and num_input_tokens <= BMM_TRANS_MAX_SUPPORTED_TOKENS
        ):
            x = x.view(-1, self.local_num_heads, self.kv_lora_rank)
            res = torch.empty((num_input_tokens, self.local_num_heads, self.v_head_dim), dtype=x.dtype, device=x.device)
            torch.ops._C_ascend.batch_matmul_transpose(x, self.W_UV, res)
            x = res.reshape(-1, self.local_num_heads * self.v_head_dim)
        else:
            # Convert from (B, N, L) to (N, B, L)
            x = x.view(-1, self.local_num_heads, self.kv_lora_rank).transpose(0, 1)
            # # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
            x = torch.bmm(x, self.W_UV)
            # # Convert from (N, B, V) to (B, N * V)
            x = x.transpose(0, 1).reshape(-1, self.local_num_heads * self.v_head_dim)
        return x

    def _sfa_preprocess_with_mlapo(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_input_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k_nope, k_pe = kv_cache[0], kv_cache[1]
        ql_nope = torch.empty(
            (num_input_tokens, self.W_UK_T.shape[0], k_nope.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_pe = torch.empty(
            (num_input_tokens, self.W_UK_T.shape[0], k_pe.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_c = torch.empty(
            (num_input_tokens, self.q_lora_rank),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops._C_ascend.mla_preprocess(
            hidden_states,
            self.wd_qkv,
            self.deq_scale_qkv,
            self.gamma1,
            self.beta1,
            self.wu_q,
            self.qb_deq_scl,
            self.gamma2,
            cos,
            sin,
            self.W_UK_T,
            k_nope,
            k_pe,
            slot_mapping,
            quant_scale0=self.quant_scale0,
            quant_offset0=self.quant_offset0,
            bias0=self.quant_bias_qkv,
            quant_scale1=self.quant_scale1,
            quant_offset1=self.quant_offset1,
            bias1=self.qb_qt_bias,
            ctkv_scale=self.ctkv_scale,
            q_nope_scale=self.q_nope_scale,
            cache_mode="krope_ctkv",
            quant_mode="per_tensor_quant_asymm",
            enable_inner_out=True,
            q_out0=ql_nope,
            kv_cache_out0=k_nope,
            q_out1=q_pe,
            kv_cache_out1=k_pe,
            inner_out=q_c,
        )
        return hidden_states, ql_nope, q_pe, q_c

    def indexer_select_pre_process(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        k_li, _ = self.wk(x)  # [b,s,7168] @ [7168,128] = [b,s,128]
        k_li = self.k_norm(k_li).unsqueeze(1)
        k_li = k_li.view(-1, 1, self.head_dim)

        if HAS_TRITON:
            cos = cos.view(-1, self.qk_rope_head_dim)
            sin = sin.view(-1, self.qk_rope_head_dim)
            k_li = rope_forward_triton_siso(
                k_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        else:
            k_li_pe, k_li_nope = torch.split(
                k_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )

            cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
            sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

            k_li_pe = k_li_pe.unsqueeze(2)
            k_li_pe = torch_npu.npu_rotary_mul(k_li_pe, cos, sin)
            k_li_pe = k_li_pe.squeeze(2)

            k_li = torch.cat([k_li_pe, k_li_nope], dim=-1)  # [b*s,128]

        if self.use_sparse_c8_indexer:
            k_li = k_li @ AscendSFAImpl.k_hadamard
            k_li, k_li_scale = torch_npu.npu_dynamic_quant(k_li.view(-1, self.head_dim), dst_type=self.c8_k_cache_dtype)
            k_li_scale = k_li_scale.to(self.c8_k_scale_cache_dtype)  # [b*s,]
            k_li_scale = k_li_scale.unsqueeze(-1)  # [b*s,1]
        else:
            k_li_scale = None

        return k_li, k_li_scale

    def _dsa_indexer_diag_req_ids(self, attn_metadata: M) -> list[str]:
        req_ids = getattr(attn_metadata, "decode_request_ids_compact", None)
        if not req_ids:
            req_ids = getattr(attn_metadata, "req_ids", None)
        if not req_ids:
            return ["unknown"]
        return [str(req_id) for req_id in req_ids]

    def _dsa_indexer_diag_signature(
        self,
        *,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        block_size: int,
        digest_values: bool,
    ) -> str:
        h = hashlib.blake2b(digest_size=8)
        for name, tensor in (
            ("q", actual_seq_lengths_query),
            ("k", actual_seq_lengths_key),
        ):
            h.update(name.encode("utf-8"))
            if digest_values:
                h.update(_dsa_diag_tensor_digest(tensor).encode("utf-8"))
            else:
                h.update(str(tuple(int(dim) for dim in tensor.shape)).encode("utf-8"))
                h.update(str(tensor.dtype).encode("utf-8"))
                h.update(str(tensor.device).encode("utf-8"))
        h.update(str(self.index_topk).encode("utf-8"))
        h.update(str(block_size).encode("utf-8"))
        return h.hexdigest()

    def _dsa_indexer_diag_run(
        self,
        req_ids: list[str],
        signature: str,
    ) -> int:
        primary_req_id = req_ids[0] if req_ids else "unknown"
        cached = _DSA_INDEXER_DIAG_REQ_RUNS.get(primary_req_id)
        if cached is not None and cached[0] == signature:
            return cached[1]
        run = _DSA_INDEXER_DIAG_RUNS.get(signature, 0) + 1
        _DSA_INDEXER_DIAG_RUNS[signature] = run
        _DSA_INDEXER_DIAG_REQ_RUNS[primary_req_id] = (signature, run)
        return run

    def _dsa_indexer_diag_active_rows(
        self,
        tensor: torch.Tensor | None,
        block_table: torch.Tensor | None,
        actual_seq_lengths_key: torch.Tensor,
        block_size: int,
        max_tokens: int | None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        meta: dict[str, Any] = {
            "block_size": int(block_size),
            "max_tokens": max_tokens,
            "sampled_tokens": 0,
            "truncated": False,
            "errors": [],
        }
        if tensor is None:
            meta["errors"].append("missing_tensor")
            return None, meta
        if block_table is None:
            meta["errors"].append("missing_block_table")
            return None, meta
        if block_size <= 0:
            meta["errors"].append("invalid_block_size")
            return None, meta

        try:
            seq_lens = (
                actual_seq_lengths_key.detach()
                .to(device="cpu", dtype=torch.long)
                .reshape(-1)
                .tolist()
            )
            block_table_cpu = block_table.detach().to(device="cpu", dtype=torch.long)
            max_reqs = min(len(seq_lens), int(block_table_cpu.shape[0]))
            rows: list[torch.Tensor] = []
            request_records: list[dict[str, Any]] = []
            remaining_tokens = max_tokens
            for req_idx in range(max_reqs):
                seq_len = max(0, int(seq_lens[req_idx]))
                if remaining_tokens is not None and remaining_tokens <= 0:
                    meta["truncated"] = True
                    request_records.append(
                        {
                            "req_index": req_idx,
                            "seq_len": seq_len,
                            "sample_len": 0,
                            "num_blocks": 0,
                            "block_head": [],
                        }
                    )
                    continue
                sample_len = seq_len
                if remaining_tokens is not None:
                    sample_len = min(sample_len, int(remaining_tokens))
                    if sample_len < seq_len:
                        meta["truncated"] = True
                num_blocks = (sample_len + int(block_size) - 1) // int(block_size)
                phys_blocks = block_table_cpu[req_idx, :num_blocks].reshape(-1)
                valid = (
                    (phys_blocks >= 0)
                    & (phys_blocks < int(tensor.shape[0]))
                )
                if not bool(valid.all().item()):
                    bad = phys_blocks[~valid][:16].tolist()
                    meta["errors"].append(
                        f"req={req_idx} invalid_blocks={bad}"
                    )
                    phys_blocks = phys_blocks[valid]
                request_records.append(
                    {
                        "req_index": req_idx,
                        "seq_len": seq_len,
                        "sample_len": sample_len,
                        "num_blocks": int(num_blocks),
                        "block_head": phys_blocks[:16].tolist(),
                    }
                )
                if sample_len <= 0 or int(phys_blocks.numel()) == 0:
                    continue
                phys_device = phys_blocks.to(
                    device=tensor.device,
                    dtype=torch.long,
                )
                gathered = tensor.index_select(0, phys_device)
                flat = gathered.reshape(-1, *tensor.shape[2:])[:sample_len]
                rows.append(flat.detach().to(device="cpu").contiguous())
                meta["sampled_tokens"] += int(sample_len)
                if remaining_tokens is not None:
                    remaining_tokens -= int(sample_len)
            meta["requests"] = request_records
            if not rows:
                return torch.empty(0, dtype=tensor.dtype), meta
            return torch.cat(rows, dim=0).contiguous(), meta
        except Exception as exc:
            meta["errors"].append(f"{type(exc).__name__}: {exc}")
            return None, meta

    @staticmethod
    def _dsa_indexer_diag_cpu_tensor(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu").contiguous()
        return value

    def _dsa_indexer_diag_log(
        self,
        *,
        layer_name: str | None,
        x: torch.Tensor,
        q_c: torch.Tensor,
        q_li: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        indexer_block_table: torch.Tensor,
        attn_metadata: M,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        topk_indices: torch.Tensor,
        key_dequant_scale: torch.Tensor | None = None,
        query_dequant_scale: torch.Tensor | None = None,
    ) -> None:
        if not _DSA_INDEXER_DIAG:
            return
        if int(getattr(attn_metadata, "num_decode_tokens", 0)) <= 0:
            return
        layer_id = _dsa_diag_layer_id(layer_name)
        if _DSA_INDEXER_DIAG_LAYERS >= 0 and layer_id >= _DSA_INDEXER_DIAG_LAYERS:
            return

        req_ids = self._dsa_indexer_diag_req_ids(attn_metadata)
        seen_key = (req_ids[0], str(layer_name))
        if seen_key in self._dsa_indexer_diag_seen:
            return
        self._dsa_indexer_diag_seen.add(seen_key)

        if _DSA_INDEXER_DIAG_SLEEP_MS > 0:
            time.sleep(_DSA_INDEXER_DIAG_SLEEP_MS / 1000.0)

        do_signature_values = _dsa_indexer_diag_component("signature")
        do_query = _dsa_indexer_diag_component("query")
        do_meta = _dsa_indexer_diag_component("meta")
        do_key_active = _dsa_indexer_diag_component("key_active")
        do_topk = _dsa_indexer_diag_component("topk")
        do_dump = _DSA_INDEXER_DIAG_DUMP and _dsa_indexer_diag_component("dump")
        do_file = (
            _dsa_indexer_diag_component("file")
            or _dsa_indexer_diag_component("compare")
            or do_dump
        )
        do_compare = _dsa_indexer_diag_component("compare")
        do_log = bool(_DSA_INDEXER_DIAG_COMPONENTS)

        if do_file:
            os.makedirs(_DSA_INDEXER_DIAG_DUMP_DIR, exist_ok=True)
        key_tensor = kv_cache[2] if len(kv_cache) > 2 else None
        block_size = (
            int(key_tensor.shape[1])
            if isinstance(key_tensor, torch.Tensor) and key_tensor.dim() >= 2
            else -1
        )
        max_digest_numel = (
            None
            if _DSA_INDEXER_DIAG_EXACT_VALUES
            else _DSA_INDEXER_DIAG_MAX_DIGEST_NUMEL
        )
        max_key_tokens = (
            None
            if _DSA_INDEXER_DIAG_EXACT_VALUES
            else _DSA_INDEXER_DIAG_KEY_SAMPLE_TOKENS
        )
        signature = self._dsa_indexer_diag_signature(
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_size=block_size,
            digest_values=do_signature_values,
        )
        run = self._dsa_indexer_diag_run(req_ids, signature)

        key_active = None
        key_active_meta: dict[str, Any] | None = None
        key_scale_active = None
        key_scale_active_meta: dict[str, Any] | None = None
        if do_key_active:
            key_active, key_active_meta = self._dsa_indexer_diag_active_rows(
                key_tensor,
                indexer_block_table,
                actual_seq_lengths_key,
                block_size,
                max_key_tokens,
            )
            if key_dequant_scale is not None:
                key_scale_active, key_scale_active_meta = (
                    self._dsa_indexer_diag_active_rows(
                        key_dequant_scale,
                        indexer_block_table,
                        actual_seq_lengths_key,
                        block_size,
                        max_key_tokens,
                    )
                )

        input_summaries = {}
        output_summaries = {}
        if do_query:
            input_summaries.update(
                {
                    "x": _dsa_diag_tensor_summary(
                        x, max_digest_numel=max_digest_numel
                    ),
                    "q_c": _dsa_diag_tensor_summary(
                        q_c, max_digest_numel=max_digest_numel
                    ),
                    "q_li": _dsa_diag_tensor_summary(
                        q_li, max_digest_numel=max_digest_numel
                    ),
                    "weights": _dsa_diag_tensor_summary(
                        weights, max_digest_numel=max_digest_numel
                    ),
                    "query_dequant_scale": _dsa_diag_tensor_summary(
                        query_dequant_scale, max_digest_numel=max_digest_numel
                    ),
                }
            )
        if do_key_active:
            input_summaries.update(
                {
                    # The full key tensors are paged allocation pools. Copying/hashing
                    # them is prohibitively slow and includes slots the first token will
                    # never read. Compare key_active instead.
                    "key": _dsa_diag_tensor_summary(
                        key_tensor, digest_values=False
                    ),
                    "key_active": _dsa_diag_tensor_summary(
                        key_active, max_digest_numel=max_digest_numel
                    ),
                    "key_dequant_scale": _dsa_diag_tensor_summary(
                        key_dequant_scale, digest_values=False
                    ),
                    "key_dequant_scale_active": _dsa_diag_tensor_summary(
                        key_scale_active, max_digest_numel=max_digest_numel
                    ),
                }
            )
        if do_meta:
            input_summaries.update(
                {
                    "indexer_block_table": _dsa_diag_tensor_summary(
                        indexer_block_table, max_digest_numel=max_digest_numel
                    ),
                    "indexer_slot_mapping": _dsa_diag_tensor_summary(
                        getattr(attn_metadata, "indexer_slot_mapping", None),
                        max_digest_numel=max_digest_numel,
                    ),
                    "slot_mapping": _dsa_diag_tensor_summary(
                        getattr(attn_metadata, "slot_mapping", None),
                        max_digest_numel=max_digest_numel,
                    ),
                    "actual_seq_lengths_query": _dsa_diag_tensor_summary(
                        actual_seq_lengths_query,
                        max_digest_numel=max_digest_numel,
                    ),
                    "actual_seq_lengths_key": _dsa_diag_tensor_summary(
                        actual_seq_lengths_key,
                        max_digest_numel=max_digest_numel,
                    ),
                    "cos": _dsa_diag_tensor_summary(
                        cos, max_digest_numel=max_digest_numel
                    ),
                    "sin": _dsa_diag_tensor_summary(
                        sin, max_digest_numel=max_digest_numel
                    ),
                    "prompt_lens": _dsa_diag_tensor_summary(
                        getattr(attn_metadata, "prompt_lens", None),
                        max_digest_numel=max_digest_numel,
                    ),
                    "decode_req_indices": _dsa_diag_tensor_summary(
                        getattr(attn_metadata, "decode_req_indices", None),
                        max_digest_numel=max_digest_numel,
                    ),
                    "decode_target_slot_mapping": _dsa_diag_tensor_summary(
                        getattr(attn_metadata, "decode_target_slot_mapping", None),
                        max_digest_numel=max_digest_numel,
                    ),
                }
            )
        if do_topk:
            output_summaries["topk_indices"] = _dsa_diag_tensor_summary(
                topk_indices, max_digest_numel=max_digest_numel
            )

        field_digests = {
            f"input.{name}": summary.get("value_digest")
            for name, summary in input_summaries.items()
        }
        field_digests.update(
            {
                f"output.{name}": summary.get("value_digest")
                for name, summary in output_summaries.items()
            }
        )

        summary_path = _dsa_diag_summary_path(
            session_id=_DSA_INDEXER_DIAG_SESSION_ID,
            signature=signature,
            run=run,
            layer_name=layer_name,
            tp_rank=int(self.tp_rank),
        )
        dump_path = None
        if do_dump:
            dump_path = summary_path.replace("_summary.json", ".pt")
            tensors = {}
            if do_query:
                tensors.update(
                    {
                        "x": self._dsa_indexer_diag_cpu_tensor(x),
                        "q_c": self._dsa_indexer_diag_cpu_tensor(q_c),
                        "q_li": self._dsa_indexer_diag_cpu_tensor(q_li),
                        "weights": self._dsa_indexer_diag_cpu_tensor(weights),
                        "query_dequant_scale": self._dsa_indexer_diag_cpu_tensor(
                            query_dequant_scale
                        ),
                    }
                )
            if do_key_active:
                tensors.update(
                    {
                        "key_active": key_active,
                        "key_dequant_scale_active": key_scale_active,
                    }
                )
            if do_meta:
                tensors.update(
                    {
                        "indexer_block_table": self._dsa_indexer_diag_cpu_tensor(
                            indexer_block_table
                        ),
                        "actual_seq_lengths_query": self._dsa_indexer_diag_cpu_tensor(
                            actual_seq_lengths_query
                        ),
                        "actual_seq_lengths_key": self._dsa_indexer_diag_cpu_tensor(
                            actual_seq_lengths_key
                        ),
                        "cos": self._dsa_indexer_diag_cpu_tensor(cos),
                        "sin": self._dsa_indexer_diag_cpu_tensor(sin),
                    }
                )
            if do_topk:
                tensors["topk_indices"] = self._dsa_indexer_diag_cpu_tensor(
                    topk_indices
                )
            if _DSA_INDEXER_DIAG_DUMP_FULL_KEY and _DSA_INDEXER_DIAG_EXACT_VALUES:
                tensors["key"] = self._dsa_indexer_diag_cpu_tensor(key_tensor)
                tensors["key_dequant_scale"] = self._dsa_indexer_diag_cpu_tensor(
                    key_dequant_scale,
                )
            elif _DSA_INDEXER_DIAG_DUMP_FULL_KEY:
                tensors["key_full_dump_skipped"] = (
                    "Set VLLM_ASCEND_DSA_INDEXER_DIAG_EXACT_VALUES=1 to "
                    "enable full paged-key dumps. Default diagnostic mode "
                    "only dumps sampled logical key_active rows."
                )
            torch.save(
                {
                    "meta": {
                        "req_ids": req_ids,
                        "diag_session": _DSA_INDEXER_DIAG_SESSION_ID,
                        "diag_components": sorted(_DSA_INDEXER_DIAG_COMPONENTS),
                        "signature": signature,
                        "prompt_run": run,
                        "layer_name": layer_name,
                        "layer_id": layer_id,
                        "tp_rank": int(self.tp_rank),
                        "key_active_meta": key_active_meta,
                        "key_dequant_scale_active_meta": key_scale_active_meta,
                    },
                    "tensors": tensors,
                },
                dump_path,
            )

        summary = {
            "summary_path": summary_path,
            "dump_path": dump_path,
            "req_ids": req_ids,
            "diag_session": _DSA_INDEXER_DIAG_SESSION_ID,
            "diag_components": sorted(_DSA_INDEXER_DIAG_COMPONENTS),
            "signature": signature,
            "prompt_run": run,
            "layer_name": layer_name,
            "layer_id": layer_id,
            "tp_rank": int(self.tp_rank),
            "input_summaries": input_summaries,
            "output_summaries": output_summaries,
            "field_digests": field_digests,
            "key_active_meta": key_active_meta,
            "key_dequant_scale_active_meta": key_scale_active_meta,
        }
        if do_file:
            _dsa_diag_write_json(summary_path, summary)
        if do_log:
            logger.warning(
            "[DSA_INDEXER_DIAG_DUMP] first_token_indexer req_ids=%s "
            "diag_session=%s components=%s signature=%s prompt_run=%s layer=%s "
            "tp_rank=%s summary=%s dump=%s field_digests=%s "
            "key_active_meta=%s",
            req_ids,
            _DSA_INDEXER_DIAG_SESSION_ID,
            sorted(_DSA_INDEXER_DIAG_COMPONENTS),
            signature,
            run,
            layer_name,
            int(self.tp_rank),
            summary_path,
            dump_path,
            field_digests,
            key_active_meta,
            )

        if not do_compare or run <= 1:
            return
        prev_path = _dsa_diag_summary_path(
            session_id=_DSA_INDEXER_DIAG_SESSION_ID,
            signature=signature,
            run=run - 1,
            layer_name=layer_name,
            tp_rank=int(self.tp_rank),
        )
        if not os.path.exists(prev_path):
            logger.warning(
                "[DSA_INDEXER_DIAG_RUN_COMPARE] first_token_indexer "
                "diag_session=%s signature=%s prev_run=%s prompt_run=%s "
                "layer=%s tp_rank=%s prev_summary_missing=%s "
                "current_summary=%s req_ids=%s",
                _DSA_INDEXER_DIAG_SESSION_ID,
                signature,
                run - 1,
                run,
                layer_name,
                int(self.tp_rank),
                prev_path,
                summary_path,
                req_ids,
            )
            return
        with open(prev_path, "r", encoding="utf-8") as f:
            previous = json.load(f)
        previous_digests = previous.get("field_digests", {})
        all_fields = sorted(set(previous_digests) | set(field_digests))
        mismatches = [
            field
            for field in all_fields
            if previous_digests.get(field) != field_digests.get(field)
        ]
        logger.warning(
            "[DSA_INDEXER_DIAG_RUN_COMPARE] first_token_indexer "
            "diag_session=%s signature=%s prev_run=%s prompt_run=%s "
            "layer=%s tp_rank=%s mismatch=%s mismatches=%s "
            "prev_req_ids=%s curr_req_ids=%s prev_summary=%s "
            "current_summary=%s prev_dump=%s current_dump=%s",
            _DSA_INDEXER_DIAG_SESSION_ID,
            signature,
            run - 1,
            run,
            layer_name,
            int(self.tp_rank),
            bool(mismatches),
            mismatches,
            previous.get("req_ids"),
            req_ids,
            prev_path,
            summary_path,
            previous.get("dump_path"),
            dump_path,
        )

    def indexer_select_post_process(
        self,
        x: torch.Tensor,
        q_c: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        layer_name: str | None = None,
    ):
        # DSA two-group mode: the indexer cache has its own block ids; fall back
        # to the (shared) latent block table in single-group mode.
        indexer_block_table = (
            attn_metadata.indexer_block_table
            if attn_metadata.indexer_block_table is not None
            else attn_metadata.block_table
        )
        weights, _ = self.weights_proj(x)

        q_li, _ = self.wq_b(q_c)  # [b,s,1536] @ [1536,64*128] = [b,s,64*128]
        q_li = q_li.view(-1, self.n_head, self.head_dim)  # [n_toks,64,128]
        if HAS_TRITON:
            q_li = rope_forward_triton_siso(
                q_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        else:
            q_li_pe, q_li_nope = torch.split(
                q_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )  # [b,s,64,64+64]

            q_li_pe = q_li_pe.unsqueeze(2)
            q_li_pe = torch_npu.npu_rotary_mul(q_li_pe, cos, sin)
            q_li_pe = q_li_pe.squeeze(2)
            q_li = torch.cat([q_li_pe, q_li_nope], dim=-1)  # [b*s,64,128]

        if self.use_sparse_c8_indexer:
            q_li_shape_ori = q_li.shape
            q_li = q_li @ AscendSFAImpl.q_hadamard
            q_li, q_li_scale = torch_npu.npu_dynamic_quant(q_li.view(-1, self.head_dim), dst_type=self.c8_k_cache_dtype)
            q_li_scale = q_li_scale.to(self.c8_k_scale_cache_dtype)

        diag_q_li = q_li
        diag_key_dequant_scale = None
        diag_query_dequant_scale = None

        # DSV3.2 currently has graph compilation issues when using torch_npu.npu.lightning_indexer.
        # So two branches are maintained temporarily.
        # TODO: torch.ops._C_ascend.npu_lightning_indexer needs to be removed.
        if self.use_sparse_c8_indexer:
            assert len(kv_cache) == 4
            weights = weights.to(torch.float16)
            diag_q_li = q_li.view(q_li_shape_ori)
            diag_key_dequant_scale = kv_cache[3].squeeze(2)
            diag_query_dequant_scale = q_li_scale.view(q_li_shape_ori[:-1])
            topk_indices = torch.ops._C_ascend.npu_lightning_indexer_quant(
                query=diag_q_li,
                key=kv_cache[2],
                weights=weights,
                query_dequant_scale=diag_query_dequant_scale,
                key_dequant_scale=diag_key_dequant_scale,  # B S N D -> B S D
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=indexer_block_table,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
            )
        elif self.use_torch_npu_lightning_indexer:
            topk_indices, _ = torch_npu.npu_lightning_indexer(
                query=q_li,
                key=kv_cache[2],
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=indexer_block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
            )
        else:
            topk_indices = torch.ops._C_ascend.npu_lightning_indexer(
                query=q_li,
                key=kv_cache[2],
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=indexer_block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
            )
        if _DSA_INDEXER_DIAG:
            try:
                self._dsa_indexer_diag_log(
                    layer_name=layer_name,
                    x=x,
                    q_c=q_c,
                    q_li=diag_q_li,
                    weights=weights,
                    kv_cache=kv_cache,
                    indexer_block_table=indexer_block_table,
                    attn_metadata=attn_metadata,
                    cos=cos,
                    sin=sin,
                    actual_seq_lengths_query=actual_seq_lengths_query,
                    actual_seq_lengths_key=actual_seq_lengths_key,
                    topk_indices=topk_indices,
                    key_dequant_scale=diag_key_dequant_scale,
                    query_dequant_scale=diag_query_dequant_scale,
                )
            except Exception:
                logger.warning(
                    "[DSA_INDEXER_DIAG_ERROR] first_token_indexer diagnostic "
                    "failed; continuing without diagnostic output",
                    exc_info=True,
                )
        return topk_indices

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        kv_override=None,
        key_rope_override=None,
        block_table_override=None,
        layer_name: str | None = None,
        trace_label: str = "native",
    ):
        # DSA latent offload: when overrides are given, read latent from the A1 scratch
        # (kv_override/key_rope_override) via the scratch block_table instead of the
        # full paged latent cache. Used by the decode-gather path.
        if kv_override is not None:
            block_table = block_table_override
            kv = kv_override
            key_rope = key_rope_override
        else:
            block_table = attn_metadata.block_table
            kv = kv_cache[0]
            key_rope = kv_cache[1]

        _dsa_decode_sparse_fa = (
            self.dsa_shrink_latent
            and block_table is not None
            and attn_metadata.num_decode_tokens > 0
            and attn_metadata.attn_state in (
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding,
            )
        )
        topk_2d = None
        if _dsa_decode_sparse_fa:
            topk_indices, topk_2d = _dsa_mask_padding_sparse_rows(
                topk_indices,
                getattr(attn_metadata, "decode_req_indices", None),
                getattr(attn_metadata, "num_actual_tokens", None),
            )

        if _dsa_decode_sparse_fa:
            if topk_2d is None:
                topk_2d = _dsa_topk_to_2d_indices(topk_indices)
            topk_rows = int(topk_2d.shape[0])
            block_table_rows = int(block_table.shape[0])
            batch_size = int(actual_seq_lengths_query.numel())
            if block_table_rows != batch_size:
                decode_req_indices = getattr(attn_metadata, "decode_req_indices", None)
                decode_req_indices_sample = None
                if decode_req_indices is not None:
                    decode_req_indices_sample = (
                        decode_req_indices[: min(topk_rows, 8)]
                        .detach()
                        .to(device="cpu")
                        .tolist()
                    )
                raise RuntimeError(
                    "DSA sparse FA block_table batch dimension mismatch: "
                    f"layer={layer_name} trace_label={trace_label} "
                    f"attn_state={attn_metadata.attn_state} "
                    f"topk_shape={tuple(topk_indices.shape)} "
                    f"topk_rows={topk_rows} "
                    f"block_table_shape={tuple(block_table.shape)} "
                    f"block_table_rows={block_table_rows} "
                    f"batch_size={batch_size} "
                    f"num_decode_tokens={attn_metadata.num_decode_tokens} "
                    f"decode_req_indices_shape="
                    f"{tuple(decode_req_indices.shape) if decode_req_indices is not None else None} "
                    f"decode_req_indices_sample={decode_req_indices_sample}"
                )
            if _DSA_TARGET_SLOT_GUARD:
                bad_block_hit = _dsa_sparse_fa_bad_block_hit(
                    topk_2d=topk_2d,
                    block_table=block_table,
                    row_req_indices=getattr(attn_metadata, "decode_req_indices", None),
                    block_size=int(kv.shape[1]),
                )
                if bad_block_hit is not None:
                    raise RuntimeError(
                        "DSA sparse FA topk resolves to a null/freed block: "
                        f"layer={layer_name} trace_label={trace_label} "
                        f"attn_state={attn_metadata.attn_state} "
                        f"topk_shape={tuple(topk_indices.shape)} "
                        f"block_table_shape={tuple(block_table.shape)} "
                        f"block_size={int(kv.shape[1])} "
                        f"num_decode_tokens={attn_metadata.num_decode_tokens} "
                        f"actual_seq_lengths_query_sample="
                        f"{_dsa_debug_sample(actual_seq_lengths_query)} "
                        f"actual_seq_lengths_key_sample="
                        f"{_dsa_debug_sample(actual_seq_lengths_key)} "
                        f"topk_minmax_count={_dsa_debug_minmax_count(topk_indices)} "
                        f"bad_hit={bad_block_hit}"
                    )

        _dsa_sync_probe("before_sparse_fa", layer_name)
        attn_output = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=ql_nope,
            key=kv,
            value=kv,
            sparse_indices=topk_indices,
            scale_value=self.scale,
            sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_key,
            query_rope=q_pe,
            key_rope=key_rope,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
        )
        return attn_output

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            if self.enable_dsa_cp_with_layer_shard and not _EXTRA_CTX.in_profile_run:
                for layer in self.layer_sharding_kwargs or []:
                    if is_hidden_layer(layer):
                        reach_layer_for_shard_weight_series(layer)
            return output.fill_(0)

        _dsa_prof.set_step_kind(
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
        )
        _sfa_t = _dsa_prof.begin("sfa_fwd")
        _is_pure_decode = attn_metadata.attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        )
        index_layer_name = (
            _dsa_indexer_layer_name(layer_name)
            if self.dsa_offload_unbundle
            else None
        )
        index_lmcache_enabled = (
            self.dsa_offload_unbundle
            and index_layer_name is not None
            and _dsa_index_lmcache_enabled()
        )
        if self.dsa_offload_unbundle and len(kv_cache) < 3:
            # Un-bundled: the indexer key is its own KV group (DeepseekV32IndexerCache).
            # layer_name is the inner MLAAttention name (...self_attn.attn); the indexer
            # cache is the sibling ...self_attn.indexer.k_cache. Re-assemble a 3-tuple so
            # the indexer read/write (kv_cache[2]) work unchanged — both groups share the
            # request's block ids, so attn_metadata.block_table/slot_mapping address both.
            # NOTE: in two-group mode the indexer group has its own block table and
            # slot mapping; the shared-block assumption only applies to legacy layouts.
            # The indexer KV tensor is allocated once at startup; cache the ref to avoid a
            # per-layer no_compile_layers dict lookup + tuple rebuild on the decode path.
            _idx_t = getattr(self, "_dsa_idx_cache_t", None)
            if _idx_t is None:
                _fc_ub = get_forward_context()
                assert index_layer_name is not None
                _idx_name = index_layer_name
                _idx_cache = _fc_ub.no_compile_layers[_idx_name].kv_cache[_fc_ub.virtual_engine]
                _idx_t = _idx_cache[0] if isinstance(_idx_cache, (tuple, list)) else _idx_cache
                self._dsa_idx_cache_t = _idx_t
            kv_cache = (kv_cache[0], kv_cache[1], _idx_t)

        cos = attn_metadata.cos
        sin = attn_metadata.sin
        slot_mapping = attn_metadata.slot_mapping
        # DSA two-group mode: the indexer cache write must use the indexer
        # group's own slots; falls back to the shared slots in single-group mode.
        idx_slot_mapping = (
            attn_metadata.indexer_slot_mapping
            if attn_metadata.indexer_slot_mapping is not None
            else slot_mapping
        )
        slot_mapping_cp = None
        if self.enable_dsa_cp:
            assert attn_metadata.dsa_cp_context is not None
            slot_mapping_cp = attn_metadata.dsa_cp_context.slot_mapping_cp
            actual_seq_lengths_query = attn_metadata.dsa_cp_context.actual_seq_lengths_query
            actual_seq_lengths_key = attn_metadata.dsa_cp_context.actual_seq_lengths_key
        else:
            actual_seq_lengths_query = attn_metadata.cum_query_lens
            actual_seq_lengths_key = attn_metadata.seq_lens

        # Inputs and outputs may be padded for CUDA graphs
        num_input_tokens = attn_metadata.num_input_tokens
        output_padded = output

        # all-gather o_proj weight for prefill stage of PD mix node
        o_proj_full_handle = None
        # if is PD mix stage, using original TP o_proj weight, and also need to full gather for o_proj
        # weight for prefill stage.
        full_gather_o_proj_enabled = self.enable_dsa_cp_with_o_proj_tp and attn_metadata.attn_state not in {
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        }

        # run mlapo ops when dsa-cp is disabled, and ensure that num_tokens satisfies the count limitation
        if self.enable_mlapo and num_input_tokens <= MLAPO_MAX_SUPPORTED_TOKENS:
            hidden_states, ql_nope, q_pe, q_c = self._sfa_preprocess_with_mlapo(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                cos=cos,
                sin=sin,
                slot_mapping=slot_mapping,
                num_input_tokens=num_input_tokens,
            )
            k_li, k_li_scale = self.indexer_select_pre_process(x=hidden_states, cos=cos, sin=sin)
        # native
        else:
            assert self.fused_qkv_a_proj is not None, "q lora is required for DSA."
            weight_prefetch_method = get_weight_prefetch_method()
            weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
                inputs=self.fused_qkv_a_proj.weight, dependency=hidden_states
            )
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_no_split = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            assert self.q_a_layernorm is not None, "q_a_layernorm must be initialized"
            q_c = self.q_a_layernorm(q_c)

            k_li, k_li_scale = self.indexer_select_pre_process(x=hidden_states, cos=cos, sin=sin)

            # Step B2: in compact-scratch mode the connector load is driven by
            # the post-indexer call (with selected_tokens). Calling here too
            # would advance the per-request layerwise retriever TWICE per layer
            # (this one with a dense arange) and desync it — skip whenever the
            # batch has decode rows (mixed steps included).
            if not (self.dsa_shrink_latent and attn_metadata.num_decode_tokens > 0):
                wait_for_kv_layer_from_connector(layer_name)

            if self.enable_dsa_cp:
                assert slot_mapping_cp is not None
                k_pe, k_nope = self.exec_kv(kv_no_split, cos, sin, kv_cache, slot_mapping_cp, attn_metadata)
            else:
                _fc = get_forward_context()
                _dsa_mgr_xkv = getattr(_fc, "dsa_offload_manager", None)
                if self.dsa_offload_free_paged and _dsa_mgr_xkv is not None:
                    # FREE_PAGED (prefill + decode): write latent into the PagedLatentPool
                    # (not the 1-block dummy kv_cache[0]/[1]); the op writes
                    # ckv_cache/k_cache at the pool's own slots. positions = arange(ctx,
                    # ctx+qlen) per request handles both prefill chunks and decode (qlen=1).
                    # HW-VERIFY: pool tensors are paged-layout for the op.
                    _qsl = torch.cat(
                        [attn_metadata.cum_query_lens.new_zeros(1), attn_metadata.cum_query_lens]
                    )
                    _ctx = attn_metadata.seq_lens - (_qsl[1:] - _qsl[:-1])
                    with _dsa_prof.section("exec_kv_slots"):
                        _pslots, _pknope, _pkpe = _dsa_mgr_xkv.pool_exec_kv_slots(
                            layer_name, _fc.dsa_req_ids, _qsl, _ctx,
                            decode=attn_metadata.attn_state == AscendAttentionState.DecodeOnly,
                        )
                    with _dsa_prof.section("exec_kv_op"):
                        k_pe, k_nope = self.exec_kv(
                            kv_no_split, cos, sin, (_pknope, _pkpe), _pslots, attn_metadata
                        )
                else:
                    with _dsa_prof.section("exec_kv"):
                        k_pe, k_nope = self.exec_kv(kv_no_split, cos, sin, kv_cache, slot_mapping, attn_metadata)

            if self.enable_dsa_cp:
                assert k_pe is not None
                assert k_nope is not None
                assert k_li is not None
                async_op = self.enable_dsa_cp_with_layer_shard or full_gather_o_proj_enabled
                # support all_gather kv async for communication calculation overlap
                if not self.use_sparse_c8_indexer:
                    fused_kv_no_split, kv_ag_handle = all_gather_async(
                        torch.cat(
                            [
                                k_pe.view(-1, k_pe.shape[-1]),
                                k_nope.view(-1, k_nope.shape[-1]),
                                k_li.view(-1, k_li.shape[-1]),
                            ],
                            dim=1,
                        ),
                        get_tp_group(),
                        async_op=async_op,
                    )
                else:
                    # due to different dtypes, we have to split commu pass
                    assert k_li_scale is not None
                    fused_kv_no_split, _ = all_gather_async(
                        torch.cat(
                            [
                                k_pe.view(-1, k_pe.shape[-1]),
                                k_nope.view(-1, k_nope.shape[-1]),
                            ],
                            dim=1,
                        ),
                        get_tp_group(),
                        async_op=async_op,
                    )
                    k_li, _ = all_gather_async(
                        k_li,
                        get_tp_group(),
                        async_op=async_op,
                    )
                    k_li_scale, kv_ag_handle = all_gather_async(
                        k_li_scale,
                        get_tp_group(),
                        async_op=async_op,
                    )

            ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
            q_pe = self.rope_single(q_pe, cos, sin)

            if self.enable_dsa_cp:
                if kv_ag_handle is not None:
                    kv_ag_handle.wait()

                if self.enable_dsa_cp_with_layer_shard:
                    for layer in self.layer_sharding_kwargs or []:
                        if is_hidden_layer(layer):
                            reach_layer_for_shard_weight_series(layer)
                elif full_gather_o_proj_enabled:
                    _, o_proj_full_handle = all_gather_async(
                        self.o_proj_tp_weight, get_tp_group(), output=AscendSFAImpl.o_proj_full_pool
                    )

                if kv_cache is not None:
                    assert fused_kv_no_split is not None
                    if not self.use_sparse_c8_indexer:
                        k_pe, k_nope, k_li = fused_kv_no_split.split(
                            [self.qk_rope_head_dim, self.kv_lora_rank, self.head_dim], dim=-1
                        )
                    else:
                        k_pe, k_nope = fused_kv_no_split.split([self.qk_rope_head_dim, self.kv_lora_rank], dim=-1)
                    k_nope = k_nope.view(k_nope.shape[0], 1, -1)
                    k_pe = k_pe.view(k_pe.shape[0], 1, -1)
                    DeviceOperator.reshape_and_cache(
                        key=k_nope[: attn_metadata.num_actual_tokens],
                        value=k_pe[: attn_metadata.num_actual_tokens],
                        key_cache=kv_cache[0],
                        value_cache=kv_cache[1],
                        slot_mapping=slot_mapping[: attn_metadata.num_actual_tokens],
                    )

            k_li = self._get_full_kv(k_li, attn_metadata)

        if kv_cache is not None:
            if index_lmcache_enabled:
                # Shared-cache sparse decode needs prompt index rows before
                # top-k selection; latent rows are loaded later after top-k.
                with _dsa_prof.section("lmc_index_retrieve"):
                    wait_for_kv_layer_from_connector(index_layer_name)

            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
            torch_npu.npu_scatter_nd_update_(
                kv_cache[2].view(-1, k_li.shape[-1]), idx_slot_mapping.view(-1, 1), k_li.view(-1, k_li.shape[-1])
            )  # b, s, n, d
            if self.use_sparse_c8_indexer:
                assert len(kv_cache) == 4
                assert k_li_scale is not None
                torch_npu.npu_scatter_nd_update_(
                    kv_cache[3].view(-1, k_li_scale.shape[-1]),
                    idx_slot_mapping.view(-1, 1),
                    k_li_scale.view(-1, k_li_scale.shape[-1]),
                )
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()

        _dsa_sync_probe("before_indexer", layer_name)
        with _dsa_prof.section("indexer"):
            topk_indices = self.indexer_select_post_process(
                x=hidden_states,
                q_c=q_c,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                cos=cos,
                sin=sin,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                layer_name=layer_name,
            )
        _dsa_sync_probe("after_indexer", layer_name)

        # DSA Step B2 (compact-scratch decode): the indexer just produced topk.
        # Remap prefill-selected entries to compact scratch rows [0..n_ret) (the
        # request's first ceil(k/block_size) latent blocks) and have LMCache
        # scatter exactly those tokens into the scratch; decode-selected entries
        # keep their ABSOLUTE positions (>= prompt_len >= k, disjoint from the
        # scratch row space) and are read in place via the same block table.
        # All fixed-shape device math — no D2H sync. No-op without a connector.
        if (
            self.dsa_shrink_latent
            and attn_metadata.prompt_lens is not None
            and attn_metadata.num_decode_tokens > 0
        ):
            # prompt_lens is per ROW: decode rows carry their request's prompt
            # length, prefill/padding rows carry 0 and stay untouched — so this
            # also covers mixed chunked-prefill + decode steps.
            # The packed front-list only feeds LMCache's selected_tokens; skip building
            # it (and its scatter) when no v1 connector will consume it (profiling /
            # no-offload runs). Production with an LMCache connector is unchanged.
            _need_packed = attn_metadata.need_sparse_lmcache_payload
            _topk_rows = int(topk_indices.shape[0])
            _scratch_base = getattr(attn_metadata, "decode_scratch_base", None)
            if _scratch_base is not None:
                _scratch_base = _scratch_base[:_topk_rows]
                if _scratch_base.device != topk_indices.device:
                    _scratch_base = _scratch_base.to(device=topk_indices.device)
            elif attn_metadata.decode_row_offsets is not None:
                _topk_width = int(topk_indices.numel() // max(_topk_rows, 1))
                _scratch_base = (
                    attn_metadata.decode_row_offsets[:_topk_rows]
                    .to(device=topk_indices.device)
                    * _topk_width
                )
            with _dsa_prof.section("scratch_remap"):
                topk_indices, _sel_packed = scratch_remap(
                    topk_indices,
                    attn_metadata.prompt_lens,
                    need_packed=_need_packed,
                    scratch_base=_scratch_base,
                )
            _dsa_sync_probe("after_scratch_remap", layer_name)
            # Stage 3 = isolation diagnostic: remap + FA on (garbage) scratch but
            # NO LMCache call. Output is expected wrong; only crash/no-crash
            # matters (crash => our remap/FA, clean => LMCache transfer kernel).
            if self.dsa_shrink_latent != 3 and _sel_packed is not None:
                _target_slot_mapping_for_wait = None
                _request_ids_for_wait = None
                _row_req_indices_for_wait = None
                _row_scratch_base_for_wait = None
                _valid_rows_all = getattr(
                    attn_metadata, "decode_valid_rows_all", False
                )
                _valid_row_indices = getattr(
                    attn_metadata, "decode_valid_row_indices", None
                )
                if _valid_rows_all or _valid_row_indices is not None:
                    if _valid_rows_all:
                        _selected_for_wait = _sel_packed
                    else:
                        _selected_for_wait = _sel_packed.index_select(
                            0, _valid_row_indices
                        )
                    _row_req_indices_for_wait = getattr(
                        attn_metadata, "decode_req_indices_compact", None
                    )
                    _row_scratch_base_for_wait = getattr(
                        attn_metadata, "decode_scratch_base_compact", None
                    )
                    _target_slot_mapping_for_wait = getattr(
                        attn_metadata, "decode_target_slot_mapping", None
                    )
                    _request_ids_for_wait = getattr(
                        attn_metadata, "decode_request_ids_compact", None
                    )
                elif attn_metadata.decode_req_indices is not None and _scratch_base is not None:
                    _decode_req_indices = attn_metadata.decode_req_indices[
                        : _sel_packed.shape[0]
                    ]
                    _decode_row_mask = _decode_req_indices >= 0
                    _selected_for_wait = _sel_packed[_decode_row_mask]
                    _row_req_indices = _decode_req_indices[_decode_row_mask]
                    _row_scratch_base = _scratch_base[: _sel_packed.shape[0]][
                        _decode_row_mask
                    ]
                    _row_req_indices_for_wait = _row_req_indices
                    _row_scratch_base_for_wait = _row_scratch_base
                    _target_slot_mapping_for_wait = _dsa_build_target_slot_mapping(
                        attn_metadata.block_table,
                        _row_req_indices,
                        _row_scratch_base,
                        int(_selected_for_wait.shape[1]),
                        int(kv_cache[0].shape[1]),
                    )
                    _dsa_req_ids = getattr(get_forward_context(), "dsa_req_ids", None)
                    if _dsa_req_ids is not None:
                        _decode_req_indices_cpu = getattr(
                            attn_metadata, "decode_req_indices_cpu", None
                        )
                        if _decode_req_indices_cpu is not None:
                            _request_ids_for_wait = [
                                _dsa_req_ids[int(req_idx)]
                                for req_idx in _decode_req_indices_cpu[
                                    : int(_sel_packed.shape[0])
                                ]
                                if int(req_idx) >= 0
                            ]
                        else:
                            _request_ids_for_wait = [
                                _dsa_req_ids[int(req_idx)]
                                for req_idx in _row_req_indices.detach()
                                .to(device="cpu")
                                .tolist()
                            ]
                else:
                    # Compatibility fallback for metadata built before row-level DSA
                    # fields existed. Standard MTP should not take this path.
                    _selected_for_wait = _sel_packed[: attn_metadata.num_decode_tokens]
                if (
                    _target_slot_mapping_for_wait is not None
                    and _DSA_TARGET_SLOT_GUARD
                ):
                    _dsa_validate_target_slot_mapping(
                        layer_name=layer_name,
                        selected_tokens=_selected_for_wait,
                        target_slot_mapping=_target_slot_mapping_for_wait,
                        kv_cache_layer=kv_cache[0],
                        row_req_indices=_row_req_indices_for_wait,
                        row_scratch_base=_row_scratch_base_for_wait,
                        block_table=attn_metadata.block_table,
                    )
                _wait_fn = wait_for_kv_layer_from_connector
                with _dsa_prof.section("lmc_retrieve"):
                    _wait_fn(
                        layer_name,
                        selected_tokens=_selected_for_wait,
                        target_slot_mapping=_target_slot_mapping_for_wait,
                        request_ids=_request_ids_for_wait,
                    )
                _dsa_sync_probe("after_lmc_retrieve", layer_name)

        # DSA latent KV offload (GLM5.1), single-card native non-CP path only:
        #   * prefill steps  -> store this layer's prompt latent, use native attention;
        #   * DecodeOnly step -> gather indexer-selected latent into the A1 scratch and
        #     run sparse attention against it. With ASSERT_PARITY, also run the native
        #     path and log the max-abs output diff, driving generation with the native
        #     result so a wrong scratch path can't corrupt output. Falls back to native
        #     when disabled or on unsupported paths (CP / sparse-c8 / mlapo).
        _dsa_fc = get_forward_context()
        _dsa_mgr = getattr(_dsa_fc, "dsa_offload_manager", None)
        _dsa_adapter = getattr(_dsa_fc, "dsa_adapter_cache", None)
        _dsa_on_native_path = not (
            self.enable_mlapo and num_input_tokens <= MLAPO_MAX_SUPPORTED_TOKENS
        )
        _dsa_supported = (
            _dsa_mgr is not None
            and not self.enable_dsa_cp
            and not self.use_sparse_c8_indexer
            and _dsa_on_native_path
        )
        # Adapter latent cache (separate flag). Needs per-request ids in the forward
        # context (absent in dummy/profile runs -> skip -> native).
        _adapter_supported = (
            _dsa_adapter is not None
            and getattr(_dsa_fc, "dsa_req_ids", None) is not None
            and not self.enable_dsa_cp
            and not self.use_sparse_c8_indexer
            and _dsa_on_native_path
        )
        if _dsa_mgr is not None and not _dsa_supported:
            # One-time heads-up if offload is enabled but this path can't use it, so a
            # missing [DSA-PARITY] log on the box is self-explanatory.
            logger.warning_once(
                "[DSA] latent offload enabled but inactive on this path "
                f"(dsa_cp={self.enable_dsa_cp}, sparse_c8={self.use_sparse_c8_indexer}, "
                f"mlapo_native={_dsa_on_native_path}); using native attention."
            )

        attn_output = None
        if _adapter_supported:
            # Adapter-backed latent hot cache: FA reads the resident pool in place
            # (zero-copy), the adapter owns residency (hit/miss) + eviction.
            _ac = _dsa_adapter
            _req_ids_a = _dsa_fc.dsa_req_ids
            _kn_a = k_nope.reshape(-1, self.kv_lora_rank)
            _kp_a = k_pe.reshape(-1, self.qk_rope_head_dim)
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                with _dsa_prof.section("ad_prep"):
                    # computed per layer (fresh): a cross-layer memo of these went
                    # stale on batch changes (wrong size) and bought no TPOT, so it was
                    # removed -- correctness over a non-win micro-opt.
                    _req_slots_a = _ac.req_slots_tensor(_req_ids_a)
                    _cur_pos_a = (attn_metadata.seq_lens.to(torch.long) - 1).tolist()
                    _topk2d = topk_indices[:, 0, :] if topk_indices.dim() == 3 else topk_indices
                with _dsa_prof.section("ad_insert"):
                    _insert_meta_op = False
                    for _b in range(len(_req_ids_a)):
                        # insert this step's generated token (one row per request);
                        # returns True only when it ran adapter metadata kernels (new
                        # block: load + mark_dirty) -- the only thing that races.
                        _insert_meta_op |= _ac.insert_decode_token(
                            layer_name, _req_ids_a[_b], int(_cur_pos_a[_b]), _kn_a[_b], _kp_a[_b]
                        )
                # WORKAROUND: the adapter's native metadata kernels (mark_dirty / load)
                # don't order with retrieve's load
                # on the device -> retrieve reads torn slot metadata -> bad slot ->
                # block_table OOB -> device hang. mark_dirty is once-per-block now, so
                # only block-allocation steps run those kernels; sync ONLY then. Normal
                # in-block steps do an ordered pool write and need no sync. Remove
                # entirely once the native kernels enforce their own device-side order.
                if _insert_meta_op and hasattr(torch, "npu"):
                    torch.npu.synchronize()
                with _dsa_prof.section("ad_retrieve"):
                    _res_a = _ac.retrieve(layer_name, _req_slots_a, _topk2d)
                with _dsa_prof.section("ad_fa"):
                    adapter_out = self._execute_sparse_flash_attention_process(
                        ql_nope,
                        q_pe,
                        kv_cache,
                        _res_a.sparse_indices.unsqueeze(1),
                        attn_metadata,
                        actual_seq_lengths_query,
                        _res_a.seq_lens,
                        kv_override=_res_a.knope_pool,
                        key_rope_override=_res_a.kpe_pool,
                        block_table_override=_res_a.block_table,
                        layer_name=layer_name,
                        trace_label="adapter",
                    )
                _dbg("fa_done")
                with _dsa_prof.section("ad_release"):
                    _ac.release_after_fa(layer_name, _res_a.loaded_ids)
                _dsa_prof.step()
                if envs.VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY:
                    native_out = self._execute_sparse_flash_attention_process(
                        ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
                        actual_seq_lengths_query, actual_seq_lengths_key,
                        layer_name=layer_name,
                        trace_label="adapter_parity_native",
                    )
                    diff = (native_out.float() - adapter_out.float()).abs().max()
                    logger.info("[DSA-ADAPTER-PARITY] layer=%s max_abs_diff=%s", layer_name, float(diff))
                    attn_output = native_out  # generation uses the native result
                else:
                    attn_output = adapter_out
            else:
                # prefill: store this layer's prompt latent into the adapter backend so
                # decode-time retrieve can fetch prefill-selected blocks; attention
                # itself uses the native prefill path (attn_output stays None).
                _qsl_a = torch.cat(
                    [attn_metadata.cum_query_lens.new_zeros(1), attn_metadata.cum_query_lens]
                )
                _ctx_a = attn_metadata.seq_lens - (_qsl_a[1:] - _qsl_a[:-1])
                _ac.store_prefill(layer_name, _req_ids_a, _qsl_a, _ctx_a, _kn_a, _kp_a)

        if attn_output is None and _dsa_supported:
            from vllm_ascend.distributed.kv_transfer.sparse_offload import sfa_hooks as _dsa_hooks

            _block_size = kv_cache[0].shape[1]
            # latent for this step's tokens comes straight from exec_kv's return
            # (is_output_kv=True), aligned with token order — no paged read-back, so this
            # no longer depends on the latent being resident in the paged cache (10b).
            _kn = k_nope.reshape(-1, self.kv_lora_rank)
            _kp = k_pe.reshape(-1, self.qk_rope_head_dim)
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                # store this step's token into the growing decode pool, then gather the
                # selected latent (prefill from LMCache, decode from pool) into scratch.
                _cur_pos = attn_metadata.seq_lens.to(torch.long) - 1
                with _dsa_prof.section("gather"):
                    s_knope, s_kpe, c_idx, s_bt, s_kv = _dsa_hooks.gather_decode(
                        _dsa_mgr,
                        layer_name,
                        _dsa_fc.dsa_req_ids,
                        topk_indices,
                        _dsa_fc.dsa_prompt_lens,
                        _cur_pos,
                        _block_size,
                        _kn,
                        _kp,
                        store_current=not self.dsa_offload_free_paged,
                    )
                # kernel expects sparse_indices as 3-D [num_tokens, 1, topk].
                with _dsa_prof.section("kernel"):
                    scratch_out = self._execute_sparse_flash_attention_process(
                        ql_nope,
                        q_pe,
                        kv_cache,
                        c_idx.unsqueeze(1),
                        attn_metadata,
                        actual_seq_lengths_query,
                        s_kv,
                        kv_override=s_knope,
                        key_rope_override=s_kpe,
                        block_table_override=s_bt,
                        layer_name=layer_name,
                        trace_label="lmcache_scratch",
                    )
                _dsa_prof.step()
                if envs.VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY and not self.dsa_offload_free_paged:
                    native_out = self._execute_sparse_flash_attention_process(
                        ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
                        actual_seq_lengths_query, actual_seq_lengths_key,
                        layer_name=layer_name,
                        trace_label="lmcache_parity_native",
                    )
                    diff = (native_out.float() - scratch_out.float()).abs().max()
                    logger.info("[DSA-PARITY] layer=%s max_abs_diff=%s", layer_name, float(diff))
                    attn_output = native_out  # safe: generation uses the native result
                else:
                    attn_output = scratch_out
            else:
                # prefill: (1) offload prompt latent to LMCache; (2) ALSO scatter it into
                # the self-managed PagedLatentPool and run prefill attention from the pool
                # (Route 1 / R1b). The vLLM paged latent is still written by the op, so
                # the parity path can compare pool-attn vs native-paged-attn.
                _qsl = torch.cat(
                    [attn_metadata.cum_query_lens.new_zeros(1), attn_metadata.cum_query_lens]
                )
                _ctx = attn_metadata.seq_lens - (_qsl[1:] - _qsl[:-1])
                _dsa_hooks.store_prefill(
                    _dsa_mgr, layer_name, _dsa_fc.dsa_req_ids, _qsl, _ctx, _kn, _kp
                )
                _dsa_mgr.populate_pool_layer(
                    _dsa_fc.dsa_req_ids, layer_name, _qsl, _ctx, _kn, _kp
                )
                _p_knope, _p_kpe, _p_bt = _dsa_mgr.pool_attn_args(
                    layer_name, _dsa_fc.dsa_req_ids, attn_metadata.block_table.shape[1]
                )
                pool_out = self._execute_sparse_flash_attention_process(
                    ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
                    actual_seq_lengths_query, actual_seq_lengths_key,
                    kv_override=_p_knope, key_rope_override=_p_kpe, block_table_override=_p_bt,
                    layer_name=layer_name,
                    trace_label="pool_prefill",
                )
                if envs.VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY and not self.dsa_offload_free_paged:
                    native_out = self._execute_sparse_flash_attention_process(
                        ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
                        actual_seq_lengths_query, actual_seq_lengths_key,
                        layer_name=layer_name,
                        trace_label="pool_prefill_parity_native",
                    )
                    diff = (native_out.float() - pool_out.float()).abs().max()
                    logger.info("[DSA-PARITY-PREFILL] layer=%s max_abs_diff=%s", layer_name, float(diff))
                    attn_output = native_out  # safe: generation uses the native result
                else:
                    attn_output = pool_out

        if attn_output is None:
            with _dsa_prof.section("fa"):
                attn_output = self._execute_sparse_flash_attention_process(
                    ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
                    actual_seq_lengths_query, actual_seq_lengths_key,
                    layer_name=layer_name,
                    trace_label="native",
                )
            # one step per layer-call on the native (user) path so the profiler
            # logs mean ms/layer-call periodically (mirrors the manager path).
            _dsa_prof.step()

        attn_output = self._v_up_proj(attn_output)
        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
            inputs=self.o_proj.weight,
            dependency=attn_output,
            max_size=MAX_O_PROJ_PREFETCH_SIZE,
            linear_layer=self.o_proj,
        )

        if self.enable_dsa_cp_with_o_proj_tp:
            # When using SFA-CP with pd mixed, o_proj has two cases:
            # 1. prefill: o_proj is a TP weight, we need to all-gather o_proj weight to switch TP=1.
            # 2. decode: all-to-all the hidden_state before the o_proj forward.
            result, require_o_proj_forward = self._handle_o_proj_weight_switch_and_forward(
                attn_output=attn_output,
                output=output,
                o_proj_full_handle=o_proj_full_handle,
                should_shard_weight=full_gather_o_proj_enabled,
            )
            if not require_o_proj_forward:
                _dsa_prof.end(_sfa_t)
                return result
            attn_output = result

        if self.enable_dsa_cp_strict_accuracy:
            send = (
                attn_output.view(-1, self.tp_size, self.num_heads * self.v_head_dim)
                .permute(1, 0, 2)
                .reshape(-1, self.num_heads * self.v_head_dim)
            )

            attn_output = torch.empty_like(send)
            torch.distributed.all_to_all_single(attn_output, send, group=get_tp_group().device_group)

        output[...] = self.o_proj(attn_output)[0]

        # Offload to LMCache. Legacy un-bundled connectors save only the latent
        # (k_nope, k_pe). Connectors declaring DSA index LMCache support also
        # save the sibling indexer layer in prefill; pure decode still skips
        # indexer save. Bundled path saves the whole tuple as before.
        # Shrink-latent: a pure-decode step's latent lives in the resident tail and is
        # never reloaded from LMCache, so saving it every decode layer is redundant
        # connector work (scales with batch). Skip save on steps with no prefill tokens
        # gated per step (num_prefills is shared by all layers), so the layerwise save
        # generator is never created that step and wait_for_save tolerates its absence.
        # NOTE: the SFA builder never populates attn_metadata.num_prefills (stays at
        # its dataclass default 0 on every step, prefill included), so gating on it
        # skipped the save unconditionally. Gate on attn_state instead, which the
        # builder does set: pure-decode steps are DecodeOnly/SpecDecoding.
        _skip_decode_save = (
            bool(self.dsa_shrink_latent)
            and _is_pure_decode
        )
        if not _skip_decode_save:
            if self.dsa_offload_unbundle and len(kv_cache) >= 2:
                maybe_save_kv_layer_to_connector(layer_name, [kv_cache[0], kv_cache[1]])
                if (
                    len(kv_cache) >= 3
                    and index_layer_name is not None
                    and index_lmcache_enabled
                    and not _is_pure_decode
                ):
                    maybe_save_kv_layer_to_connector(index_layer_name, [kv_cache[2]])
            else:
                maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))

        _dsa_prof.end(_sfa_t)
        return output_padded
