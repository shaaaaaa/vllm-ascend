# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA index-group PD connector (NPU->NPU) for two-group DSA models.

In DSA two-group mode (``VLLM_ASCEND_DSA_UNBUNDLE=1`` +
``VLLM_ASCEND_DSA_TWO_GROUPS=1``) the model exposes two KV cache groups:

  * group 0 = latent (k_nope + k_pe), a 2-tuple per layer;
  * group 1 = indexer key, a 1-tuple per layer whose layer name contains
    ``"indexer"``.

For PD disaggregation the indexer key must be transferred P->D and land
resident in the decoder's NPU index cache. This connector reuses the existing
Mooncake NPU->NPU block transport but restricts it to the indexer group only.
The latent group is handled by a separate connector such as LMCache, and the
two are composed with AscendMultiConnector.
"""

from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    MooncakeConnector,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

_INDEX_LAYER_MARKER = "indexer"
_DEFAULT_INDEX_GROUP_ID = 1


def _is_index_layer(layer_name: str) -> bool:
    return _INDEX_LAYER_MARKER in layer_name.lower()


class MooncakeDSAIndexConnector(MooncakeConnector, SupportsHMA):
    """Mooncake connector that transfers only the DSA indexer KV group."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        assert vllm_config.kv_transfer_config is not None
        self.index_group_id = int(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "index_group_id", _DEFAULT_INDEX_GROUP_ID
            )
        )
        super().__init__(vllm_config, role, kv_cache_config)

    def register_kv_caches(self, kv_caches: dict) -> None:
        index_only = {
            name: cache for name, cache in kv_caches.items() if _is_index_layer(name)
        }
        if not index_only:
            raise RuntimeError(
                "MooncakeDSAIndexConnector found no 'indexer' layers in kv_caches. "
                "This connector requires DSA two-group mode "
                "(VLLM_ASCEND_DSA_UNBUNDLE=1 and VLLM_ASCEND_DSA_TWO_GROUPS=1). "
                f"Saw {len(kv_caches)} layers, none matching '{_INDEX_LAYER_MARKER}'."
            )
        register_full_source = (
            getattr(self, "_latent_live_source_enabled", False)
            and self.connector_worker is not None
            and self.connector_worker.kv_role in ("kv_producer", "kv_both")
            and self.connector_worker.tp_rank == 0
        )
        logger.debug(
            "MooncakeDSAIndexConnector: registering %d indexer-group layers "
            "(of %d total) for NPU->NPU transfer from group %d; latent group "
            "handled elsewhere.",
            len(index_only),
            len(kv_caches),
            self.index_group_id,
        )
        if register_full_source:
            super().register_kv_caches(
                kv_caches, ordinary_kv_caches=index_only
            )
        else:
            super().register_kv_caches(index_only)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        # Indexer KV alone is not sufficient to skip prefill. Let the latent
        # connector decide the matched-token count.
        return 0, False

    def _select_index_group_ids(
        self,
        block_ids: tuple[list[int], ...],
    ) -> list[int]:
        if self.index_group_id >= len(block_ids):
            raise RuntimeError(
                "MooncakeDSAIndexConnector expected index group "
                f"{self.index_group_id}, but only {len(block_ids)} groups exist."
            )
        return block_ids[self.index_group_id]

    def _select_index_group_blocks(self, blocks: "KVCacheBlocks") -> "KVCacheBlocks":
        if self.index_group_id >= len(blocks.blocks):
            raise RuntimeError(
                "MooncakeDSAIndexConnector expected index group "
                f"{self.index_group_id}, but only {len(blocks.blocks)} groups exist."
            )
        return KVCacheBlocks((blocks.blocks[self.index_group_id],))

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens <= 0:
            return
        assert self.connector_scheduler is not None
        params: dict[str, Any] | None = getattr(request, "kv_transfer_params", None)
        old_do_remote_prefill = (
            params.get("do_remote_prefill") if params is not None else None
        )
        index_blocks = self._select_index_group_blocks(blocks)
        logger.debug(
            "MooncakeDSAIndexConnector D alloc: request_id=%s index_group=%d "
            "external_tokens=%d local_index_blocks=%d do_remote_prefill=%s",
            request.request_id,
            self.index_group_id,
            num_external_tokens,
            len(index_blocks.blocks[0]),
            old_do_remote_prefill,
        )
        self.connector_scheduler.update_state_after_alloc(
            request,
            index_blocks,
            num_external_tokens,
        )
        if params is not None and old_do_remote_prefill is not None:
            params["do_remote_prefill"] = old_do_remote_prefill

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        index_block_ids = self._select_index_group_ids(block_ids)
        logger.debug(
            "MooncakeDSAIndexConnector P finish: request_id=%s index_group=%d "
            "remote_index_blocks=%d",
            request.request_id,
            self.index_group_id,
            len(index_block_ids),
        )
        return self.connector_scheduler.request_finished(request, index_block_ids)
