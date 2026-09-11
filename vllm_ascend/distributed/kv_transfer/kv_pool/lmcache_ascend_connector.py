# SPDX-License-Identifier: Apache-2.0
import lmcache_ascend  # noqa: F401
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1 as _LMCacheConnectorV1,
)


class LMCacheConnectorV1(_LMCacheConnectorV1):
    supports_dsa_index_lmcache = True

    def update_connector_worker_metadata(
        self,
        worker_metadata: KVConnectorWorkerMetadata,
        active_req_ids: set[str],
    ) -> None:
        """Forward same-step completion metadata into LMCache."""
        update = getattr(
            self._lmcache_engine, "update_connector_worker_metadata", None
        )
        if callable(update):
            update(worker_metadata, active_req_ids)


__all__ = ["LMCacheConnectorV1"]
