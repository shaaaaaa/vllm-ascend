# SPDX-License-Identifier: Apache-2.0
import lmcache_ascend  # noqa: F401
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1 as _LMCacheConnectorV1,
)


class LMCacheConnectorV1(_LMCacheConnectorV1):
    supports_dsa_index_lmcache = True


__all__ = ["LMCacheConnectorV1"]
