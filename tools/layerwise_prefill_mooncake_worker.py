# SPDX-License-Identifier: Apache-2.0
"""Selected ONLY by the two-holder test; install routing before LMCache init."""

from layerwise_prefill_mooncake_routing import install_routing

install_routing()


class MooncakeRoutingWorker:
    """No tensor probes, model overrides, or normal-serving hooks."""

    def prefill_check_seal_source(self):
        # Run once after generate, outside all layer/forward callbacks. Keep
        # workers and their native stores alive until the backup is complete.
        from mooncake.store import MooncakeDistributedStore
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        engine = get_kv_transfer_group()._lmcache_engine.lmcache_engine
        if not engine._force_layerwise_prefill_store:
            raise RuntimeError("Expected the layerwise prefill production path")
        engine.poll_layerwise_prefill_puts(final=True)
        engine.wait_for_pending_sync_stores()
        sources = [
            source
            for store in MooncakeDistributedStore.observed_stores
            if (source := store.completed_source()) is not None
        ]
        if len(sources) > 1:
            raise RuntimeError("Expected one P rank0-owned Mooncake store")
        return sources[0] if sources else None
