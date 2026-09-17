# SPDX-License-Identifier: Apache-2.0
"""Only selected by the file SDK test. No tensor probes or model changes."""

from layerwise_prefill_file_store import install

install()


class FileStoreWorker:
    def prefill_check_flush_store(self):
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        engine = get_kv_transfer_group()._lmcache_engine.lmcache_engine
        if not engine._force_layerwise_prefill_store:
            raise RuntimeError("Expected the real layerwise prefill store path")
        engine.poll_layerwise_prefill_puts(final=True)
        engine.wait_for_pending_sync_stores()
        return True
