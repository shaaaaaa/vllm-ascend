# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eight-layer loading fixture WITHOUT parity hooks or per-forward inspection."""

import os
from unittest.mock import patch

import torch
from vllm.distributed import get_tp_group
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

from vllm_ascend import envs
from vllm_ascend.attention.sfa_parity import coordinated_check
from vllm_ascend.worker.sfa_parity_worker import SFAParityWorker, deterministic_dummy_load
from vllm_ascend.worker.worker import NPUWorker


class SFABenchmarkWorker(NPUWorker):
    """Only explicitly selected by tools/sfa_graph_benchmark.py.

    Deliberately NOT a subclass of SFAParityWorker. Reuse its startup-only
    quantization adjustment, not its load_model, forward wrappers, snapshots,
    checkpoint import/export, sampling changes, or per-step collectives.
    """

    def load_model(self) -> None:
        parallel = self.vllm_config.parallel_config
        if (
            self.vllm_config.additional_config.get("sfa_benchmark") is not True
            or parallel.tensor_parallel_size not in (1, 2, 4, 8)
            or parallel.data_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.enable_expert_parallel
        ):
            raise ValueError("Use the isolated TP-only SFA benchmark driver")
        # This method only reads/replaces vllm_config.quant_config. It installs
        # no hooks and needs no parity request/reference state.
        coordinated_check(
            lambda: SFAParityWorker._prepare_quant_config(self),
            group=get_tp_group(),
            phase="benchmark startup MTP quantization",
        )
        original = DummyModelLoader.load_weights

        def load(loader, model, model_config):
            deterministic_dummy_load(original, loader, model, model_config)

        with patch.object(DummyModelLoader, "load_weights", load):
            super().load_model()

    def benchmark_state(self) -> dict:
        """Called between requests, NEVER from a timed forward."""
        torch.npu.synchronize()
        runner = self.model_runner
        graph = runner._sfa_full_graph
        return {
            "rank": self.rank,
            "pid": os.getpid(),
            "layers": self.model_config.hf_config.num_hidden_layers,
            "staged": bool(envs.VLLM_ASCEND_SFA_STAGED_GRAPH),
            "full": bool(envs.VLLM_ASCEND_SFA_FULL_GRAPH),
            "root_replays": graph.replay_count,
            "root_sealed": graph.sealed,
            "root_keys": len(graph.entries),
        }

    def benchmark_process_info(self) -> dict:
        """Identify owned workers even if the subsequent graph-state gate fails."""
        return {"rank": self.rank, "pid": os.getpid()}

    def benchmark_release_resources(self) -> dict:
        from vllm.distributed.kv_transfer import ensure_kv_transfer_shutdown

        from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel

        if not getattr(self, "_benchmark_released", False):
            if torch.npu.is_initialized():
                torch.npu.synchronize()
            try:
                ensure_kv_transfer_shutdown()
            finally:
                destroy_ascend_model_parallel()
            self._benchmark_released = True
        return self.benchmark_process_info()

    def shutdown(self) -> None:
        try:
            self.benchmark_release_resources()
        finally:
            super().shutdown()
