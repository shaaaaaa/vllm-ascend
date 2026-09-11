# SPDX-License-Identifier: Apache-2.0
"""Explicitly selected diagnostic worker; never imported by normal serving."""

import os

import torch
from sfa_startup_trace import StartupTrace
from vllm.config import set_current_vllm_config

from vllm_ascend.worker.sfa_parity_worker import SFAParityWorker


class SFAStartupWorker(SFAParityWorker):
    def __init__(self, vllm_config, local_rank, rank, **kwargs):
        self.startup_options = vllm_config.additional_config["sfa_startup"]
        self.startup_trace = StartupTrace(self.startup_options["trace_dir"], rank=rank, local_rank=local_rank)
        self.startup_trace.install(torch)
        with self.startup_trace.span("worker_constructor"):
            super().__init__(vllm_config, local_rank, rank, **kwargs)

    def init_device(self):
        from vllm.distributed.parallel_state import GroupCoordinator

        self.startup_trace.trace_coordinator(GroupCoordinator)
        with self.startup_trace.span("worker_init_device"):
            super().init_device()
        device = torch.npu.current_device()
        visible = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
        # Visibility-derived ID is labelled as such, not represented as a
        # driver-reported physical ID. UUID is optional on older torch_npu.
        uuid = None
        try:
            uuid = getattr(torch.npu.get_device_properties(device), "uuid", None) or None
        except Exception as exc:
            self.startup_trace.emit("device_uuid_unavailable", error=str(exc))
        self.startup_trace.emit(
            "device_mapping",
            device=device,
            visible_device=visible[device],
            uuid=uuid,
        )

    def load_model(self):
        # Paths/versions are recorded in the actual worker, not inferred from
        # what the parent imported. Useful for mixed native builds after merge.
        self.startup_trace.runtime_libraries()
        if self.startup_options["load_model"]:
            with self.startup_trace.span("original_parity_load_model"):
                super().load_model()
        else:
            # Match the real failing W4A8 constructor, under the same config.
            # No synthetic torch tensors/extra HCCL collectives are introduced.
            from vllm_ascend.quantization.methods.w4a8 import AscendW4A8DynamicFusedMoEMethod

            with self.startup_trace.span("w4a8_mc2_probe"), set_current_vllm_config(self.vllm_config):
                self._prepare_quant_config()
                self.startup_scheme = AscendW4A8DynamicFusedMoEMethod()
                if not self.startup_scheme.moe_all_to_all_group_name:
                    raise RuntimeError("W4A8 skipped HCCL initialization; this is not a startup pass")
        if not self.startup_trace.comm_calls:
            raise RuntimeError("No get_hccl_comm_name C call observed; diagnostic coverage is incomplete")
        self.startup_trace.emit("startup_complete", comm_name_calls=self.startup_trace.comm_calls)

    def startup_report(self):
        return {"rank": self.rank, "local_rank": self.local_rank, "pid": os.getpid(), "complete": True}
