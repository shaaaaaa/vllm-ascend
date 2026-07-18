# mypy: ignore-errors
import signal

import torch
import torch.distributed as dist
import vllm
from vllm.config import ParallelConfig
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.structured_output import StructuredOutputManager


class BalanceScheduler(Scheduler):
    def __init__(
        self,
        vllm_config,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            mm_registry,
            include_finished_set,
            log_stats,
        )
        self.balance_queue = [
            torch.tensor([0], dtype=torch.int, device="cpu")
            for _ in range(self.vllm_config.parallel_config.data_parallel_size)
        ]
        logger.info(
            "[BALANCE_SCHED_INIT] dp_size=%d max_running=%d "
            "schedule_source=base admission_hook=global_near_full "
            "final_hidden_compatible=true",
            self.vllm_config.parallel_config.data_parallel_size,
            self.max_num_running_reqs,
        )

    def balance_gather(self, dp_group):
        running_tensor = torch.tensor(
            [len(self.running)], dtype=torch.int, device="cpu"
        )
        dist.all_gather(self.balance_queue, running_tensor, group=dp_group)

    def _should_stop_scheduling_waiting(self) -> bool:
        if super()._should_stop_scheduling_waiting():
            return True
        return (
            max(t.item() for t in self.balance_queue)
            >= self.max_num_running_reqs - 1
        )


class BalanceDPEngineCoreProc(DPEngineCoreProc):
    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        while True:
            self._process_input_queue()

            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    continue

                self.execute_dummy_batch()

            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )
            self.scheduler.balance_gather(self.dp_group)

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                self.current_wave += 1
                self.step_counter = 0


def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    """Launch EngineCore busy loop in background process."""

    shutdown_requested = False
    maybe_register_config_serialize_by_value()

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    engine_core: EngineCoreProc | None = None
    try:
        parallel_config: ParallelConfig = kwargs["vllm_config"].parallel_config
        if parallel_config.data_parallel_size > 1 or dp_rank > 0:
            set_process_title("EngineCore", f"DP{dp_rank}")
            decorate_logs()
            parallel_config.data_parallel_rank = dp_rank
            parallel_config.data_parallel_rank_local = local_dp_rank
            engine_core = BalanceDPEngineCoreProc(*args, **kwargs)
        else:
            set_process_title("EngineCore")
            decorate_logs()
            engine_core = EngineCoreProc(*args, **kwargs)

        engine_core.run_busy_loop()

    except SystemExit:
        logger.debug("EngineCore exiting.")
        raise
    except Exception as e:
        if engine_core is None:
            logger.exception("EngineCore failed to start.")
        else:
            logger.exception("EngineCore encountered a fatal error.")
            engine_core._send_engine_dead()
        raise e
    finally:
        if engine_core is not None:
            engine_core.shutdown()


EngineCoreProc.run_engine_core = run_engine_core
vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler
