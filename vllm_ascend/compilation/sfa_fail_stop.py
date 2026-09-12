# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Error-only fail-stop for a single supervised, local SFA worker cohort.

No normal-step communication, timers or watchdog polling. WorkerProc's existing
process-sentinel monitor wakes when the failed worker exits and terminates its
peer workers, including peers blocked in an NPU collective. Ordinary Exception
is insufficient: WorkerProc's RPC loop catches it and continues serving.
"""

import os
from multiprocessing import parent_process
from threading import Timer
from typing import Any, NoReturn

SFA_FATAL_EXIT_CODE = 1
SFA_FATAL_CLEANUP_TIMEOUT_SECONDS = 5.0


def uses_local_sfa_fail_stop(parallel_config: Any) -> bool:
    """Only opt into fail-stop when one local MP supervisor owns all peers.

    Unknown/custom executors, multi-node and multi-DP jobs retain synchronous
    error agreement. This guard must never terminate an in-process LLM caller.
    Startup capture keeps agreement until worker initialization has completed.
    """
    return (
        getattr(parallel_config, "distributed_executor_backend", None) == "mp"
        and getattr(parallel_config, "nnodes", None) == 1
        and getattr(parallel_config, "data_parallel_size", None) == 1
        and getattr(parallel_config, "pipeline_parallel_size", None) == 1
        and parent_process() is not None
    )


def exit_failed_sfa_worker(error: Exception) -> NoReturn:
    """Exit this failed worker, with a bounded cleanup fallback.

    The caller logs the original traceback first. SystemExit bypasses the RPC
    loop's catch-and-continue handler. If connector/runtime cleanup blocks on
    the way out, the error-only daemon timer exits THIS process, never a PID
    selected from the environment. The parent then handles peer termination.
    This is fail-stop, not recovery/retry of an incomplete distributed forward.
    """
    try:
        fallback = Timer(SFA_FATAL_CLEANUP_TIMEOUT_SECONDS, os._exit, args=(SFA_FATAL_EXIT_CODE,))
        fallback.daemon = True
        fallback.start()
    except Exception:
        # Resource exhaustion must not turn this into a catch-and-continue RPC
        # error or leave cleanup unbounded without its fallback timer.
        os._exit(SFA_FATAL_EXIT_CODE)
    raise SystemExit(SFA_FATAL_EXIT_CODE) from error
