# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only stall snapshots for the explicitly armed serving diagnostic.

No device queries, output materialization, timers per token, or collectives.
One background thread checks host RPC progress and emits at most one snapshot
per worker/run. This observes stalls; it does not establish their device cause.
"""

import json
import os
import sys
import threading
import time
import traceback
from contextlib import ExitStack
from functools import wraps
from unittest.mock import patch

STALL_SECONDS = 30
POLL_SECONDS = 5
STACK_LIMIT = 24


class ServingStallDiagnostic:
    def __init__(self, worker, timing):
        self.worker = worker
        self.timing = timing
        self.owner_thread = threading.get_ident()
        self.progress = None
        self.reported = False
        self.stop = threading.Event()
        self.thread = None
        self.patches = ExitStack()
        self.root_start = worker.model_runner._sfa_full_graph.replay_count

    def observe(self, name):
        original = getattr(self.worker, name)

        @wraps(original)
        def call(*args, **kwargs):
            self.progress = (name + ".enter", time.monotonic())
            try:
                return original(*args, **kwargs)
            finally:
                # A returned async output is NOT necessarily device-complete.
                # Keep watching between RPCs and inspect the output thread too.
                self.progress = (name + ".return", time.monotonic())

        self.patches.enter_context(patch.object(self.worker, name, call))

    def snapshot_if_stalled(self):
        progress = self.progress
        if self.reported or self.stop.is_set() or progress is None:
            return None
        elapsed = time.monotonic() - progress[1]
        if elapsed < STALL_SECONDS:
            return None
        names = {thread.ident: thread.name for thread in threading.enumerate()}
        threads = []
        for ident, frame in sys._current_frames().items():
            stack = traceback.extract_stack(frame, limit=STACK_LIMIT)
            if ident != self.owner_thread and not any(entry.name == "async_output_busy_loop" for entry in stack):
                continue
            threads.append(
                {
                    "name": names.get(ident, str(ident)),
                    "stack": [f"{entry.filename}:{entry.lineno}:{entry.name}" for entry in stack],
                }
            )
        # Ignore a stale snapshot if the execution thread progressed meanwhile.
        if self.progress is not progress or self.stop.is_set():
            return None
        self.reported = True
        parallel = self.worker.vllm_config.parallel_config
        return {
            "pid": os.getpid(),
            "rank": self.worker.rank,
            "dp_rank": parallel.data_parallel_rank,
            "tp_rank": self.worker.rank % parallel.tensor_parallel_size,
            "event": "no_worker_rpc_progress",
            "seconds": round(elapsed, 1),
            "last_phase": progress[0],
            "decode_steps": self.timing.decode_steps,
            "root_replays_submitted": self.worker.model_runner._sfa_full_graph.replay_count - self.root_start,
            "threads": threads,
        }

    def watch(self):
        while not self.stop.wait(POLL_SECONDS):
            snapshot = self.snapshot_if_stalled()
            if snapshot is not None:
                print("[SFA_SERVING_STALL] " + json.dumps(snapshot), flush=True)

    def start(self):
        try:
            # Include idle DP participation: those workers have no real request
            # and never enter the active worker's sample_tokens RPC.
            for name in ("execute_model", "sample_tokens", "execute_dummy_batch"):
                self.observe(name)
            self.thread = threading.Thread(target=self.watch, name="sfa-serving-stall", daemon=True)
            self.thread.start()
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        self.stop.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1)
        self.patches.close()
