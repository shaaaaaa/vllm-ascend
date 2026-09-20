# SPDX-License-Identifier: Apache-2.0
"""Tool-only worker hooks: collect the ends of one fixed-budget prefill request.

Installed by collective RPC AFTER model startup. No production worker changes,
tensor inspection, per-layer checks, or extra per-chunk synchronization.
"""

EDGE_CHUNKS = 3
PREFIX = "[PREFILL_PROFILE]"


def synchronize_boundary():
    # Only four window boundaries, not every chunk/layer/kernel. Flush all
    # streams so unfinished middle-chunk work cannot leak into the tail trace.
    import torch_npu

    torch_npu.npu.synchronize()


def make_capture_plan(prompt_tokens, chunk_tokens):
    chunks = (prompt_tokens + chunk_tokens - 1) // chunk_tokens
    windows = (
        [{"name": "all", "first_chunk": 1, "last_chunk": chunks}]
        if chunks <= 2 * EDGE_CHUNKS
        else [
            {"name": "head", "first_chunk": 1, "last_chunk": EDGE_CHUNKS},
            {"name": "tail", "first_chunk": chunks - EDGE_CHUNKS + 1, "last_chunk": chunks},
        ]
    )
    return {"prompt_tokens": prompt_tokens, "chunk_tokens": chunk_tokens, "total_chunks": chunks, "windows": windows}


class ChunkProfileCapture:
    def __init__(self, worker, case, plan):
        self.worker = worker
        self.case = case
        self.plan = plan
        self.original_execute = worker.execute_model
        self.active = None
        self.recorded_windows = []
        self.chunks = []
        self.tokens = 0

    def stop_window(self):
        if self.active is not None:
            print(f"{PREFIX} rank={self.worker.rank}: {self.case}/{self.active} profiler stop begin", flush=True)
            synchronize_boundary()
            self.worker.profile(is_start=False)
            # Worker.profile() otherwise restarts the OLD trace name. Each
            # segment needs a fresh profiler and a distinct head/tail handler.
            self.worker.profiler = None
            self.active = None

    def execute_model(self, scheduler_output, *args, **kwargs):
        count = scheduler_output.total_num_scheduled_tokens
        if count > 0:
            chunk = len(self.chunks) + 1
            window = next(
                (w["name"] for w in self.plan["windows"] if w["first_chunk"] <= chunk <= w["last_chunk"]), None
            )
            if window != self.active:
                # Stop before the next execute_model, not after the previous
                # one: its sample_tokens/MTP RPC and async copies belong to it.
                self.stop_window()
                if window is not None:
                    synchronize_boundary()
                    print(
                        f"{PREFIX} rank={self.worker.rank}: {self.case}/{window} profiler start; chunk={chunk}",
                        flush=True,
                    )
                    self.worker.profile(is_start=True, profile_prefix=f"{self.case}_{window}")
                    self.active = window
                    self.recorded_windows.append(window)
            self.chunks.append(
                {"chunk": chunk, "token_start": self.tokens, "token_end": self.tokens + count, "window": window}
            )
            self.tokens += count
        return self.original_execute(scheduler_output, *args, **kwargs)

    def finish(self):
        try:
            self.stop_window()
        finally:
            self.worker.execute_model = self.original_execute
        return {"rank": self.worker.rank, "windows": self.recorded_windows, "chunks": self.chunks}


def install_chunk_profile(worker, case, plan):
    """Callable collective RPC; the parent invokes this once, before generate."""
    # MultiprocExecutor passes WorkerWrapperBase. Mutate the actual worker,
    # not the proxy (whose __getattr__ forwards reads but NOT assignments).
    worker = getattr(worker, "worker", worker)
    capture = ChunkProfileCapture(worker, case, plan)
    worker._prefill_chunk_profile_capture = capture
    worker.execute_model = capture.execute_model


def finish_chunk_profile(worker):
    worker = getattr(worker, "worker", worker)
    capture = worker._prefill_chunk_profile_capture
    try:
        return capture.finish()
    finally:
        del worker._prefill_chunk_profile_capture


def validate_capture(plan, workers):
    """Post-request coverage check, never on a production execution path."""
    expected = [
        (start, min(start + plan["chunk_tokens"], plan["prompt_tokens"]))
        for start in range(0, plan["prompt_tokens"], plan["chunk_tokens"])
    ]
    windows = [w["name"] for w in plan["windows"]]
    if not workers:
        raise RuntimeError("No worker capture reports; cannot verify chunk coverage")
    for report in workers:
        actual = [(c["token_start"], c["token_end"]) for c in report["chunks"]]
        if actual != expected or report["windows"] != windows:
            raise RuntimeError(
                f"rank={report['rank']}: actual prefill chunk layout differs from the capture plan; "
                "see capture_windows.json. Do not treat these traces as the first/last three chunks."
            )
