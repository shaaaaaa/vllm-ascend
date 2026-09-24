# SPDX-License-Identifier: Apache-2.0
"""CPU-only, per-worker progress for the explicit correctness experiment.

Main-thread updates only replace scalar state. A monitor publishes the latest
snapshot every 30 seconds and captures Python stacks after 90 seconds in one
unchanged operation. It cannot run while another thread holds the GIL forever,
and does not install or replace any process-global faulthandler watchdog.
"""

import faulthandler
import json
import os
import threading
import time
from contextlib import suppress
from pathlib import Path

POLL_SECONDS = 30.0
STALL_SECONDS = 90.0
JOIN_SECONDS = 1.0
MAX_LOG_CHARACTERS = 100


def _print_progress(line):
    print(line, flush=True)


class ProgressMonitor:
    def __init__(
        self,
        directory,
        rank,
        *,
        clock=time.monotonic,
        log=_print_progress,
        stack_dumper=faulthandler.dump_traceback,
        poll_seconds=POLL_SECONDS,
        stall_seconds=STALL_SECONDS,
    ):
        if poll_seconds <= 0 or stall_seconds <= 0:
            raise ValueError("Progress intervals must be positive")
        self.directory = Path(directory)
        self.rank = int(rank)
        self.clock = clock
        self.log = log
        self.stack_dumper = stack_dumper
        self.poll_seconds = float(poll_seconds)
        self.stall_seconds = float(stall_seconds)
        self._lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._started = clock()
        self._phase_started = self._started
        self._ended = None
        self._generation = 0
        self._last_stall = -1
        self._last_log = self._started
        self._diagnostic_error_logged = False
        self._state = {
            "rank": self.rank,
            "pid": os.getpid(),
            "status": "running",
            "operation": "install",
            "phase": "install",
            "step": -1,
            "layer": -1,
            "kind": "",
            "name": "",
            "records": 0,
            "files": 0,
            "bytes": 0,
        }

    def phase(self, phase, **fields):
        """Publish no files here: forward hooks only update scalar CPU state."""
        with self._lock:
            if self._state["status"] != "running":
                return
            self._state.update(fields, phase=phase)
            self._phase_started = self.clock()
            self._generation += 1

    def begin_record(self, *, step, layer, kind, name):
        self.phase("copy_cpu", step=int(step), layer=int(layer), kind=str(kind), name=str(name))

    def record_done(self, records, files, size):
        self.phase("compute", records=int(records), files=int(files), bytes=int(size))

    def begin_call(self, operation):
        self.phase("compute", operation=operation)

    def end_call(self, operation):
        self.phase("await_sample" if operation == "execute_model" else "running", operation=f"{operation}_done")

    def snapshot(self):
        with self._lock:
            now = self._ended if self._ended is not None else self.clock()
            return {
                **self._state,
                "phase_elapsed_seconds": round(max(0.0, now - self._phase_started), 3),
                "elapsed_seconds": round(max(0.0, now - self._started), 3),
                "generation": self._generation,
            }

    def _short_log(self, text):
        # Closed worker stdout must not replace an archive/model exception.
        with suppress(Exception):
            self.log(text[:MAX_LOG_CHARACTERS])

    def _diagnostic_error(self, error):
        # A failed diagnostic must not replace the original model/archive error.
        if not self._diagnostic_error_logged:
            self._diagnostic_error_logged = True
            self._short_log(f"[PFC] r{self.rank} diagnostic I/O failed: {type(error).__name__}")

    def publish(self):
        if not self._publish_lock.acquire(blocking=False):
            return
        locked = True
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = self.directory / "progress.json.tmp"
            while True:
                snapshot = self.snapshot()
                temporary.write_text(
                    json.dumps(snapshot, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8"
                )
                os.replace(temporary, self.directory / "progress.json")
                with self._lock:
                    # A final update may have skipped publication while this I/O
                    # was in flight. Refresh it before atomically releasing our
                    # publication lock, without making forward updates wait on I/O.
                    final_changed = self._state["status"] != "running" and self._state["status"] != snapshot["status"]
                    if not final_changed:
                        self._publish_lock.release()
                        locked = False
                        return
        except Exception as error:
            self._diagnostic_error(error)
        finally:
            if locked:
                self._publish_lock.release()

    def tick(self):
        """One monitor iteration; fake clocks can exercise this without threads."""
        self.publish()
        snapshot = self.snapshot()
        if snapshot["status"] != "running":
            return
        stalled = snapshot["phase_elapsed_seconds"] >= self.stall_seconds
        dump = False
        with self._lock:
            if (
                stalled
                and self._state["status"] == "running"
                and snapshot["generation"] == self._generation
                and self._last_stall != self._generation
            ):
                self._last_stall = self._generation
                dump = True
        location = f"s{snapshot['step']} l{snapshot['layer']} {snapshot['phase']} {snapshot['kind']}/{snapshot['name']}"
        if dump:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                # Keep only the latest stalled operation, bounding diagnostic disk use.
                with (self.directory / "stacks.txt").open("w", encoding="utf-8") as output:
                    output.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
                    output.flush()
                    self.stack_dumper(file=output, all_threads=True)
                self._short_log(
                    f"[PFC] r{self.rank} stalled {snapshot['phase_elapsed_seconds']:.0f}s {location} -> stacks.txt"
                )
            except Exception as error:
                self._diagnostic_error(error)
            self._last_log = self.clock()
        elif self.rank == 0 and self.clock() - self._last_log >= self.poll_seconds:
            self._short_log(
                f"[PFC] r0 {location} {snapshot['phase_elapsed_seconds']:.0f}s "
                f"n={snapshot['records']} f={snapshot['files']} b={snapshot['bytes']}"
            )
            self._last_log = self.clock()

    def _run(self):
        while not self._stop.wait(self.poll_seconds):
            try:
                self.tick()
            except Exception as error:
                self._diagnostic_error(error)

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Progress monitor was already started")
        self.publish()
        self._thread = threading.Thread(target=self._run, name=f"correctness-progress-r{self.rank}", daemon=True)
        self._thread.start()

    def fail(self, error, *, records=None, files=None, size=None):
        with self._lock:
            if self._state["status"] in ("failed", "finished"):
                return
            self._state.update(status="failed", error=str(error)[:240])
            for name, value in (("records", records), ("files", files), ("bytes", size)):
                if value is not None:
                    self._state[name] = int(value)
            self._ended = self.clock()
        self.stop()

    def finish(self):
        with self._lock:
            if self._state["status"] != "failed":
                self._state.update(status="finished", phase="finished")
                self._ended = self.clock()
                self._phase_started = self._ended
        self.stop()

    def stop(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=JOIN_SECONDS)
        with self._lock:
            if self._state["status"] == "running":
                self._state["status"] = "stopped"
                self._ended = self.clock()
        self.publish()
