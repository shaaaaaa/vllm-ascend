# SPDX-License-Identifier: Apache-2.0
"""Compact host-only evidence of request membership in executed P batches."""

from __future__ import annotations

import os
import socket
from collections.abc import Callable, Sequence
from typing import Any


class PrefillBatchLogger:
    """Log one main-model submission, including requests with no cache transfer.

    The caller supplies only CPU scheduler/input-batch metadata. No device
    objects, events, payload reads, or synchronization are needed here.
    """

    def __init__(self, emit: Callable[[str], None], *, dp_rank: int, pp_rank: int):
        self.emit = emit
        self.identity = f"{socket.gethostname()}:{os.getpid()}"
        self.dp_rank = dp_rank
        self.pp_rank = pp_rank
        self.sequence = 0

    def log(self, batch: Any, scheduler_output: Any, ubatches: Sequence[Any] | None = None) -> None:
        """Emit actual batch order and half-open request-local token intervals."""
        self.sequence += 1
        batch_id = f"{self.identity}:{self.sequence}"
        try:
            rows = []
            offset = 0
            for index, req_id in enumerate(batch.req_ids[: batch.num_reqs]):
                count = int(scheduler_output.num_scheduled_tokens[req_id])
                start = int(batch.num_computed_tokens_cpu[index])
                prompt = int(batch.num_prompt_tokens[index])
                end = start + count
                mode = "decode" if start >= prompt else "prefill" if end <= prompt else "mixed"
                rows.append((req_id, start, end, prompt, offset, offset + count, mode))
                offset += count
            common = (
                f"[P_BATCH] b={batch_id} dp={self.dp_rank} pp={self.pp_rank} n={len(rows)} mixed={int(len(rows) > 1)}"
            )
            # Microbatch token intervals let readers distinguish sharing the
            # outer batch from sharing a DBO sub-batch. Empty padding is absent.
            ranges = (
                ",".join(f"{u.token_slice.start}:{u.token_slice.stop}" for u in ubatches) if ubatches else f"0:{offset}"
            )
            self.emit(f"{common} stage=submit model=main q={offset} ubatch={ranges}")
            for index, (req, start, end, prompt, left, right, mode) in enumerate(rows):
                # repr escapes newlines in caller-supplied IDs; each record
                # remains one line and full server IDs can still be grepped.
                self.emit(
                    f"{common} i={index} req={req!r} span=[{start},{end}) "
                    f"prompt={prompt} flat=[{left},{right}) mode={mode} "
                    f"last_prefill={int(start < prompt <= end)}"
                )
        except Exception as error:
            # A diagnostic must never abort inference. A missing record is an
            # explicit error, not silently interpreted as a single-request batch.
            self.emit(f"[P_BATCH] b={batch_id} error={str(error)!r}")
