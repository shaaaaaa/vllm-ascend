# SPDX-License-Identifier: Apache-2.0
"""Backport same-step KV connector worker metadata to the stock scheduler."""

from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler

_original_update_from_output = Scheduler.update_from_output


def _update_from_output(
    self: Scheduler,
    scheduler_output: Any,
    model_runner_output: Any,
) -> Any:
    connector_output = model_runner_output.kv_connector_output
    if (
        connector_output
        and connector_output.kv_connector_worker_meta is not None
        and self.connector
    ):
        update = getattr(
            self.connector, "update_connector_worker_metadata", None
        )
        if callable(update):
            update(
                connector_output.kv_connector_worker_meta,
                {
                    req_id
                    for req_id, request in self.requests.items()
                    if not request.is_finished()
                },
            )
    return _original_update_from_output(
        self, scheduler_output, model_runner_output
    )


Scheduler.update_from_output = _update_from_output
