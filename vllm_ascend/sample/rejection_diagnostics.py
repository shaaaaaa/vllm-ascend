from collections.abc import Callable
from contextvars import ContextVar, Token
from functools import wraps
from typing import Any

StageRecorder = Callable[[str, Callable[..., Any], tuple[Any, ...], dict[str, Any]], Any]
_active_recorder: ContextVar[StageRecorder | None] = ContextVar("ascend_rejection_stage_recorder", default=None)


def set_stage_recorder(recorder: StageRecorder) -> Token:
    return _active_recorder.set(recorder)


def reset_stage_recorder(token: Token) -> None:
    _active_recorder.reset(token)


def stage_recorder_active() -> bool:
    return _active_recorder.get() is not None


def record_stage(name: str, operation: Callable[..., Any], *args, **kwargs):
    recorder = _active_recorder.get()
    if recorder is None:
        return operation(*args, **kwargs)
    return recorder(name, operation, args, kwargs)


def diagnostic_stage(name: str):
    def decorate(operation):
        @wraps(operation)
        def wrapped(*args, **kwargs):
            return record_stage(name, operation, *args, **kwargs)

        return wrapped

    return decorate
