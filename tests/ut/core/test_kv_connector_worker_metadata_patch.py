from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_ascend.patch.platform import (
    patch_kv_connector_worker_metadata as patch,
)


def test_worker_metadata_precedes_stock_scheduler_update(monkeypatch):
    calls = []
    connector = MagicMock()
    connector.update_connector_worker_metadata.side_effect = (
        lambda *_args: calls.append("metadata")
    )
    scheduler = SimpleNamespace(
        connector=connector,
        requests={
            "active": SimpleNamespace(is_finished=lambda: False),
            "finished": SimpleNamespace(is_finished=lambda: True),
        },
    )
    worker_metadata = object()
    model_output = SimpleNamespace(
        kv_connector_output=SimpleNamespace(
            kv_connector_worker_meta=worker_metadata
        )
    )
    monkeypatch.setattr(
        patch,
        "_original_update_from_output",
        lambda *_args: calls.append("output"),
    )

    patch._update_from_output(scheduler, object(), model_output)

    assert calls == ["metadata", "output"]
    connector.update_connector_worker_metadata.assert_called_once_with(
        worker_metadata, {"active"}
    )
