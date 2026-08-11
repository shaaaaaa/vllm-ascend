# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from vllm_ascend.sfa_flight_recorder import SFAFlightRecorder


def test_sfa_flight_recorder_dumps_jsonl_on_failure(tmp_path: Path) -> None:
    recorder = SFAFlightRecorder(
        str(tmp_path),
        {"dp_rank": 1, "device": "npu:5"},
    )
    recorder.record(
        "boundary_copy_begin",
        is_dummy=True,
        is_capturing=False,
        boundary_shape=(2,),
    )

    path = recorder.dump("execute_dummy_batch_failed", RuntimeError("boom"))

    assert path is not None
    payloads = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert payloads[0] == {
        "schema_version": 1,
        "kind": "header",
        "pid": payloads[0]["pid"],
        "reason": "execute_dummy_batch_failed",
        "error_type": "RuntimeError",
        "error": "boom",
        "event_count": 1,
        "dp_rank": 1,
        "device": "npu:5",
    }
    assert payloads[1]["phase"] == "boundary_copy_begin"
    assert payloads[1]["is_dummy"] is True
    assert payloads[1]["is_capturing"] is False
    assert payloads[1]["boundary_shape"] == [2]


def test_sfa_flight_recorder_does_not_raise_when_dump_fails(
    tmp_path: Path,
) -> None:
    occupied_path = tmp_path / "not-a-directory"
    occupied_path.write_text("occupied", encoding="utf-8")
    recorder = SFAFlightRecorder(str(occupied_path))

    assert recorder.dump("failure", RuntimeError("boom")) is None
