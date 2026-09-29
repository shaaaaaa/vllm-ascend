# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the LoCoMo API entry and unchanged tensor probe lifecycle."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from layerwise_prefill_correctness_server import capture_request


@pytest.mark.parametrize("output_kind", ["DELTA", "CUMULATIVE", "FINAL_ONLY"])
def test_one_request_preserves_api_outputs_and_finishes_capture(tmp_path, output_kind):
    directory = tmp_path / "off"
    directory.mkdir()
    (directory / "engine_options.json").write_text(json.dumps({"tensor_parallel_size": 1}))
    ids = [7] * 4099
    calls = []

    async def rpc(method, **kwargs):
        calls.append((method, kwargs))
        return [{"rank": 0, "complete": True, "errors": []}]

    first = NS(finished=False, num_cached_tokens=0, outputs=[NS(token_ids=[8], text="a")])
    last = NS(
        finished=True,
        num_cached_tokens=0,
        outputs=[NS(token_ids=[9] if output_kind == "DELTA" else [8, 9], text="b" if output_kind == "DELTA" else "ab")],
    )

    async def original(client, prompt, params, request_id, **kwargs):
        assert prompt == {"prompt_token_ids": ids} and kwargs == {"priority": 3}
        if output_kind != "FINAL_ONLY":
            yield first
        yield last

    wrapper = capture_request(
        original,
        root=tmp_path,
        case="off",
        max_features=8,
        save_on_tensors=False,
        timeout=1800,
        extract_tokens=lambda client, prompt: prompt["prompt_token_ids"],
    )

    async def run():
        stream = wrapper(
            NS(collective_rpc=rpc),
            {"prompt_token_ids": ids},
            NS(n=1, output_kind=NS(name=output_kind)),
            "request-1",
            priority=3,
        )
        received = []
        async for result in stream:
            received.append(result)
            if result.finished:
                # API consumers can stop immediately after a final response.
                assert json.loads((directory / "result.json").read_text())["completed"]
                await stream.aclose()
                break
        return received

    received = asyncio.run(run())
    assert received[-1] is last
    assert [name for name, _ in calls] == ["install_correctness_probe", "finish_correctness_probe"]
    assert calls[0][1]["args"] == (str(directory), 4099, False, 8)
    result = json.loads((directory / "result.json").read_text())
    assert result["completed"] and result["token_ids"] == [8, 9] and result["text"] == "ab"
    assert result["output_length"] == 2 and result["prompt_token_ids"] == ids
    assert json.loads((tmp_path / "prompt.json").read_text())["token_ids"] == ids


def test_on_different_locomo_prompt_is_rejected_before_model_execution(tmp_path):
    (tmp_path / "on").mkdir()
    (tmp_path / "prompt.json").write_text(json.dumps({"token_ids": [7] * 4099}))

    def unexpected(*args, **kwargs):
        pytest.fail("Mismatched request must not install the probe or run the model")

    wrapper = capture_request(
        unexpected,
        root=tmp_path,
        case="on",
        max_features=8,
        save_on_tensors=False,
        timeout=1800,
        extract_tokens=lambda *_: [8] * 4099,
    )

    async def run():
        async for _ in wrapper(NS(collective_rpc=unexpected), {}, NS(n=1), "request-2"):
            pass

    with pytest.raises(ValueError, match="prompt token IDs differ"):
        asyncio.run(run())
    assert not json.loads((tmp_path / "on/result.json").read_text())["completed"]


def test_incomplete_probe_cannot_be_reported_as_successful_request(tmp_path):
    directory = tmp_path / "off"
    directory.mkdir()
    (directory / "engine_options.json").write_text(json.dumps({"tensor_parallel_size": 1}))

    async def rpc(method, **kwargs):
        return [{"rank": 0, "complete": False, "errors": ["Missing MTP output"]}]

    async def original(*args, **kwargs):
        yield NS(finished=True, num_cached_tokens=0, outputs=[NS(token_ids=[1], text="x")])

    wrapper = capture_request(
        original,
        root=tmp_path,
        case="off",
        max_features=8,
        save_on_tensors=False,
        timeout=1800,
        extract_tokens=lambda *_: [7] * 4099,
    )

    async def run():
        async for _ in wrapper(NS(collective_rpc=rpc), {}, NS(n=1, output_kind=NS(name="FINAL_ONLY")), "r"):
            pytest.fail("Incomplete coverage must not yield a successful final response")

    with pytest.raises(ValueError, match="Missing MTP output"):
        asyncio.run(run())
    assert "Missing MTP output" in json.loads((directory / "result.json").read_text())["error"]
