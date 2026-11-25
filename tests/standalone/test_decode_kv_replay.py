# SPDX-License-Identifier: Apache-2.0
"""CPU snapshot metadata validation, including partial chunks."""

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
import decode_kv_replay as R  # noqa: E402


def capture(root, rank=0, size=1, count=5):
    directory = root / "request" / f"rank{rank}"
    directory.mkdir(parents=True)
    tokens = list(range(count))
    (directory / "tokens.json").write_text(json.dumps(tokens))
    files = []
    for group, layers in enumerate((2, 2)):
        for layer in range(layers):
            for component in range(2):
                # Planar K/V or C8 key/scale. Intentionally non-uniform rows.
                name = f"g{group}-l{layer}-c{component}.bin"
                values = bytes((group * 30 + layer * 10 + component * 3 + t) % 256 for t in tokens)
                (directory / name).write_bytes(values)
                files.append(
                    dict(
                        group=group,
                        layer=layer,
                        component=component,
                        row_bytes=1,
                        dtype="torch.uint8",
                        path=name,
                        start=0,
                        end=count,
                        bytes=count,
                    )
                )
    (directory / "manifest.json").write_text(
        json.dumps(
            dict(
                format="decode-kv-raw-v1",
                status="complete",
                request_id="request",
                tp_rank=rank,
                tp_size=size,
                tokens=count,
                byteorder=sys.byteorder,
                groups=[2, 2],
                layer_names=[["l0", "l1"], ["i0", "i1"]],
                files=files,
                replay_file="replay.json",
            )
        )
    )
    (directory / "replay.json").write_text(
        json.dumps(
            dict(
                request=dict(schema=1, prompt_token_ids=tokens + [99], sampling={"temperature": 0}, cache_salt=None),
                runtime=dict(
                    engine_options=dict(
                        model="model",
                        tokenizer="model",
                        tensor_parallel_size=size,
                        enforce_eager=False,
                        max_model_len=count + 1000,
                        additional_config={},
                    ),
                    environment={},
                    chunk_size=4,
                ),
            )
        )
    )
    return directory


def make_plan(root, output, size=1, count=5):
    for rank in range(size):
        capture(root, rank, size, count)
    output.mkdir()
    plan = R.build_plan([root], "request", output)
    path = output / "plan.json"
    path.write_text(json.dumps(plan))
    return path, plan


def test_missing_replay_metadata_and_tp_disagreement_fail(tmp_path):
    first = capture(tmp_path, 0, 2)
    second = capture(tmp_path, 1, 2)
    payload = second / "g0-l0-c0.bin"
    payload.write_bytes(b"BAD!!")
    with pytest.raises(ValueError, match="latent bytes differ"):
        R.build_plan([tmp_path], "request", tmp_path / "out")
    meta = json.loads((first / "manifest.json").read_text())
    meta.pop("replay_file")
    (first / "manifest.json").write_text(json.dumps(meta))
    (first / "replay.json").unlink()
    with pytest.raises(ValueError, match="matching client request"):
        R.build_plan([tmp_path], "request", tmp_path / "out")


def test_500k_prefix_preserves_actual_d_input(tmp_path):
    from decode_kv_live_replay import request_payload

    _, plan = make_plan(tmp_path / "data", tmp_path / "out", count=500000)
    payload = request_payload(plan["request"], "served-model", "new-replay", 128)
    assert len(plan["received_tokens"]) == 500000
    assert payload["prompt"] == list(range(500000)) + [99]
    assert payload["max_tokens"] == 128
    assert payload["temperature"] == 0
    assert payload["model"] == "served-model"
