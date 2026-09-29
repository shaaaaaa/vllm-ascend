# SPDX-License-Identifier: Apache-2.0
"""Exercise actual recompute admission without the NPU/model bootstrap."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


def admission(length=25345, free_blocks=10000, **overrides):
    path = Path(__file__).parents[2] / "vllm_ascend/core/recompute_scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Keep the actual allocation, allocation-failure break and connector call.
    body = next(
        node.body
        for node in ast.walk(tree)
        if isinstance(node, ast.While)
        and any(
            isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "effective_lookahead_tokens" for target in stmt.targets
            )
            for stmt in node.body
        )
    )
    first = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "effective_lookahead_tokens" for target in node.targets)
    )
    last = next(
        i
        for i, node in enumerate(body)
        if i > first
        and isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "request" for target in node.targets)
    )
    request = NS(
        request_id="r",
        num_tokens=length,
        num_prompt_tokens=length,
        num_computed_tokens=0,
        num_preemptions=0,
        has_encoder_inputs=False,
    )
    for key in tuple(overrides):
        if key.startswith("request_"):
            setattr(request, key[8:], overrides.pop(key))
    before = vars(request).copy()
    allocated = []

    def allocate(req, new_tokens, **kwargs):
        end = (
            req.num_computed_tokens
            + new_tokens
            + kwargs["num_new_computed_tokens"]
            + kwargs["num_external_computed_tokens"]
            + kwargs["num_lookahead_tokens"]
        )
        count = (end + 127) // 128
        if count > free_blocks:
            return None
        allocated[:] = range(1, count + 1)
        return allocated

    manager = NS(allocate_slots=Mock(side_effect=allocate), get_blocks=lambda _: allocated)
    connector = NS(update_state_after_alloc=Mock())
    scope = dict(
        self=NS(
            block_size=128,
            num_lookahead_tokens=4,
            is_encoder_decoder=False,
            kv_cache_manager=manager,
            connector=connector,
            connector_prefix_cache_stats=None,
        ),
        request=request,
        request_id="r",
        num_new_tokens=0,
        num_new_local_computed_tokens=0,
        new_computed_blocks=None,
        num_external_computed_tokens=length - 1,
        load_kv_async=True,
        dsa_compact_external_load=True,
    )
    scope.update(overrides)
    module = ast.parse("for _ in range(1): pass")
    module.body[0].body = body[first:last]
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    assert vars(request) == before  # Reserving storage must not advance compute.
    assert manager.allocate_slots.call_args.args[1] == scope["num_new_tokens"]
    return manager.allocate_slots.call_args.kwargs, allocated, connector


@pytest.mark.parametrize("length,extra", [(25344, 0), (25345, 1), (25346, 0), (81665, 1)])
def test_fresh_compact_restore_reserves_full_hit(length, extra):
    kwargs, blocks, connector = admission(length)
    assert kwargs["num_lookahead_tokens"] == extra
    assert len(blocks) == (length + 127) // 128
    assert kwargs["num_external_computed_tokens"] == length - 1
    assert kwargs["delay_cache_blocks"] is True
    assert kwargs["dsa_compact_external_load"] is True
    connector.update_state_after_alloc.assert_called_once()


@pytest.mark.parametrize(
    "overrides",
    [
        {"load_kv_async": False},
        {"dsa_compact_external_load": False},
        {"request_num_preemptions": 1},
        {"num_new_local_computed_tokens": 128},
        {"request_num_prompt_tokens": 25344},
        {"num_external_computed_tokens": 25000},
    ],
)
def test_unrelated_admissions_keep_zero_initial_lookahead(overrides):
    kwargs, _, _ = admission(**overrides)
    assert kwargs["num_lookahead_tokens"] == 0


def test_existing_request_keeps_speculative_lookahead():
    kwargs, _, _ = admission(request_num_computed_tokens=128)
    assert kwargs["num_lookahead_tokens"] == 4


@pytest.mark.parametrize("free_blocks", [198, 199])
def test_insufficient_capacity_defers_before_connector_metadata(free_blocks):
    _, blocks, connector = admission(free_blocks=free_blocks)
    if free_blocks == 198:
        assert blocks == []
        connector.update_state_after_alloc.assert_not_called()
    else:
        assert len(blocks) == 199
        connector.update_state_after_alloc.assert_called_once()
