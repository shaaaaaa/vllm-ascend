# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the native-latent selected source read side."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.selected_prefill_lmcache_source import (
    NativeSelectedLoadStep,
    PendingSelectedPrefill,
    build_selected_prefill_step,
    destination_rows,
    gather_layer_rows,
    lmcache_selected_supported,
    native_block_size,
    native_part_widths,
    scatter_native_rows,
    selected_lmcache_max_query,
)
from vllm_ascend.distributed.kv_transfer.sparse_offload.prefill_selected_workspace import (
    MIN_HISTORY_TOKENS,
    PAGE_ROWS,
)

WIDTH = 576
NOPE = 512
ROPE = 64


class FakePage:
    """One layer-page object exposing only what the gather needs."""

    def __init__(self, rows: int, *, flat: bool = False) -> None:
        self._rows = torch.arange(rows * WIDTH, dtype=torch.float32).view(rows, WIDTH)
        self._flat = flat
        self._layer_tensor = None

    def is_valid(self) -> bool:
        return True

    def layer_tensor(self, layer_id: int) -> torch.Tensor:
        # LMCache's MLA_LATENT page is a flat plane-stacked buffer:
        # [N * k_nope | N * k_pe], not row-major interleaved.
        tensor = (
            self._rows
            if not self._flat
            else torch.cat(
                [
                    self._rows[:, :NOPE].reshape(-1),
                    self._rows[:, NOPE:].reshape(-1),
                ]
            )
        )
        self._layer_tensor = tensor
        return tensor


class FakeLayerPageSource:
    def __init__(self, pages, layer_id: int = 3, suffix=()) -> None:
        self.pages = tuple(pages)
        self.layer_id = layer_id
        self.suffix = tuple(suffix)


def span_pages(spans, *, flat_first: bool = False):
    pages = []
    for index, (start, end) in enumerate(spans):
        page = FakePage(end - start, flat=flat_first and index == 1)
        page._rows += start * 100  # make each span recognizable
        pages.append(page)
    return FakeLayerPageSource(pages)


def test_selected_source_spans_use_pages_and_suffix():
    source = FakeLayerPageSource(
        [FakePage(4)], layer_id=2, suffix=(FakePage(3), FakePage(2))
    )
    from vllm_ascend.distributed.kv_transfer.sparse_offload.selected_prefill_lmcache_source import (
        layer_objects,
    )

    objects, layer_id = layer_objects(source)
    assert layer_id == 2
    assert len(objects) == 3
    assert isinstance(objects[0].layer_tensor(2), torch.Tensor)


def test_gather_only_selected_rows_across_spans():
    spans = ((0, 4), (4, 7), (7, 9))
    source = span_pages(spans, flat_first=True)
    rows = [0, 3, 4, 8]
    out = gather_layer_rows(source, 3, spans, rows, (NOPE, ROPE))
    assert tuple(out.shape) == (len(rows), WIDTH)
    expected = torch.cat(
        [source.pages[0]._rows[:4], source.pages[1]._rows[:3], source.pages[2]._rows[:2]]
    )
    for offset, row in enumerate(rows):
        assert torch.equal(out[offset], expected[row])


def test_gather_rejects_rows_outside_the_staged_history():
    spans = ((0, 4), (4, 7))
    source = span_pages(spans)
    with pytest.raises(ValueError, match="outside the staged source"):
        gather_layer_rows(source, 3, spans, [0, 7], (NOPE, ROPE))


def test_gather_requires_one_object_per_span():
    source = span_pages(((0, 4), (4, 8)))
    with pytest.raises(ValueError, match="objects=2, spans=3"):
        gather_layer_rows(source, 3, ((0, 4), (4, 8), (8, 9)), [0], (NOPE, ROPE))


def test_destination_rows_follow_the_native_page_table():
    block_ids = [5, 7, 9]
    rows = [0, 1, PAGE_ROWS - 1, PAGE_ROWS, 2 * PAGE_ROWS + 44]
    cache = torch.zeros((16, PAGE_ROWS, 1, WIDTH), dtype=torch.float32)
    destinations = destination_rows(block_ids, PAGE_ROWS, rows, cache)
    assert destinations.tolist() == [640, 641, 767, 7 * 128, 9 * 128 + 44]


def test_native_block_size_must_match_the_selected_page_rows():
    cache = (torch.zeros((8, 64, 1, NOPE)), torch.zeros((8, 64, 1, ROPE)))
    with pytest.raises(ValueError, match="128 rows per block"):
        native_block_size(cache)


def test_scatter_splits_nope_and_rope_into_native_pages():
    pages = 6
    k_nope = torch.zeros((pages, PAGE_ROWS, 1, NOPE), dtype=torch.bfloat16)
    k_pe = torch.zeros((pages, PAGE_ROWS, 1, ROPE), dtype=torch.bfloat16)
    source = torch.randn((3, WIDTH), dtype=torch.bfloat16)
    block_ids = [1, 4, 5]
    rows = [0, PAGE_ROWS + 2, 2 * PAGE_ROWS + PAGE_ROWS - 1]
    cache = (k_nope, k_pe)
    destinations = destination_rows(block_ids, PAGE_ROWS, rows, cache)
    written = scatter_native_rows(cache, destinations, source)
    assert written == 3 * WIDTH * source.element_size()
    flat_nope = k_nope.reshape(-1, NOPE)
    flat_pe = k_pe.reshape(-1, ROPE)
    for index, destination in enumerate(destinations.tolist()):
        assert torch.equal(flat_nope[destination], source[index, :NOPE])
        assert torch.equal(flat_pe[destination], source[index, NOPE:])


def test_scatter_accepts_one_fused_native_tensor():
    cache = torch.zeros((4, PAGE_ROWS, 1, WIDTH), dtype=torch.bfloat16)
    source = torch.randn((2, WIDTH), dtype=torch.bfloat16)
    destinations = destination_rows([0, 0, 0, 3], PAGE_ROWS, [1, 3 * PAGE_ROWS + 5], cache)
    scatter_native_rows(cache, destinations, source)
    flat = cache.reshape(-1, WIDTH)
    assert torch.equal(flat[1], source[0])
    assert torch.equal(flat[3 * PAGE_ROWS + 5], source[1])


def test_scatter_rejects_a_width_mismatch():
    cache = (torch.zeros((2, PAGE_ROWS, 1, NOPE)), torch.zeros((2, PAGE_ROWS, 1, ROPE)))
    with pytest.raises(ValueError, match="does not match the native latent cache"):
        scatter_native_rows(cache, torch.zeros(1, dtype=torch.long), torch.zeros((1, NOPE)))


class FakeRequest:
    request_id = "req-0"
    start = 0
    end = MIN_HISTORY_TOKENS


class FakeSource:
    def __init__(self) -> None:
        self.request = FakeRequest()
        self.spans = ((0, 128),)
        self.layers = ()


class FakeStepRelease:
    def __init__(self) -> None:
        self.released_with = None

    def release(self, completion=None) -> None:
        self.released_with = completion


def test_selected_adapter_exposes_protocol_fields_and_releases_once():
    source = FakeSource()
    step = FakeStepRelease()
    adapter = NativeSelectedLoadStep(step, source, ["layer.0"], {"layer.0": 0}, 128)
    request = adapter.requests[0]
    assert (request.request_id, request.history_tokens, request.query_tokens) == (
        "req-0",
        MIN_HISTORY_TOKENS,
        128,
    )
    assert request.selected is True
    completion = object()
    adapter.finish(completion=completion)
    assert step.released_with is completion
    assert adapter.try_retire() is True
    adapter.finish(completion=completion)
    assert adapter.try_retire() is True


def test_selected_adapter_rejects_unknown_layers():
    adapter = NativeSelectedLoadStep(
        FakeStepRelease(), FakeSource(), ["layer.0"], {"layer.0": 0}, 128
    )
    with pytest.raises(ValueError, match="not a latent layer"):
        adapter.load_native(
            "layer.9",
            torch.zeros((1, 8), dtype=torch.int32),
            torch.zeros((1, 8), dtype=torch.int32),
            torch.zeros((2, PAGE_ROWS, 1, WIDTH)),
            torch.zeros((1, 4), dtype=torch.int32),
        )


class FakeStep:
    def __init__(self, requests) -> None:
        self.requests = requests


def test_selected_lmcache_max_query_env(monkeypatch):
    name = "VLLM_ASCEND_SELECTED_PREFILL_LMCACHE_MAX_QUERY"
    monkeypatch.setenv(name, "128")
    assert selected_lmcache_max_query() == 128
    monkeypatch.setenv(name, "1")
    with pytest.raises(ValueError, match="at least 2"):
        selected_lmcache_max_query()
    monkeypatch.setenv(name, "not-an-int")
    with pytest.raises(ValueError, match="must be an integer"):
        selected_lmcache_max_query()


def test_builder_requires_the_minimum_history(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "lmcache.v1.selected_prefill",
        SimpleNamespace(
            SelectedPrefillRequest=lambda request_id, start, end: SimpleNamespace(
                request_id=request_id, start=start, end=end
            ),
            SelectedPrefillStep=FakeStep,
        ),
    )
    assert build_selected_prefill_step("req-0", MIN_HISTORY_TOKENS - 1) is None
    pending = build_selected_prefill_step("req-0", MIN_HISTORY_TOKENS)
    assert isinstance(pending, PendingSelectedPrefill)
    assert pending.request_id == "req-0"
    assert pending.history_tokens == MIN_HISTORY_TOKENS
    request = pending.step.requests[0]
    assert (request.request_id, request.start, request.end) == (
        "req-0",
        0,
        MIN_HISTORY_TOKENS,
    )


def test_connector_support_requires_the_explicit_handoff_parameter():
    def with_parameter(forward_context, *, selected_prefill_step=None, **kwargs):
        pass

    def kwargs_only(forward_context, **kwargs):
        pass

    assert lmcache_selected_supported(
        SimpleNamespace(_lmcache_engine=SimpleNamespace(start_load_kv=with_parameter))
    )
    assert not lmcache_selected_supported(
        SimpleNamespace(_lmcache_engine=SimpleNamespace(start_load_kv=kwargs_only))
    )
    assert lmcache_selected_supported(
        SimpleNamespace(
            _lmcache_engine=SimpleNamespace(start_load_kv=kwargs_only),
            supports_selected_prefill_sources=True,
        )
    )

    class Base:
        def start_load_kv(self, forward_context, *, selected_prefill_step=None, **kwargs):
            pass

    class AscendOverride(Base):
        def start_load_kv(self, forward_context, **kwargs):
            return super().start_load_kv(forward_context, **kwargs)

    assert lmcache_selected_supported(
        SimpleNamespace(_lmcache_engine=AscendOverride())
    )


def test_flat_layer_page_is_plane_stacked_not_row_major():
    page = FakePage(4, flat=True)
    out = gather_layer_rows(
        FakeLayerPageSource([page]), 3, ((0, 4),), [1], (NOPE, ROPE)
    )
    assert torch.equal(out[0], page._rows[1])
    wrong = page.layer_tensor(3).view(4, WIDTH)[1]
    assert not torch.equal(out[0], wrong)


def test_native_part_widths_reads_the_last_dimension():
    cache = (
        torch.zeros((2, PAGE_ROWS, 1, NOPE)),
        torch.zeros((2, PAGE_ROWS, 1, ROPE)),
    )
    assert native_part_widths(cache) == (NOPE, ROPE)
    assert native_part_widths(torch.zeros((2, PAGE_ROWS, 1, WIDTH))) == (WIDTH,)


def test_gather_accepts_a_singleton_leading_page_dimension():
    rows = torch.arange(4 * WIDTH, dtype=torch.float32).view(1, 4, WIDTH)
    page = FakePage(4)
    page.layer_tensor = lambda layer_id: rows.view(1, 4, WIDTH)
    out = gather_layer_rows(
        FakeLayerPageSource([page]), 3, ((0, 4),), [2], (NOPE, ROPE)
    )
    assert torch.equal(out[0], rows[0, 2])
