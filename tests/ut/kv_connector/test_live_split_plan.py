# SPDX-License-Identifier: Apache-2.0
"""Ownership contracts for pure live-split plan construction."""

from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace

from vllm_ascend.distributed.kv_transfer.kv_p2p import live_split_plan as planning
from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_connector as connector
from vllm_ascend.distributed.kv_transfer.kv_p2p.live_split_protocol import (
    SplitSourceDescriptor,
    SplitSourceSegment,
)


def test_canonicalization_borrows_handshake_without_copy_or_iteration():
    metadata = SimpleNamespace(
        kv_caches_base_addr=[1000],
        kv_caches_buffer_sizes=(128,),
        buffer_group_ids=(1,),
    )
    lookups = []

    class BorrowedRegistry(Mapping):
        def __getitem__(self, key):
            lookups.append(key)
            if key != (2, 1):
                raise KeyError(key)
            return metadata

        def __iter__(self):
            raise AssertionError("planning must not copy the handshake registry")

        def __len__(self):
            raise AssertionError("planning must not scan the handshake registry")

    raw = {
        "tp_rank": 2,
        "dp_rank": 1,
        "group_byte_totals": [0, 8],
        "segments": [
            {
                "group_id": 1,
                "source_buffer_base": 1000,
                "source_offset": 16,
                "length": 8,
            }
        ],
    }
    original = deepcopy(raw)
    registry = BorrowedRegistry()
    result = planning.canonicalize_source_descriptor(raw, registry, (1,))
    assert lookups == [(2, 1)]
    assert raw == original
    assert metadata.kv_caches_base_addr == [1000]
    assert result["descriptors"][0]["segments"][0]["source_buffer_index"] == 0
    scheduler = connector.MooncakeConnectorScheduler.__new__(connector.MooncakeConnectorScheduler)
    scheduler.local_source_metadata = registry
    scheduler.live_split_source_groups = (1,)
    assert scheduler._canonicalize_source_descriptor(raw) == result
    assert lookups == [(2, 1), (2, 1)]


def test_merge_preserves_source_diagnostic_object_and_destination_input():
    diagnostics = {"combined_hash": "ab" * 16}
    source = SplitSourceDescriptor(
        segments=(SplitSourceSegment(1, 0, 1000, 16, 8),),
        group_byte_totals=(0, 8),
        tp_rank=2,
        dp_rank=0,
        content_diagnostics=diagnostics,
    )
    destination = {
        "group_byte_totals": [0, 8],
        "requested_groups": [1],
        "tp_rank": 2,
        "dp_rank": 1,
        "segments": [{"group_id": 1, "destination_address": 2000, "length": 8, "destination_kind": "npu"}],
    }
    original = deepcopy(destination)
    result = planning.merge_source_and_destinations(source, destination)
    assert result.content_diagnostics is diagnostics
    assert destination == original
    assert result.source_rank == (2, 0)
    assert result.destination_rank == (2, 1)
    assert result.segments[0].source_offset == 16
    assert result.segments[0].destination_address == 2000


def test_existing_planning_entrypoints_are_direct_aliases():
    assert connector.KVCacheRecvingThread._expand_compact_split_plan is planning.expand_compact_split_plan
    assert connector.MooncakeConnectorMetadata._merge_source_and_destinations is planning.merge_source_and_destinations
