# SPDX-License-Identifier: Apache-2.0
"""Compatibility and wire contracts for the extracted live-split protocol."""

import pickle

import msgspec
import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p import live_split_protocol as protocol
from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_connector as connector


@pytest.mark.parametrize(
    "name",
    [
        "SplitTransferSegment",
        "SplitSourceSegment",
        "SplitCompactLayer",
        "SplitCompactRun",
        "SplitCompactLayout",
        "SplitLatentRun",
        "SplitLatentPage",
        "SplitLatentLayout",
        "SplitLatentDestinationPage",
        "SplitSourceDescriptor",
        "SplitTransferPlan",
    ],
)
def test_connector_reexports_same_record_classes(name):
    assert getattr(connector, name) is getattr(protocol, name)


@pytest.mark.parametrize(
    "name",
    [
        "_parse_compact_layout",
        "_parse_latent_layout",
        "_parse_source_descriptor",
        "_parse_split_plan",
    ],
)
def test_metadata_parser_aliases_call_protocol_directly(name):
    assert getattr(connector.MooncakeConnectorMetadata, name) is getattr(protocol, name[1:])


def test_partial_page_plan_roundtrip_preserves_source_destination_ranks():
    wire = {
        "segments": [],
        "group_byte_totals": [24, 0],
        "tp_rank": 2,
        "dp_rank": 1,
        "source_tp_rank": 2,
        "source_dp_rank": 0,
        "requested_groups": [0],
        "latent_source": {
            "group_id": 0,
            "token_count": 3,
            "layers": [{"layer_id": 0, "buffer_base": 1000, "token_bytes": 8, "slot_capacity": 8, "buffer_index": 0}],
            "pages": [
                {
                    "logical_token_start": 0,
                    "token_count": 3,
                    "runs": [{"logical_token_start": 0, "physical_slot_start": 1, "token_count": 3}],
                }
            ],
        },
        "latent_destination_pages": [
            {"logical_token_start": 0, "destination_address": 2000, "length": 24},
        ],
    }
    parsed = protocol.parse_split_plan(wire)
    assert parsed.source_rank == (2, 0)
    assert parsed.destination_rank == (2, 1)
    assert parsed.latent_source.pages[0].token_count == 3
    assert pickle.loads(pickle.dumps(parsed)) == parsed
    restored = msgspec.msgpack.decode(msgspec.msgpack.encode(parsed), type=protocol.SplitTransferPlan)
    assert restored == parsed
    assert connector.MooncakeConnectorMetadata._parse_split_plan(restored) is restored


@pytest.mark.parametrize("value", [False, True, 1.5, 1 << 63, -(1 << 63)])
def test_wire_integer_rejects_lossy_or_out_of_range_values(value):
    with pytest.raises(ValueError):
        protocol.wire_int(value, "field")
