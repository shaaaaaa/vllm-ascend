# SPDX-License-Identifier: Apache-2.0
"""Live-split plan construction from borrowed layouts and handshake metadata.

These functions own no transfer/request state. They return the existing wire
mappings or records without copying the handshake registry or retaining owners.
Callers retain registration, thread/rank mutation, completion and cleanup duties;
worker registered-address validation still precedes native submission.
"""

from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Protocol

from vllm_ascend.distributed.kv_transfer.kv_p2p.live_split_protocol import (
    SplitLatentDestinationPage,
    SplitSourceDescriptor,
    SplitTransferPlan,
    SplitTransferSegment,
    parse_compact_layout,
    parse_latent_layout,
    sanitize_content_diagnostics,
    wire_int,
)


class SourceHandshake(Protocol):
    """Read-only use of the existing registered source metadata."""

    kv_caches_base_addr: list[int]
    kv_caches_buffer_sizes: tuple[int, ...]
    buffer_group_ids: tuple[int, ...]


def expand_compact_split_plan(
    plan: SplitTransferPlan,
) -> SplitTransferPlan:
    source, destination = plan.compact_source, plan.compact_destination
    if source is None or destination is None or plan.requested_groups not in ((1,), (0, 1)):
        raise RuntimeError("Incomplete compact split transfer plan")
    if any(layer.buffer_index < 0 for layer in source.layers):
        raise RuntimeError("Compact split source buffer is unresolved")
    source_layers = {layer.layer_id: layer for layer in source.layers}
    destination_layers = {layer.layer_id: layer for layer in destination.layers}
    if (
        len(source_layers) != len(source.layers)
        or len(destination_layers) != len(destination.layers)
        or source_layers.keys() != destination_layers.keys()
    ):
        raise RuntimeError("Compact split layer sets differ")
    segments: list[SplitTransferSegment] = []
    if 0 in plan.requested_groups:
        latent = plan.latent_source
        pages = plan.latent_destination_pages
        if latent is None or len(pages) != len(latent.pages):
            raise RuntimeError("Incomplete latent CPU split transfer plan")
        if any(layer.buffer_index < 0 for layer in latent.layers):
            raise RuntimeError("Latent split source buffer is unresolved")
        for source_page, destination_page in zip(latent.pages, pages):
            destination_offset = 0
            for layer in latent.layers:
                for run in source_page.runs:
                    logical_offset = run.logical_token_start - source_page.logical_token_start
                    if logical_offset < 0:
                        raise RuntimeError("Latent run precedes its page")
                    length = run.token_count * layer.token_bytes
                    segments.append(
                        SplitTransferSegment(
                            group_id=0,
                            source_buffer_index=layer.buffer_index,
                            source_buffer_base=layer.buffer_base,
                            source_offset=(run.physical_slot_start * layer.token_bytes),
                            destination_address=(
                                destination_page.destination_address
                                + destination_offset
                                + logical_offset * layer.token_bytes
                            ),
                            length=length,
                            destination_kind="cpu",
                        )
                    )
                destination_offset += source_page.token_count * layer.token_bytes
            if destination_offset != destination_page.length:
                raise RuntimeError("Expanded latent destination page byte total differs")
    src_pos = dst_pos = src_offset = dst_offset = 0
    while src_pos < len(source.runs) and dst_pos < len(destination.runs):
        src_run, dst_run = source.runs[src_pos], destination.runs[dst_pos]
        src_logical = src_run.logical_token_start + src_offset
        dst_logical = dst_run.logical_token_start + dst_offset
        if src_logical != dst_logical:
            raise RuntimeError("Compact split logical coverage differs")
        count = min(
            src_run.token_count - src_offset,
            dst_run.token_count - dst_offset,
        )
        for layer_id in source_layers:
            src_layer = source_layers[layer_id]
            dst_layer = destination_layers[layer_id]
            if src_layer.token_bytes != dst_layer.token_bytes:
                raise RuntimeError("Compact split token widths differ")
            segments.append(
                SplitTransferSegment(
                    group_id=1,
                    source_buffer_index=src_layer.buffer_index,
                    source_buffer_base=src_layer.buffer_base,
                    source_offset=(src_run.physical_slot_start + src_offset) * src_layer.token_bytes,
                    destination_address=dst_layer.buffer_base
                    + (dst_run.physical_slot_start + dst_offset) * dst_layer.token_bytes,
                    length=count * src_layer.token_bytes,
                    destination_kind="npu",
                )
            )
        src_offset += count
        dst_offset += count
        if src_offset == src_run.token_count:
            src_pos += 1
            src_offset = 0
        if dst_offset == dst_run.token_count:
            dst_pos += 1
            dst_offset = 0
    if src_pos != len(source.runs) or dst_pos != len(destination.runs):
        raise RuntimeError("Compact split run coverage differs")
    totals = tuple(sum(segment.length for segment in segments if segment.group_id == group_id) for group_id in range(2))
    if totals != plan.group_byte_totals:
        raise RuntimeError("Expanded compact split byte totals differ")
    return SplitTransferPlan(
        segments=tuple(segments),
        group_byte_totals=plan.group_byte_totals,
        tp_rank=plan.tp_rank,
        dp_rank=plan.dp_rank,
        requested_groups=plan.requested_groups,
        source_tp_rank=plan.source_tp_rank,
        source_dp_rank=plan.source_dp_rank,
        content_diagnostics=plan.content_diagnostics,
    )


def merge_source_and_destinations(
    source: SplitSourceDescriptor,
    plan: Any,
    supported_groups: tuple[int, ...] = (0, 1),
) -> SplitTransferPlan:
    if not isinstance(plan, dict):
        raise ValueError("Live split destination plan must be a mapping")
    dest_totals = tuple(wire_int(value, "destination.group_byte_total") for value in plan["group_byte_totals"])
    requested_group_values = tuple(
        wire_int(value, "destination.requested_group") for value in plan.get("requested_groups", (0, 1))
    )
    if len(set(requested_group_values)) != len(requested_group_values) or not set(requested_group_values).issubset(
        {0, 1}
    ):
        raise ValueError("Destination requested groups are invalid")
    requested_groups = tuple(group for group in requested_group_values if group in supported_groups)
    if len(dest_totals) != 2 or not requested_groups or not set(requested_groups).issubset({0, 1}):
        raise ValueError(f"invalid destination totals/groups: totals={dest_totals}, groups={requested_groups}")
    mismatched_totals = {
        group: (source.group_byte_totals[group], dest_totals[group])
        for group in requested_groups
        if dest_totals[group] != source.group_byte_totals[group]
    }
    if mismatched_totals:
        raise ValueError(f"source/destination byte totals differ (source, destination)={mismatched_totals}")
    tp_rank = wire_int(plan["tp_rank"], "destination.tp_rank")
    dp_rank = wire_int(plan["dp_rank"], "destination.dp_rank")
    if tp_rank != source.tp_rank:
        raise ValueError(
            "source/destination TP ranks differ: "
            f"source={(source.tp_rank, source.dp_rank)}, "
            f"destination={(tp_rank, dp_rank)}"
        )
    compact_destination = parse_compact_layout(plan.get("compact_layout"))
    latent_destination_segments = tuple(plan.get("latent_pages", ()))
    hybrid = 0 in requested_groups
    latent_pages: tuple[SplitLatentDestinationPage, ...] = ()
    if hybrid:
        try:
            destination_plane_widths = tuple(
                wire_int(width, "destination.latent_token_bytes") for width in plan["latent_token_bytes"]
            )
        except (KeyError, TypeError, ValueError):
            destination_plane_widths = ()
        source_plane_widths = (
            tuple(layer.token_bytes for layer in source.latent_layout.layers)
            if source.latent_layout is not None
            else ()
        )
        if (
            requested_groups != (0, 1)
            or source.latent_layout is None
            or not latent_destination_segments
            or destination_plane_widths != source_plane_widths
            or any(
                wire_int(segment["length"], "destination.latent_page_length") <= 0
                or wire_int(
                    segment["destination_address"],
                    "destination.latent_page_address",
                )
                <= 0
                for segment in latent_destination_segments
            )
        ):
            raise ValueError("Incomplete latent CPU split plan")
        page_bytes = sum(destination_plane_widths)
        if len(latent_destination_segments) != len(source.latent_layout.pages):
            raise ValueError("Latent source/destination page counts differ")
        latent_pages = tuple(
            SplitLatentDestinationPage(
                logical_token_start=wire_int(
                    segment["logical_token_start"],
                    "destination.latent_page_logical_start",
                ),
                destination_address=wire_int(
                    segment["destination_address"],
                    "destination.latent_page_address",
                ),
                length=wire_int(
                    segment["length"],
                    "destination.latent_page_length",
                ),
            )
            for segment in latent_destination_segments
        )
        if any(
            destination.length != source_page.token_count * page_bytes
            or destination.logical_token_start != source_page.logical_token_start
            or wire_int(
                raw_page.get("valid_tokens", -1),
                "destination.latent_page_valid_tokens",
            )
            != source_page.token_count
            for destination, source_page, raw_page in zip(
                latent_pages,
                source.latent_layout.pages,
                latent_destination_segments,
            )
        ):
            raise ValueError("Latent destination page byte lengths differ")
    if source.compact_layout is not None or compact_destination is not None:
        if (
            requested_groups not in ((1,), (0, 1))
            or source.compact_layout is None
            or compact_destination is None
            or source.compact_layout.token_count != compact_destination.token_count
            or len(source.compact_layout.layers) != len(compact_destination.layers)
        ):
            raise ValueError("Compact source/destination layouts differ")
        source_widths = [layer.token_bytes for layer in source.compact_layout.layers]
        destination_widths = [layer.token_bytes for layer in compact_destination.layers]
        if source_widths != destination_widths:
            raise ValueError("Compact source/destination token widths differ")
        return SplitTransferPlan(
            segments=(),
            group_byte_totals=tuple(
                source.group_byte_totals[group] if group in requested_groups else 0 for group in range(2)
            ),
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            source_tp_rank=source.tp_rank,
            source_dp_rank=source.dp_rank,
            requested_groups=requested_groups,
            compact_source=source.compact_layout,
            compact_destination=compact_destination,
            latent_source=source.latent_layout if hybrid else None,
            latent_destination_pages=latent_pages if hybrid else (),
            content_diagnostics=source.content_diagnostics,
        )
    destinations = plan["segments"]
    merged: list[SplitTransferSegment] = []
    for group_id in requested_groups:
        sources = [s for s in source.segments if s.group_id == group_id]
        dests = [d for d in destinations if wire_int(d["group_id"], "destination.group_id") == group_id]
        source_pos = dest_pos = source_offset = dest_offset = 0
        while source_pos < len(sources) and dest_pos < len(dests):
            src, dst = sources[source_pos], dests[dest_pos]
            dst_length = wire_int(dst["length"], "destination.length")
            length = min(src.length - source_offset, dst_length - dest_offset)
            if length <= 0:
                raise ValueError("Invalid live split destination extent")
            merged.append(
                SplitTransferSegment(
                    group_id=group_id,
                    source_buffer_index=src.source_buffer_index,
                    source_offset=src.source_offset + source_offset,
                    destination_address=wire_int(
                        dst["destination_address"],
                        "destination.address",
                    )
                    + dest_offset,
                    length=length,
                    destination_kind=str(dst["destination_kind"]),
                    source_buffer_base=src.source_buffer_base,
                )
            )
            source_offset += length
            dest_offset += length
            if source_offset == src.length:
                source_pos += 1
                source_offset = 0
            if dest_offset == dst_length:
                dest_pos += 1
                dest_offset = 0
        if source_pos != len(sources) or dest_pos != len(dests):
            raise ValueError(
                "source/destination extents do not cover the same bytes: "
                f"group={group_id}, source_extents={len(sources)}, "
                f"destination_extents={len(dests)}"
            )
    return SplitTransferPlan(
        segments=tuple(merged),
        group_byte_totals=tuple(
            source.group_byte_totals[group] if group in requested_groups else 0 for group in range(2)
        ),
        tp_rank=tp_rank,
        dp_rank=dp_rank,
        source_tp_rank=source.tp_rank,
        source_dp_rank=source.dp_rank,
        requested_groups=requested_groups,
        content_diagnostics=source.content_diagnostics,
    )


def canonicalize_source_descriptor(
    descriptor: Any,
    local_source_metadata: Mapping[tuple[int, int], SourceHandshake],
    supported_groups: tuple[int, ...],
) -> dict[str, Any]:
    if not isinstance(descriptor, dict):
        raise ValueError("Live split source descriptor must be a mapping")
    raw_descriptors = descriptor.get("descriptors", (descriptor,))
    normalized = []
    for raw in raw_descriptors:
        identity = (
            wire_int(raw["tp_rank"], "source.tp_rank"),
            wire_int(raw["dp_rank"], "source.dp_rank"),
        )
        metadata = local_source_metadata.get(identity)
        if metadata is None:
            raise ValueError("Live split source rank has no local handshake")
        bases = metadata.kv_caches_base_addr
        sizes = metadata.kv_caches_buffer_sizes
        groups = metadata.buffer_group_ids
        if not (len(bases) == len(sizes) == len(groups)):
            raise ValueError("Live split source handshake is incomplete")
        indices: dict[int, list[int]] = defaultdict(list)
        for index, base in enumerate(bases):
            indices[base].append(index)
        compact = raw.get("compact_layout")
        latent = raw.get("latent_layout")
        if compact is not None or latent is not None:
            if raw.get("segments") or raw.get("format") not in (
                "layer_slot_runs_v1",
                "hybrid_compact_v1",
            ):
                raise ValueError("Unknown or mixed compact live split format")
        if compact is not None:
            compact = dict(compact)
            if wire_int(compact["group_id"], "compact.group_id") not in supported_groups:
                raise ValueError("Compact live split group is unsupported")
            layers = []
            for layer in compact["layers"]:
                layer = dict(layer)
                base = wire_int(layer["buffer_base"], "compact.buffer_base")
                token_bytes = wire_int(layer["token_bytes"], "compact.token_bytes")
                capacity = wire_int(layer["slot_capacity"], "compact.slot_capacity")
                index = next(
                    (
                        candidate
                        for candidate in indices.get(base, ())
                        if groups[candidate] == 1 and capacity * token_bytes <= sizes[candidate]
                    ),
                    None,
                )
                if index is None:
                    raise ValueError("Compact live split layer is not registered")
                layer["buffer_base"] = base
                layer["buffer_index"] = index
                layers.append(layer)
            expected_layers = sum(group == 1 for group in groups)
            expected_indices = tuple(index for index, group in enumerate(groups) if group == 1)
            if (
                len(layers) != expected_layers
                or len({layer["buffer_index"] for layer in layers}) != len(layers)
                or tuple(layer["buffer_index"] for layer in layers) != expected_indices
            ):
                raise ValueError("Compact live split layers are incomplete")
            compact["layers"] = layers
            parsed = parse_compact_layout(compact)
            if parsed is None:
                raise ValueError("Compact live split layout is missing")
        if latent is not None:
            if 0 not in supported_groups:
                raise ValueError("Latent live split group is unsupported")
            latent = dict(latent)
            layers = []
            for layer in latent["layers"]:
                layer = dict(layer)
                base = wire_int(layer["buffer_base"], "latent.buffer_base")
                token_bytes = wire_int(layer["token_bytes"], "latent.token_bytes")
                capacity = wire_int(layer["slot_capacity"], "latent.slot_capacity")
                index = next(
                    (
                        candidate
                        for candidate in indices.get(base, ())
                        if groups[candidate] == 0 and capacity * token_bytes <= sizes[candidate]
                    ),
                    None,
                )
                if index is None:
                    raise ValueError("Latent live split layer is not registered")
                layer["buffer_base"] = base
                layer["buffer_index"] = index
                layers.append(layer)
            expected_layers = sum(group == 0 for group in groups)
            expected_indices = tuple(index for index, group in enumerate(groups) if group == 0)
            if (
                len(layers) != expected_layers
                or len({layer["buffer_index"] for layer in layers}) != len(layers)
                or tuple(layer["buffer_index"] for layer in layers) != expected_indices
            ):
                raise ValueError("Latent live split layers are incomplete")
            latent["layers"] = layers
            if parse_latent_layout(latent) is None:
                raise ValueError("Latent live split layout is missing")
        if compact is not None or latent is not None:
            parsed_compact = parse_compact_layout(compact) if compact is not None else None
            parsed_latent = parse_latent_layout(latent) if latent is not None else None
            expected_totals = (
                parsed_latent.token_count * sum(layer.token_bytes for layer in parsed_latent.layers)
                if parsed_latent is not None
                else 0,
                parsed_compact.token_count * sum(layer.token_bytes for layer in parsed_compact.layers)
                if parsed_compact is not None
                else 0,
            )
            base_totals = tuple(wire_int(value, "source.group_byte_total") for value in raw["group_byte_totals"])
            if len(base_totals) != 2:
                raise ValueError("Compact live split totals require two groups")
            latent_total_raw = raw.get("latent_group_byte_total")
            compact_only_valid = parsed_latent is None and latent_total_raw is None and base_totals == expected_totals
            extension_valid = (
                parsed_latent is not None
                and base_totals == (0, expected_totals[1])
                and latent_total_raw is not None
                and wire_int(latent_total_raw, "source.latent_group_byte_total") == expected_totals[0]
            )
            legacy_hybrid_valid = (
                parsed_latent is not None
                and raw.get("format") == "hybrid_compact_v1"
                and latent_total_raw is None
                and base_totals == expected_totals
            )
            if not (compact_only_valid or extension_valid or legacy_hybrid_valid):
                raise ValueError("Compact live split byte total differs")
            content_diagnostics = sanitize_content_diagnostics(
                raw.get("content_diagnostics"),
                token_count=(parsed_compact.token_count if parsed_compact is not None else 0),
                layer_ids=(
                    tuple(layer.layer_id for layer in parsed_compact.layers) if parsed_compact is not None else ()
                ),
            )
            normalized_descriptor = {
                **raw,
                # Preserve the established group-1 carrier on the wire.
                # New decoders promote latent_group_byte_total only after
                # validating the extension; old decoders ignore it.
                "group_byte_totals": list(base_totals),
                "compact_layout": compact,
                "latent_layout": latent,
            }
            if content_diagnostics is None:
                normalized_descriptor.pop("content_diagnostics", None)
            else:
                normalized_descriptor["content_diagnostics"] = content_diagnostics
            normalized.append(normalized_descriptor)
            continue
        segments = []
        for segment in raw["segments"]:
            segment = dict(segment)
            group_id = wire_int(segment["group_id"], "source.group_id")
            if group_id not in supported_groups:
                continue
            base = wire_int(segment["source_buffer_base"], "source.buffer_base")
            offset = wire_int(segment["source_offset"], "source.offset")
            length = wire_int(segment["length"], "source.length")
            if base <= 0 or offset < 0 or length <= 0:
                raise ValueError("Invalid live split source extent")
            base_indices = indices.get(base)
            if not base_indices:
                raise ValueError("Live split source base is not registered")
            group_matched = False
            index = None
            for candidate in base_indices:
                if group_id != groups[candidate]:
                    continue
                group_matched = True
                if offset + length <= sizes[candidate]:
                    index = candidate
                    break
            if not group_matched:
                raise ValueError("Live split source group does not match buffer")
            if index is None:
                raise ValueError("Live split source extent exceeds buffer")
            segment["group_id"] = group_id
            segment["source_buffer_base"] = base
            segment["source_offset"] = offset
            segment["length"] = length
            segment["source_buffer_index"] = index
            segments.append(segment)
        if not segments:
            raise ValueError("Live split source has no supported groups")
        totals = [0, 0]
        for segment in segments:
            totals[segment["group_id"]] += segment["length"]
        normalized.append({**raw, "segments": segments, "group_byte_totals": totals})
    return {"descriptors": normalized}
