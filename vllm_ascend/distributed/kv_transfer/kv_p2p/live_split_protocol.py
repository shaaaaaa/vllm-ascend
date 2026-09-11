# SPDX-License-Identifier: Apache-2.0
"""Live-split wire records and syntactic validation.

Parsing borrows caller payloads and returns records; it owns no transfer state,
registered memory, executors or sockets. Registered-address validation remains
at the receiver/handshake boundary. Optional invalid diagnostics keep their
existing warning-and-ignore behavior.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from vllm.logger import logger

MAX_WIRE_INTEGER = (1 << 63) - 1


CONTENT_DIAGNOSTIC_SCHEMA = 1


CONTENT_DIAGNOSTIC_ALGORITHM = "blake2b-128"


MAX_CONTENT_DIAGNOSTIC_LAYERS = 3


MAX_CONTENT_DIAGNOSTIC_TOKENS = 24


def wire_int(value: Any, field: str) -> int:
    """Parse an integer field without silently truncating wire values."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer")
    parsed = int(value)
    if not -MAX_WIRE_INTEGER <= parsed <= MAX_WIRE_INTEGER:
        raise ValueError(f"{field} is outside the supported integer range")
    return parsed


def sanitize_content_diagnostics(raw: Any, *, token_count: int, layer_ids: tuple[int, ...]) -> dict[str, Any] | None:
    """Validate optional debug metadata without making P2P depend on it."""
    if raw is None:
        return None
    try:
        if not isinstance(raw, dict):
            raise ValueError("content diagnostics must be a mapping")
        if wire_int(raw.get("schema"), "diagnostic.schema") != 1:
            raise ValueError("unknown content diagnostic schema")
        if raw.get("algorithm") != CONTENT_DIAGNOSTIC_ALGORITHM:
            raise ValueError("unknown content diagnostic algorithm")
        layers = [wire_int(value, "diagnostic.layer_id") for value in raw.get("sample_layer_ids", ())]
        valid_layer_ids = set(layer_ids)
        tokens = [wire_int(value, "diagnostic.logical_token") for value in raw.get("sample_logical_tokens", ())]
        if (
            not layers
            or len(layers) > MAX_CONTENT_DIAGNOSTIC_LAYERS
            or layers != sorted(set(layers))
            or any(value not in valid_layer_ids for value in layers)
        ):
            raise ValueError("invalid diagnostic layer samples")
        if (
            not tokens
            or len(tokens) > MAX_CONTENT_DIAGNOSTIC_TOKENS
            or tokens != sorted(set(tokens))
            or any(not 0 <= value < token_count for value in tokens)
        ):
            raise ValueError("invalid diagnostic token samples")

        def hashes(name: str, limit: int) -> dict[str, str]:
            values = raw.get(name, {})
            if not isinstance(values, dict) or len(values) > limit:
                raise ValueError(f"invalid diagnostic {name}")
            result = {str(key): str(value) for key, value in values.items()}
            if any(
                len(value) != 32 or any(character not in "0123456789abcdef" for character in value)
                for value in result.values()
            ):
                raise ValueError(f"invalid diagnostic {name} hash")
            return result

        row_hashes = hashes("row_hashes", len(layers) * len(tokens))
        layer_hashes = hashes("layer_hashes", len(layers))
        expected_row_keys = {f"{layer_id}:{logical_token}" for layer_id in layers for logical_token in tokens}
        if set(row_hashes) != expected_row_keys:
            raise ValueError("diagnostic row hashes do not match samples")
        if set(layer_hashes) != {str(layer_id) for layer_id in layers}:
            raise ValueError("diagnostic layer hashes do not match samples")
        combined_hash = str(raw.get("combined_hash", ""))
        if len(combined_hash) != 32 or any(character not in "0123456789abcdef" for character in combined_hash):
            raise ValueError("invalid diagnostic combined hash")
        return {
            "schema": CONTENT_DIAGNOSTIC_SCHEMA,
            "algorithm": CONTENT_DIAGNOSTIC_ALGORITHM,
            "sample_layer_ids": layers,
            "sample_logical_tokens": tokens,
            "row_hashes": row_hashes,
            "layer_hashes": layer_hashes,
            "combined_hash": combined_hash,
            "readback_may_synchronize": bool(raw.get("readback_may_synchronize", True)),
        }
    except (KeyError, TypeError, ValueError) as error:
        logger.warning("Ignoring invalid optional NPU content diagnostics: %s", error)
        return None


@dataclass(frozen=True)
class SplitTransferSegment:
    """One registered local destination and its remote KV source extent."""

    group_id: int
    source_buffer_index: int
    source_offset: int
    destination_address: int
    length: int
    destination_kind: str
    source_buffer_base: int | None = None


@dataclass(frozen=True)
class SplitSourceSegment:
    """Opaque prefiller-owned extent in a registered source buffer."""

    group_id: int
    source_buffer_index: int
    source_buffer_base: int
    source_offset: int
    length: int


@dataclass(frozen=True)
class SplitCompactLayer:
    layer_id: int
    buffer_base: int
    token_bytes: int
    slot_capacity: int
    buffer_index: int = -1


@dataclass(frozen=True)
class SplitCompactRun:
    logical_token_start: int
    physical_slot_start: int
    token_count: int


@dataclass(frozen=True)
class SplitCompactLayout:
    group_id: int
    token_count: int
    layers: tuple[SplitCompactLayer, ...]
    runs: tuple[SplitCompactRun, ...]


@dataclass(frozen=True)
class SplitLatentRun:
    logical_token_start: int
    physical_slot_start: int
    token_count: int


@dataclass(frozen=True)
class SplitLatentPage:
    logical_token_start: int
    token_count: int
    runs: tuple[SplitLatentRun, ...]


@dataclass(frozen=True)
class SplitLatentLayout:
    group_id: int
    token_count: int
    layers: tuple[SplitCompactLayer, ...]
    pages: tuple[SplitLatentPage, ...]


@dataclass(frozen=True)
class SplitLatentDestinationPage:
    logical_token_start: int
    destination_address: int
    length: int


@dataclass(frozen=True)
class SplitSourceDescriptor:
    segments: tuple[SplitSourceSegment, ...]
    group_byte_totals: tuple[int, int]
    tp_rank: int
    dp_rank: int
    compact_layout: SplitCompactLayout | None = None
    latent_layout: SplitLatentLayout | None = None
    content_diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class SplitTransferPlan:
    segments: tuple[SplitTransferSegment, ...]
    group_byte_totals: tuple[int, int]
    tp_rank: int
    dp_rank: int
    requested_groups: tuple[int, ...] = (0, 1)
    compact_source: SplitCompactLayout | None = None
    compact_destination: SplitCompactLayout | None = None
    latent_source: SplitLatentLayout | None = None
    latent_destination_pages: tuple[SplitLatentDestinationPage, ...] = ()
    source_tp_rank: int | None = None
    source_dp_rank: int | None = None
    content_diagnostics: dict[str, Any] | None = None

    @property
    def source_rank(self) -> tuple[int, int]:
        return (
            self.tp_rank if self.source_tp_rank is None else self.source_tp_rank,
            self.dp_rank if self.source_dp_rank is None else self.source_dp_rank,
        )

    @property
    def destination_rank(self) -> tuple[int, int]:
        return self.tp_rank, self.dp_rank


def parse_compact_layout(layout: Any) -> SplitCompactLayout | None:
    if layout is None:
        return None
    if not isinstance(layout, dict):
        raise ValueError("Compact split layout must be a mapping")
    parsed = SplitCompactLayout(
        group_id=wire_int(layout["group_id"], "compact.group_id"),
        token_count=wire_int(layout["token_count"], "compact.token_count"),
        layers=tuple(
            SplitCompactLayer(
                layer_id=wire_int(layer["layer_id"], "compact.layer_id"),
                buffer_base=wire_int(layer["buffer_base"], "compact.buffer_base"),
                token_bytes=wire_int(layer["token_bytes"], "compact.token_bytes"),
                slot_capacity=wire_int(layer["slot_capacity"], "compact.slot_capacity"),
                buffer_index=wire_int(
                    layer.get("buffer_index", -1),
                    "compact.buffer_index",
                ),
            )
            for layer in layout["layers"]
        ),
        runs=tuple(
            SplitCompactRun(
                logical_token_start=wire_int(
                    run["logical_token_start"],
                    "compact.logical_token_start",
                ),
                physical_slot_start=wire_int(
                    run["physical_slot_start"],
                    "compact.physical_slot_start",
                ),
                token_count=wire_int(run["token_count"], "compact.run_token_count"),
            )
            for run in layout["runs"]
        ),
    )
    if parsed.group_id != 1 or parsed.token_count <= 0 or len(parsed.layers) > 4096 or len(parsed.runs) > 1_000_000:
        raise ValueError("Compact split layout only supports group 1")
    if not parsed.layers or not parsed.runs:
        raise ValueError("Compact split layout is empty")
    layer_ids = [layer.layer_id for layer in parsed.layers]
    if layer_ids != list(range(len(parsed.layers))):
        raise ValueError("Compact split layers must be complete and ordered")
    if any(layer.buffer_base <= 0 or layer.token_bytes <= 0 or layer.slot_capacity <= 0 for layer in parsed.layers):
        raise ValueError("Invalid compact split layer")
    logical = 0
    for run in parsed.runs:
        if (
            run.logical_token_start != logical
            or run.physical_slot_start < 0
            or run.token_count <= 0
            or run.physical_slot_start + run.token_count > min(layer.slot_capacity for layer in parsed.layers)
        ):
            raise ValueError("Invalid compact split run coverage")
        logical += run.token_count
    if logical != parsed.token_count:
        raise ValueError("Compact split runs do not cover all tokens")
    physical = sorted((run.physical_slot_start, run.physical_slot_start + run.token_count) for run in parsed.runs)
    if any(left[1] > right[0] for left, right in zip(physical, physical[1:])):
        raise ValueError("Compact split physical slots overlap")
    return parsed


def parse_latent_layout(layout: Any) -> SplitLatentLayout | None:
    if layout is None:
        return None
    if not isinstance(layout, dict):
        raise ValueError("Latent split layout must be a mapping")
    pages = tuple(
        SplitLatentPage(
            logical_token_start=wire_int(
                page["logical_token_start"],
                "latent.logical_token_start",
            ),
            token_count=wire_int(page["token_count"], "latent.page_token_count"),
            runs=tuple(
                SplitLatentRun(
                    logical_token_start=wire_int(
                        run["logical_token_start"],
                        "latent.run_logical_token_start",
                    ),
                    physical_slot_start=wire_int(
                        run["physical_slot_start"],
                        "latent.physical_slot_start",
                    ),
                    token_count=wire_int(
                        run["token_count"],
                        "latent.run_token_count",
                    ),
                )
                for run in page["runs"]
            ),
        )
        for page in layout["pages"]
    )
    parsed = SplitLatentLayout(
        group_id=wire_int(layout["group_id"], "latent.group_id"),
        token_count=wire_int(layout["token_count"], "latent.token_count"),
        layers=tuple(
            SplitCompactLayer(
                layer_id=wire_int(layer["layer_id"], "latent.layer_id"),
                buffer_base=wire_int(layer["buffer_base"], "latent.buffer_base"),
                token_bytes=wire_int(layer["token_bytes"], "latent.token_bytes"),
                slot_capacity=wire_int(layer["slot_capacity"], "latent.slot_capacity"),
                buffer_index=wire_int(
                    layer.get("buffer_index", -1),
                    "latent.buffer_index",
                ),
            )
            for layer in layout["layers"]
        ),
        pages=pages,
    )
    if (
        parsed.group_id != 0
        or parsed.token_count <= 0
        or not parsed.layers
        or len(parsed.layers) > 4096
        or not parsed.pages
        or sum(len(page.runs) for page in parsed.pages) > 1_000_000
    ):
        raise ValueError("Invalid latent split layout")
    layer_ids = [layer.layer_id for layer in parsed.layers]
    if layer_ids != list(range(len(parsed.layers))):
        raise ValueError("Latent split layers must be complete and ordered")
    if any(layer.buffer_base <= 0 or layer.token_bytes <= 0 or layer.slot_capacity <= 0 for layer in parsed.layers):
        raise ValueError("Invalid latent split layer")
    capacity = min(layer.slot_capacity for layer in parsed.layers)
    logical = 0
    for page in parsed.pages:
        if page.logical_token_start != logical or page.token_count <= 0 or not page.runs:
            raise ValueError("Invalid latent split page coverage")
        page_logical = logical
        for run in page.runs:
            if (
                run.logical_token_start != page_logical
                or run.physical_slot_start < 0
                or run.token_count <= 0
                or run.physical_slot_start + run.token_count > capacity
            ):
                raise ValueError("Invalid latent split run coverage")
            page_logical += run.token_count
        if page_logical != logical + page.token_count:
            raise ValueError("Latent split runs do not cover their page")
        logical += page.token_count
    if logical != parsed.token_count:
        raise ValueError("Latent split pages do not cover all tokens")
    physical = sorted(
        (
            run.physical_slot_start,
            run.physical_slot_start + run.token_count,
        )
        for page in parsed.pages
        for run in page.runs
    )
    if any(left[1] > right[0] for left, right in zip(physical, physical[1:])):
        raise ValueError("Latent split physical slots overlap")
    return parsed


def _source_group_byte_totals(
    raw: dict[str, Any],
    latent_layout: SplitLatentLayout | None,
) -> tuple[int, ...]:
    """Promote an optional latent extension after base-schema parsing."""
    base_totals = tuple(wire_int(value, "source.group_byte_total") for value in raw["group_byte_totals"])
    if latent_layout is None or "latent_group_byte_total" not in raw:
        return base_totals
    if len(base_totals) != 2 or int(base_totals[0]) != 0:
        raise ValueError("Latent extension requires group-1 base totals")
    return (
        wire_int(
            raw["latent_group_byte_total"],
            "source.latent_group_byte_total",
        ),
        base_totals[1],
    )


def parse_source_descriptor(
    descriptor: Any,
) -> tuple[SplitSourceDescriptor, ...] | None:
    if descriptor is None:
        return None
    if isinstance(descriptor, SplitSourceDescriptor):
        descriptors = (descriptor,)
    elif not isinstance(descriptor, dict):
        raise ValueError("Live split source descriptor must be a mapping")
    else:
        raw_descriptors = descriptor.get("descriptors", (descriptor,))
        parsed_descriptors = []
        for raw in raw_descriptors:
            latent_layout = parse_latent_layout(raw.get("latent_layout"))
            compact_layout = parse_compact_layout(raw.get("compact_layout"))
            diagnostic_token_count = compact_layout.token_count if compact_layout is not None else 0
            diagnostic_layer_ids = (
                tuple(layer.layer_id for layer in compact_layout.layers) if compact_layout is not None else ()
            )
            parsed_descriptors.append(
                SplitSourceDescriptor(
                    segments=tuple(
                        SplitSourceSegment(
                            group_id=wire_int(segment["group_id"], "source.group_id"),
                            source_buffer_index=wire_int(
                                segment["source_buffer_index"],
                                "source.buffer_index",
                            ),
                            source_buffer_base=wire_int(
                                segment["source_buffer_base"],
                                "source.buffer_base",
                            ),
                            source_offset=wire_int(
                                segment["source_offset"],
                                "source.offset",
                            ),
                            length=wire_int(segment["length"], "source.length"),
                        )
                        for segment in raw.get("segments", ())
                    ),
                    group_byte_totals=(_source_group_byte_totals(raw, latent_layout)),
                    tp_rank=wire_int(raw["tp_rank"], "source.tp_rank"),
                    dp_rank=wire_int(raw["dp_rank"], "source.dp_rank"),
                    compact_layout=compact_layout,
                    latent_layout=latent_layout,
                    content_diagnostics=sanitize_content_diagnostics(
                        raw.get("content_diagnostics"),
                        token_count=diagnostic_token_count,
                        layer_ids=diagnostic_layer_ids,
                    ),
                )
            )
        descriptors = tuple(parsed_descriptors)
    identities: set[tuple[int, int]] = set()
    for parsed in descriptors:
        if len(parsed.group_byte_totals) != 2:
            raise ValueError("Live split source totals require two groups")
        if (parsed.tp_rank, parsed.dp_rank) in identities:
            raise ValueError("Duplicate live split source rank")
        identities.add((parsed.tp_rank, parsed.dp_rank))
        totals = [0, 0]
        for segment in parsed.segments:
            if (
                segment.group_id not in (0, 1)
                or segment.source_buffer_index < 0
                or segment.source_buffer_base <= 0
                or segment.source_offset < 0
                or segment.length <= 0
            ):
                raise ValueError("Invalid live split source extent")
            totals[segment.group_id] += segment.length
        if parsed.compact_layout is not None:
            if any(layer.buffer_index < 0 for layer in parsed.compact_layout.layers):
                raise ValueError("Compact split source buffer is unresolved")
            totals[1] += parsed.compact_layout.token_count * sum(
                layer.token_bytes for layer in parsed.compact_layout.layers
            )
        if parsed.latent_layout is not None:
            if any(layer.buffer_index < 0 for layer in parsed.latent_layout.layers):
                raise ValueError("Latent split source buffer is unresolved")
            latent_total = parsed.latent_layout.token_count * sum(
                layer.token_bytes for layer in parsed.latent_layout.layers
            )
            totals[0] += latent_total
        if tuple(totals) != parsed.group_byte_totals:
            raise ValueError("Live split source byte totals mismatch")
    return descriptors


def parse_split_plan(plan: Any) -> SplitTransferPlan | None:
    if plan is None:
        return plan
    if isinstance(plan, SplitTransferPlan):
        parsed = plan
    elif not isinstance(plan, dict):
        raise ValueError("Live split destination plan must be a mapping")
    else:
        segments = tuple(
            SplitTransferSegment(
                group_id=wire_int(segment["group_id"], "plan.group_id"),
                source_buffer_index=wire_int(
                    segment["source_buffer_index"],
                    "plan.source_buffer_index",
                ),
                source_offset=wire_int(segment["source_offset"], "plan.source_offset"),
                destination_address=wire_int(
                    segment["destination_address"],
                    "plan.destination_address",
                ),
                length=wire_int(segment["length"], "plan.length"),
                destination_kind=str(segment["destination_kind"]),
                source_buffer_base=(
                    None
                    if segment.get("source_buffer_base") is None
                    else wire_int(
                        segment["source_buffer_base"],
                        "plan.source_buffer_base",
                    )
                ),
            )
            for segment in plan.get("segments", ())
        )
        parsed = SplitTransferPlan(
            segments=segments,
            group_byte_totals=tuple(wire_int(value, "plan.group_byte_total") for value in plan["group_byte_totals"]),
            tp_rank=wire_int(plan["tp_rank"], "plan.tp_rank"),
            dp_rank=wire_int(plan["dp_rank"], "plan.dp_rank"),
            requested_groups=tuple(
                wire_int(value, "plan.requested_group") for value in plan.get("requested_groups", (0, 1))
            ),
            compact_source=parse_compact_layout(plan.get("compact_source")),
            compact_destination=(parse_compact_layout(plan.get("compact_layout"))),
            latent_source=parse_latent_layout(plan.get("latent_source")),
            latent_destination_pages=tuple(
                SplitLatentDestinationPage(
                    logical_token_start=wire_int(
                        page["logical_token_start"],
                        "plan.latent_page_logical_start",
                    ),
                    destination_address=wire_int(
                        page["destination_address"],
                        "plan.latent_page_address",
                    ),
                    length=wire_int(page["length"], "plan.latent_page_length"),
                )
                for page in plan.get("latent_destination_pages", ())
            ),
            source_tp_rank=(
                None if plan.get("source_tp_rank") is None else wire_int(plan["source_tp_rank"], "plan.source_tp_rank")
            ),
            source_dp_rank=(
                None if plan.get("source_dp_rank") is None else wire_int(plan["source_dp_rank"], "plan.source_dp_rank")
            ),
            content_diagnostics=sanitize_content_diagnostics(
                plan.get("content_diagnostics"),
                token_count=(
                    wire_int(
                        plan["compact_layout"]["token_count"],
                        "plan.compact.token_count",
                    )
                    if plan.get("compact_layout") is not None
                    else 0
                ),
                layer_ids=(
                    tuple(
                        wire_int(
                            layer["layer_id"],
                            "plan.compact.layer_id",
                        )
                        for layer in plan["compact_layout"].get("layers", ())
                    )
                    if plan.get("compact_layout") is not None
                    else ()
                ),
            ),
        )
    if len(parsed.group_byte_totals) != 2:
        raise ValueError("Live split byte totals require exactly two groups")
    return parsed
