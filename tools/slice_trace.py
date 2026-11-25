#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stream a time window from a large Chrome/Ascend trace using standard Python.

Times are seconds or MM:SS:mmm, relative to the earliest timestamp in the trace.
The default window is 00:27:500 through 00:28:000. The output starts at zero.

  python slice_trace.py trace_view.json
  python slice_trace.py trace_view.json --start 00:27:500 --end 00:28:000 -o part.json

Use --origin-us when the viewer uses an explicitly chosen timestamp as its zero.
Use --keep-timestamps to retain the input's absolute microsecond time scale.
The default performs two streaming passes, first to find the earliest timestamp.
"""

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path


def iter_events(path, chunk_size=1024 * 1024):
    """Read the event array with the same incremental decoder as the profile tools."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8-sig") as stream:
        buffer = ""
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                raise ValueError("Input must contain a traceEvents array or be an event array")
            buffer += chunk
            match = re.search(r'"traceEvents"\s*:\s*\[', buffer)
            if match:
                position = match.end()
                break
            if buffer.lstrip().startswith("["):
                position = buffer.index("[") + 1
                break
            if len(buffer) > 16 * 1024 * 1024:
                raise ValueError("Trace header exceeds 16 MiB before the event array")
        while True:
            while position < len(buffer) and buffer[position] in " \r\n\t,":
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                event, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as error:
                chunk = stream.read(chunk_size)
                if not chunk:
                    raise ValueError("Input contains an incomplete trace event") from error
                buffer = buffer[position:] + chunk
                position = 0
                continue
            position = end
            if isinstance(event, dict):
                yield event


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def parse_time(value):
    """Return seconds from a decimal number or MM:SS:mmm."""
    if ":" not in value:
        seconds = number(value)
    else:
        match = re.fullmatch(r"(\d+):(\d{1,2}):(\d{1,3})", value)
        if not match:
            raise argparse.ArgumentTypeError("Use seconds or MM:SS:mmm, such as 27.5 or 00:27:500")
        minutes, second, millisecond = map(int, match.groups())
        if second >= 60 or millisecond >= 1000:
            raise argparse.ArgumentTypeError("Seconds must be below 60 and milliseconds below 1000")
        seconds = minutes * 60 + second + millisecond / 1000
    if seconds is None or seconds < 0:
        raise argparse.ArgumentTypeError("Time must be finite and nonnegative")
    return seconds


def find_origin(path):
    origin = None
    for event in iter_events(path):
        if event.get("ph") == "M":
            continue
        timestamp = number(event.get("ts"))
        if timestamp is not None:
            origin = timestamp if origin is None else min(origin, timestamp)
    if origin is None:
        raise ValueError("Input contains no timestamped events")
    return origin


def complete_events(path):
    """Normalize synchronous B/E pairs while streaming complete X events unchanged."""
    stacks = {}
    for event in iter_events(path):
        phase = event.get("ph")
        key = (event.get("pid"), event.get("tid"))
        if phase == "B":
            stacks.setdefault(key, []).append(event)
        elif phase == "E":
            stack = stacks.get(key)
            if not stack:
                continue
            begin = stack.pop()
            start, end = number(begin.get("ts")), number(event.get("ts"))
            if start is not None and end is not None and end >= start:
                yield {**begin, "ph": "X", "dur": end - start}
        else:
            yield event


def clip_event(event, start_us, end_us, shift_us):
    if event.get("ph") == "M":
        return event
    timestamp = number(event.get("ts"))
    if timestamp is None:
        return None
    duration = number(event.get("dur")) if event.get("ph") == "X" else None
    if duration is not None and duration > 0:
        end = timestamp + duration
        if timestamp >= end_us or end <= start_us:
            return None
        clipped_start, clipped_end = max(timestamp, start_us), min(end, end_us)
        result = {**event, "ts": clipped_start - shift_us, "dur": clipped_end - clipped_start}
        if clipped_start != timestamp or clipped_end != end:
            args = dict(event.get("args") or {})
            key = "trace_slice_original"
            while key in args:
                key += "_"
            args[key] = {"ts_us": event.get("ts"), "dur_us": event.get("dur")}
            result["args"] = args
        return result
    if start_us <= timestamp < end_us:
        return {**event, "ts": timestamp - shift_us}
    return None


def slice_trace(source, output, start, end, origin=None, keep_timestamps=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Input and output must use different paths")
    if end <= start:
        raise ValueError("End time must be later than start time")
    original_stat = source.stat()
    if origin is None:
        print("[TRACE_SLICE] pass 1/2: finding the earliest timestamp", flush=True)
        origin = find_origin(source)
    start_us, end_us = origin + start * 1_000_000, origin + end * 1_000_000
    shift = 0 if keep_timestamps else start_us
    print(f"[TRACE_SLICE] extracting {start:.3f}..{end:.3f} seconds", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    selected = metadata_count = scanned = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write('{"traceEvents":[')
            first = True
            for event in complete_events(source):
                scanned += 1
                result = clip_event(event, start_us, end_us, shift)
                if result is None:
                    continue
                if not first:
                    stream.write(",")
                json.dump(result, stream, ensure_ascii=False, separators=(",", ":"))
                first = False
                if event.get("ph") == "M":
                    metadata_count += 1
                else:
                    selected += 1
            if not selected:
                raise ValueError("No events overlap this window. Check --start, --end or --origin-us")
            stream.write('],"displayTimeUnit":"ms","otherData":')
            json.dump({"trace_slice": {
                "origin_us": origin,
                "start_offset_seconds": start,
                "end_offset_seconds": end,
                "timestamps_rebased": not keep_timestamps,
                "cross_boundary_spans": "clipped with original ts/dur retained in event args",
            }}, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("}")
        current_stat = source.stat()
        if (original_stat.st_size, original_stat.st_mtime_ns) != (current_stat.st_size, current_stat.st_mtime_ns):
            raise ValueError("Input changed during extraction; retry after the trace has finished writing")
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"[TRACE_SLICE] kept {selected:,} events and {metadata_count:,} metadata records from {scanned:,}")
    print(f"[TRACE_SLICE] output {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="Input trace_view.json")
    parser.add_argument("--start", type=parse_time, default="00:27:500")
    parser.add_argument("--end", type=parse_time, default="00:28:000")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--origin-us", type=float, help="Explicit viewer time zero in microseconds")
    parser.add_argument("--keep-timestamps", action="store_true")
    args = parser.parse_args(argv)
    if args.origin_us is not None and not math.isfinite(args.origin_us):
        parser.error("--origin-us must be finite")
    output = args.output or args.source.with_name(f"{args.source.stem}_{args.start:.3f}-{args.end:.3f}.json")
    try:
        slice_trace(args.source, output, args.start, args.end, args.origin_us, args.keep_timestamps)
    except (OSError, ValueError) as error:
        parser.exit(1, f"[TRACE_SLICE] {error}\n")


if __name__ == "__main__":
    main()
