#!/usr/bin/env python3
"""Locate the first failing sorted-resident remap vector stage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


STAGES = (
    (1, "topk-load"),
    (2, "accumulator-duplicate"),
    (3, "first-shard-load"),
    (4, "mapping-int16-to-float"),
    (5, "mapping-float-to-int32"),
    (6, "rank-maxs"),
    (7, "rank-compare"),
    (8, "rank-mins"),
    (9, "rank-byte-offset"),
    (10, "slot-gather-int16"),
    (11, "slot-int16-to-float"),
    (12, "slot-float-to-int32"),
    (13, "first-shard-select"),
    (14, "all-shards-select"),
    (15, "final-maxs"),
    (16, "final-compare"),
    (17, "final-select"),
    (18, "topk-writeback"),
)
TEST_FILE = (
    "tests/e2e/nightly/single_node/ops/singlecard_ops/"
    "test_resident_sorted_cache.py"
)
TEST_NAME = "test_remap_vector_stage_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mtp",
        type=int,
        choices=(1, 2),
        nargs="+",
        default=(1, 2),
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=(1, 2, 3),
        nargs="+",
        default=(1, 2, 3),
        help="Decode step whose remap pipeline should be truncated.",
    )
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--end-stage", type=int, default=18)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["ASCEND_LAUNCH_BLOCKING"] = "1"

    for mtp in args.mtp:
        for step in args.step:
            for stage, name in STAGES:
                if stage < args.start_stage or stage > args.end_stage:
                    continue
                case = f"mtp{mtp}-step{step}-stage{stage}-{name}"
                node = f"{TEST_FILE}::{TEST_NAME}[{case}]"
                print(
                    f"\n===== MTP={mtp} step={step}"
                    f" stage={stage}: {name} =====",
                    flush=True,
                )
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-sv", node],
                    cwd=repo_root,
                    env=environment,
                    check=False,
                )
                if result.returncode != 0:
                    print(
                        "\nFIRST FAILING REMAP STAGE:"
                        f" MTP={mtp}, step={step}, stage={stage},"
                        f" instruction={name}",
                        flush=True,
                    )
                    return result.returncode

    print("\nAll selected remap stages passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
