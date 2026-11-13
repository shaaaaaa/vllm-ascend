"""Independent fixtures/reference and loader for the resident-kernel experiment.

No vLLM, LMCache, model weights, or serving operator registration is imported.
Tensor positions match launch.h/binding.cpp; all persistent buffers stay owned
by Case throughout asynchronous work and graph replay.
"""

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAMES = (
    "topk",
    "boundary",
    "row_requests",
    "packed",
    "mapping",
    "shard_counts",
    "request_states",
    "request_generations",
    "state_tokens",
    "state_slots",
    "state_counts",
    "state_generations",
    "prior_slots",
    "shard_miss_tokens",
    "shard_miss_positions",
    "evictable_slots",
    "miss_tokens",
    "miss_counts",
    "target_slots",
    "block_table",
)
STAGES = {"full": 0, "union": 1, "finalize": 2, "update": 3, "union_sort": 4, "union_dedup": 5,
          "state_update": 6, "remap": 7}
VARIANTS = {"baseline": 0, "optimized": 1, "compact_remap": 2, "sharded_finalize": 3, "combined": 4,
            "vector_union": 5, "vector_intersection": 6, "vector_state_update": 7, "exact_combined": 8,
            "query_membership": 9}
MEMBERSHIP_VARIANTS = ("exact_combined", "query_membership")
SOURCE_FILES = (
    ROOT / "csrc/kernels/resident_sorted_cache.cpp",
    ROOT / "csrc/kernels/resident_sorted_cache_coordinated.cpp",
    *(HERE / name for name in ("generate_sources.py", "dispatch.cpp", "binding.cpp", "launch.h", "CMakeLists.txt")),
)


def source_digest() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_FILES:
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def load_library(build_dir: Path) -> dict:
    import torch_npu  # noqa: F401 -- register PrivateUse1 before loading operators

    stamp = json.loads((build_dir / "build-info.json").read_text())
    if stamp["source_sha256"] != source_digest():
        raise RuntimeError("Resident experiment sources changed; rerun build.sh")
    # AscendC emits its shared library in build/lib on some CANN versions.
    # Load it by absolute path before the binding, even if RPATH was stripped.
    candidates = [build_dir / subdir / "libresident_experiment_kernels.so" for subdir in ("lib", "")]
    kernel_library = next((path for path in candidates if path.is_file()), None)
    if kernel_library is None:
        raise FileNotFoundError("Resident kernel library is missing from build/lib or build; rerun build.sh")
    torch.ops.load_library(str(kernel_library.resolve()))
    torch.ops.load_library(str((build_dir / "libresident_experiment_ops.so").resolve()))
    return stamp


@dataclass
class Case:
    tensors: list[torch.Tensor]
    dummy_base: int
    block_size: int

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.tensors[NAMES.index(name)]

    @property
    def requests(self) -> int:
        return self["packed"].shape[0]

    @property
    def mtp(self) -> int:
        return self["topk"].shape[0] // self.requests

    @property
    def shards(self) -> int:
        return self["packed"].shape[1]

    @property
    def capacity(self) -> int:
        return self["packed"].shape[2]

    def clone(self, device=None) -> "Case":
        return Case(
            [t.to(device=device or t.device, copy=True) for t in self.tensors], self.dummy_base, self.block_size
        )

    def reset_from(self, snapshot: "Case") -> None:
        for target, source in zip(self.tensors, snapshot.tensors, strict=True):
            target.copy_(source)

    def run(self, optimized: bool | str, stage: str = "full") -> None:
        variant = VARIANTS[optimized] if isinstance(optimized, str) else int(optimized)
        torch.ops.resident_experiment.run_(self.tensors, self.dummy_base, self.block_size, variant, STAGES[stage])


def _state(case: Case, row: int) -> dict[int, int]:
    result = {}
    for shard in range(case.shards):
        count = int(case["state_counts"][row, shard, 0])
        tokens = case["state_tokens"][row, shard, :count].tolist()
        slots = case["state_slots"][row, shard, :count].tolist()
        assert len(tokens) == len(set(tokens)) and tokens == sorted(tokens)
        assert all(token % case.shards == shard for token in tokens)
        result.update(zip(tokens, slots, strict=True))
    assert len(result.values()) == len(set(result.values()))
    assert set(result.values()) == set(range(len(result)))
    return result


def _put_state(case: Case, row: int, values: dict[int, int]) -> None:
    for shard in range(case.shards):
        pairs = sorted((token, slot) for token, slot in values.items() if token % case.shards == shard)
        count = len(pairs)
        case["state_counts"][row, shard, 0] = count
        if count:
            case["state_tokens"][row, shard, :count] = torch.tensor([p[0] for p in pairs])
            case["state_slots"][row, shard, :count] = torch.tensor([p[1] for p in pairs])


def make_case(requests=2, mtp=2, shards_per_row=4, hit_rate=1.0, scenario="normal", seed=17,
              block_size=128, overlap=1024, live_tokens=0) -> Case:
    """Build valid CPU state, including adversarial padding/generation cases."""
    if requests < 1 or mtp not in (1, 2) or shards_per_row not in (1, 2, 4):
        raise ValueError("invalid requests/query width/shards")
    if not 0 <= hit_rate <= 1 or block_size < 1:
        raise ValueError("invalid hit rate/block size")
    if not 0 <= overlap <= 2048:
        raise ValueError("overlap must be between 0 and 2048")
    if not 0 <= live_tokens <= 2048:
        raise ValueError("live tokens per row must be between 0 and 2048")
    if live_tokens and scenario in ("zero_boundary", "subset"):
        raise ValueError("live_tokens cannot be combined with a scenario that overrides the boundary")
    shards, capacity, dummy = mtp * shards_per_row, mtp * 2048, requests + 2
    shape = (requests, shards, capacity)
    state_shape = (dummy + requests, shards, capacity)
    i32, i16, i64 = torch.int32, torch.int16, torch.int64

    def filled(dims, dtype=i32, value=-7):
        return torch.full(dims, value, dtype=dtype)

    tensors = [
        filled((requests * mtp, 1, 2048)),
        filled((requests * mtp,), value=2_000_000),
        torch.arange(requests, dtype=i32).repeat_interleave(mtp),
        filled(shape),
        filled(shape, i16),
        filled((requests, shards, 16), value=91),
        torch.arange(requests - 1, -1, -1, dtype=i32),
        filled((requests,), i64, 7),
        filled(state_shape),
        filled(state_shape, i16),
        filled((dummy + requests, shards, 16), value=0),
        filled((dummy + requests, 8), i64, -1),
        filled(shape, i16),
        filled(shape),
        filled(shape, i16),
        filled(shape, i16),
        filled((requests, capacity)),
        filled((requests, 16), value=capacity),
        filled((requests, capacity), i64),
        torch.stack(
            [
                torch.randperm(math.ceil(capacity / block_size), generator=torch.Generator().manual_seed(seed + r)).to(
                    i32
                )
                + r * 1000
                + 3
                for r in range(requests)
            ]
        ),
    ]
    case = Case(tensors, dummy, block_size)
    rng = random.Random(seed)
    history = 2048 - live_tokens
    for r in range(requests):
        base = 10000 * (r + 1)
        boundary = base + 2 * 2048 * shards
        current = set()
        for q in range(mtp):
            shift = q * (history - min(overlap, history))
            tokens = list(range(base + shift, base + shift + history))
            if scenario == "skewed":
                tokens = [base + (shift + i) * shards for i in range(history)]
            current.update(tokens)
            if live_tokens:
                case["boundary"][r * mtp + q] = boundary
                tokens.extend(range(boundary, boundary + live_tokens))
            rng.shuffle(tokens)
            case["topk"][r * mtp + q, 0] = torch.tensor(tokens, dtype=i32)
        current = sorted(current)
        hit_count = int(len(current) * hit_rate)
        old = set(rng.sample(current, hit_count))
        if scenario == "one_shard_miss":
            old = set(current)
            old.difference_update([t for t in current if t % shards == 0][:17])
        # Fill the rest of the resident capacity. Slots are dense and unique;
        # token order and physical block order deliberately differ.
        extra = base + 100000
        while len(old) < capacity:
            if scenario != "one_shard_miss" or extra % shards == shards - 1:
                old.add(extra)
            extra += 1
        state_row = int(case["request_states"][r])
        old_list = sorted(old)
        slots = list(range(len(old_list)))
        rng.shuffle(slots)
        _put_state(case, state_row, dict(zip(old_list, slots, strict=True)))
        case["state_generations"][state_row, 0] = 7
    if scenario == "cold":
        case["state_counts"][:, :, 0].zero_()
    elif scenario == "generation":
        case["request_generations"].fill_(8)
    elif scenario == "zero_boundary":
        case["boundary"].zero_()
    elif scenario == "subset":
        case["boundary"].fill_(10000 + 733)
    elif scenario == "padding":
        case["row_requests"][-1] = -1
    elif scenario == "inactive":
        case["row_requests"].fill_(-1)
        case["request_states"].fill_(-1)
    elif scenario == "invalid_indices":
        case["topk"][:, 0, :3] = torch.tensor([-1, -2, 2_100_000], dtype=i32)
    elif scenario not in ("normal", "one_shard_miss", "skewed"):
        raise ValueError(f"unknown scenario: {scenario}")
    return case


def reference(case: Case, *, free_first: bool = False) -> tuple[Case, dict]:
    """Set/dictionary oracle; independent of the AscendC merge implementation."""
    if case["topk"].device.type != "cpu":
        raise ValueError("reference expects CPU tensors")
    result = case.clone()
    stats = {"misses": 0, "selected": 0, "all_hit_requests": 0, "unchanged_shards": 0}
    for r in range(case.requests):
        state = int(case["request_states"][r])
        real = 0 <= state < case.dummy_base
        row = state if real else case.dummy_base + r
        generation = int(case["request_generations"][r])
        old = _state(case, row) if real and int(case["state_generations"][row, 0]) == generation else {}
        selected = set()
        for q in range(case.mtp):
            i = r * case.mtp + q
            if int(case["row_requests"][i]) == r:
                selected.update(t for t in case["topk"][i, 0].tolist() if 0 <= t < int(case["boundary"][i]))
            else:
                result["topk"][i].zero_()
        order = lambda token: (token % case.shards, token)
        misses = sorted(selected - old.keys(), key=order)
        fresh = min(len(misses), case.capacity - len(old)) if free_first else 0
        evicted = sorted(old.keys() - selected, key=order)[: len(misses) - fresh]
        if free_first:
            targets = list(range(len(old), len(old) + fresh)) + [old[t] for t in evicted]
        else:
            targets = [old[t] for t in evicted] + list(range(len(old), len(old) + len(misses) - len(evicted)))
        evicted_set = set(evicted)
        updated = {t: slot for t, slot in old.items() if t not in evicted_set}
        updated.update(zip(misses, targets, strict=True))
        assert len(updated) <= case.capacity
        for shard in range(case.shards):
            current = sorted(t for t in selected if t % case.shards == shard)
            shard_misses = [t for t in misses if t % case.shards == shard]
            shard_evicted = [t for t in evicted if t % case.shards == shard]
            counts = [
                len(current),
                len(shard_misses),
                sum(t % case.shards == shard and t not in selected for t in old),
                sum(t % case.shards == shard for t in old),
                len(shard_evicted),
            ]
            result["shard_counts"][r, shard, :5] = torch.tensor(counts)
            stats["unchanged_shards"] += not shard_misses and not shard_evicted
            if current:
                result["packed"][r, shard, : len(current)] = torch.tensor(current)
                result["prior_slots"][r, shard, : len(current)] = torch.tensor([updated[t] for t in current])
        result["miss_counts"][r, 0] = len(misses)
        if misses:
            result["miss_tokens"][r, : len(misses)] = torch.tensor(misses)
            result["target_slots"][r, : len(misses)] = torch.tensor(
                [
                    int(case["block_table"][r, slot // case.block_size]) * case.block_size + slot % case.block_size
                    for slot in targets
                ],
                dtype=torch.int64,
            )
        for q in range(case.mtp):
            i = r * case.mtp + q
            if int(case["row_requests"][i]) == r:
                result["topk"][i, 0] = torch.tensor(
                    [updated[t] if 0 <= t < int(case["boundary"][i]) else t for t in case["topk"][i, 0].tolist()],
                    dtype=torch.int32,
                )
        _put_state(result, row, updated)
        result["state_generations"][row, 0] = generation
        stats["misses"] += len(misses)
        stats["selected"] += len(selected)
        stats["all_hit_requests"] += not misses
    return result, stats


def membership_reference(case: Case, position_reference: Case) -> Case:
    """Build the bitmap and row output from the oracle's position-wise remap."""
    result = position_reference.clone()
    width = case["topk"].shape[-1]
    for request in range(case.requests):
        histories: list[set[int]] = []
        valid_rows = 0
        boundaries = [0, 0]
        for query in range(case.mtp):
            row = request * case.mtp + query
            history: set[int] = set()
            histories.append(history)
            if int(case["row_requests"][row]) != request:
                continue
            original = case["topk"][row, 0].tolist()
            valid = original.index(-1) if -1 in original else width
            prefix = original[:valid]
            if any(t < 0 for t in prefix) or any(t != -1 for t in original[valid:]):
                raise ValueError("query_membership requires a nonnegative prefix followed by -1 padding")
            if len(set(prefix)) != valid:
                raise ValueError("query_membership requires unique top-k tokens per row")
            boundary = int(case["boundary"][row])
            valid_rows |= 1 << query
            boundaries[query] = boundary
            remapped = position_reference["topk"][row, 0, :valid].tolist()
            historical_slots = []
            live = []
            for token, slot in zip(prefix, remapped, strict=True):
                if token < boundary:
                    history.add(token)
                    historical_slots.append((token % case.shards, token, slot))
                else:
                    live.append(token)
            assembled = [slot for _, _, slot in sorted(historical_slots)] + live
            result["topk"][row, 0] = torch.tensor(assembled + [-1] * (width - valid), dtype=torch.int32)
        result["shard_counts"][request, 0, 5:8] = torch.tensor([valid_rows, *boundaries], dtype=torch.int32)
        for shard in range(case.shards):
            count = int(result["shard_counts"][request, shard, 0])
            tokens = result["packed"][request, shard, :count].tolist()
            bits = [sum(1 << query for query, history in enumerate(histories) if token in history)
                    for token in tokens]
            result["mapping"][request, shard, :count] = torch.tensor(bits, dtype=torch.int16)
    return result


def assert_result(actual: Case, expected: Case, *, membership: bool = False) -> None:
    actual = actual.clone("cpu")
    assert torch.equal(actual["topk"], expected["topk"]), "remapped top-k differs"
    assert torch.equal(actual["state_counts"], expected["state_counts"]), "state counts/padding differ"
    assert torch.equal(actual["state_generations"], expected["state_generations"]), "generation differs"
    assert torch.equal(actual["shard_counts"][:, :, :5], expected["shard_counts"][:, :, :5]), "shard accounting differs"
    if membership:
        assert torch.equal(
            actual["shard_counts"][:, 0, 5:8], expected["shard_counts"][:, 0, 5:8]
        ), "membership row metadata differs"
    for row in range(actual["state_tokens"].shape[0]):
        assert _state(actual, row) == _state(expected, row), f"resident row {row} differs"
    for r in range(actual.requests):
        count = int(expected["miss_counts"][r, 0])
        assert int(actual["miss_counts"][r, 0]) == count, "miss count differs"
        for name in ("miss_tokens", "target_slots"):
            assert torch.equal(actual[name][r, :count], expected[name][r, :count]), name
        for shard in range(actual.shards):
            count = int(expected["shard_counts"][r, shard, 0])
            for name in ("packed", "prior_slots"):
                assert torch.equal(actual[name][r, shard, :count], expected[name][r, shard, :count]), name
            if membership:
                assert torch.equal(
                    actual["mapping"][r, shard, :count], expected["mapping"][r, shard, :count]
                ), "query membership differs"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("soc")
    parser.add_argument("--profile", choices=("full", "membership", "dcp"), default="full")
    args = parser.parse_args()
    (args.build_dir / "build-info.json").write_text(
        json.dumps(
            {
                "source_sha256": source_digest(),
                "soc": args.soc,
                "profile": args.profile,
                "torch": torch.__version__,
            },
            indent=2,
        )
    )
