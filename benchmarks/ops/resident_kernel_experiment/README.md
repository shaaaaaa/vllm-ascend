# Resident kernel experiment

Branch: `perf/resident-kernel-optimization`, based on production `origin/sparse`
at `58cbdbbc15a9fd1afcc31223b9fe018105dd0f1f`.

This is an isolated old/new kernel comparison. It does not enable a new serving
path, change the serving operator registry, or require model weights, vLLM, or
LMCache imports. Do not use this benchmark to claim a serving TPOT/TTFT gain.

## What changes

The union/intersection algorithm is identical in all variants. Select variants
with `--variants`; `baseline` is always included for comparison.

| Variant | Finalize | Update/remap |
|---|---|---|
| `baseline` | Original | Original |
| `optimized` | Earlier all-hit shortcut | Earlier unchanged-state shortcut |
| `compact_remap` | Original | Compact slot gather; original state merge |
| `sharded_finalize` | Existing experimental sharded worker | Original |
| `combined` | Sharded worker | Compact slot gather; original state merge |
| `vector_union` | Original | Original; vectorized union deduplication/mapping |
| `vector_intersection` | Original | Original sorting/dedup; exact vector resident intersection only |

`vector_union` changes only the union kernel. Sorting, the subsequent scalar
resident intersection, allocation policy, finalize and update remain baseline.
After sorting, vector adjacent comparisons plus `GatherMask` compact unique
tokens. MTP1 uses its existing unique-row contract. Inverse mapping uses a
bounded vector lower-bound search over the unique tokens, avoiding the original
per-occurrence scalar `GetValue`/`SetValue` loop. It preserves each row's own
boundary and invalid/padding behavior. It deliberately does not use local
`Scatter`, which is not supported on the A2 training product path.

The vector implementation reuses dead sort/input scratch and expands only two
bit masks to request width (512 additional UB bytes for query width 2). There
is no new persistent or global-memory workspace. Binary search adds vector
passes and rereads input rows; it may be slower, especially for small shards.
The experiment must be timed; no speedup is assumed.

The two new experiments do not enable the earlier unchanged-state shortcut.
They keep the same three dependency stages; no cross-core spin waits or new
host synchronization are introduced in kernel dispatch.

**Compact remap:** build one packed prior-slot array in existing dead merge
scratch, select a packed offset per original top-k position, then use one slot
Gather. This retains the per-shard mapping reads/casts but removes repeated
slot Gather/conversion/selection. The packed array requires at most
`capacity + 15 * shards` int16 elements; the reused buffer holds `2 * capacity`.
No additional UB or persistent device workspace is allocated. Buffers still
used for state writeback are not reused. Empty selections preserve union's
padding-zero behavior and unselected input positions.

**Sharded finalize:** reuse `resident_sorted_cache_coordinated.cpp`'s existing
worker without changing its allocation semantics or payload/cacheline ownership.
Each request/shard computes prefixes from the existing union metadata and owns
its outputs. This exposes more workers than the original one-per-request
finalizer, but duplicates some reads; benefit is not guaranteed.

The earlier `optimized` experiment remains available unchanged:

- Finalize does not copy hit-only shards' prior slots back and forth. With no
  misses anywhere in a request, it also avoids the block-table read and payload
  processing, while clearing **every shard's stale selected-eviction count** and
  publishing a zero miss count.
- Update skips the old-state load/merge/write when that shard has **both zero
  misses and zero selected evictions**. A zero-miss shard can supply eviction
  slots to other shards, so checking only its miss count is incorrect.
- Top-k remapping, generation publication, dummy-row handling, payload fences,
  and the three-kernel order remain intact. An invalidated empty state already
  has zero counts published by union; stale payload tails remain invalid.

The production translation unit defaults the compile-time specialization to
zero. Only generated experiment sources enable them. There is no new runtime environment knob
or serving dispatch check. The baseline/optimized symbols have separate suffixes
to avoid interposition with an installed serving extension.

Both variants reuse the same source rather than maintaining a second large
kernel copy. At CMake configuration, `generate_sources.py` emits thirteen translation
units, each with one explicitly named AIV entry point: the original six plus
compact update, sharded finalize, vector union, and four union-prefix probes.
Helper/class and entry-point bodies come from the
resident source; entry names are literal, not preprocessor aliases. The host
dispatcher preserves the same launch order. This packaging avoids the former
multi-entry-point/include-and-macro build, which failed binary registration on
CANN 8.5.1 with `finalize ... get kernel type failed`. The six-entry version was
successfully run on the user's 910B3. The two new variants still require native validation.

The standalone build compiles only these resident entry points and a small PyTorch-NPU binding.
It does **not** rebuild `vllm_ascend_C` or any other model/attention/MoE kernels.

## Build on the NPU host

Activate the same Python/torch-npu environment used by the deployment, and source
the installed CANN `set_env.sh`. CMake, a C++ compiler, and the CANN AscendC
development toolchain must be installed. Run from the repository root:

```bash
EXP=benchmarks/ops/resident_kernel_experiment
bash "$EXP/build.sh" "$SOC_VERSION"
```

`SOC_VERSION` must be the **exact value used for your existing vLLM-Ascend build**,
not a guessed marketing model name. For example, if that build uses
`SOC_VERSION=Ascend910B3`, pass `Ascend910B3`. The script requires it explicitly
and does not select a device family on your behalf. This experiment targets the
same hardware supported by the existing resident kernels, not 310P.

Output: `build/libresident_experiment_ops.so`, its resident-kernel library (under
`build/lib` on CANN 8.5.1), generated sources under `build/generated`, and
`build/build-info.json`. The build directory is local to this experiment. No
package installation or root access is required. A different directory can be
passed as the second build-script argument.

The loader rejects a stale build when the source digest changes. Use the same
torch/torch-npu/CANN environment for building and running the resulting library.
The wrapper uses `OpCommand`, as production does, so launches do not bypass
torch-npu's task queue and overtake pending tensor copies.

The loader explicitly loads the kernel library by absolute path before loading
the binding. The binding also has an explicit `$ORIGIN/lib` runtime search path.
After updating from the original packaging, use a new build directory and pass
the actual device target, for example on the reported 910B3 host:

```bash
BUILD="$EXP/build-910b3-vector-union"
bash "$EXP/build.sh" ascend910b3 "$BUILD"
python -m pytest --confcutdir="$EXP/tests" -o addopts= "$EXP/tests/test_kernels.py" --resident-build-dir "$BUILD" -k 'normal and 1-1' -xq
```

If registration fails, do not benchmark: successful Python loading alone does
not prove that the AscendC-generated registration stub accepted its binary.

## Correctness tests

```bash
python -m pytest --confcutdir="$EXP/tests" -o addopts= "$EXP/tests" --resident-build-dir "$BUILD" -q
```

For a custom build directory/device, append
`--resident-build-dir /path/to/build --resident-device 0`.

The native tests compare all compiled variants with an independent CPU
set/dictionary oracle, not merely against each other. They cover query widths
1/2, 1/2/4 shards per query row, cold/mixed/all hits, full capacity, subsets,
cross-shard eviction, stale generations, inactive/dummy requests, padding,
negative/out-of-boundary indices, skewed shards, grid-stride execution, and
fragmented physical block tables, and disjoint/fully overlapping MTP rows.
Use `-k compact_remap` or `-k sharded_finalize` to select an individual variant's
parameterized cases. Graph tests change inputs/generations/padding
at fixed addresses and queue multiple replays before a final host fence. Launch
shape/dtype/alias errors are rejected before a kernel is submitted.

Only valid prefixes of variable-length payloads are compared; unused workspace
tails have no defined value. State counts/generation padding, remapped top-k,
valid resident slot bijections, exact miss order, and physical destinations are
checked. The fixture poisons previous miss/eviction counters to detect stale
metadata in the new fast path.

Host-only execution requires CPU Torch and pytest; native tests skip if
torch-npu/NPU is unavailable. **Host passes do not validate compiled kernels.**
Host validation also covers the packed-remap formulation/UB bound, variant
source generation, profiler parsing, and replay-test snapshot ownership.
Native compilation and execution cannot be performed on the Windows development host.

## Benchmark

First run a small profiler comparison:

```bash
python "$EXP/benchmark.py" --build-dir "$BUILD" --requests 8 --mtp 2 --shards-per-row 4 --hit-rates 0.9 --variants baseline vector_union --iterations 30 --warmup 20 --json "$EXP/vector_union.json"
```

`--mtp` means **query width**: `2` represents one speculative token plus the
verification row; it is not the number of speculative tokens alone.

The default `--mode profile --stage full` captures a graph, runs it under
`torch_npu.profiler`, and extracts actual device kernel tasks from the exported
trace. It reports the sum of the three kernel durations, each stage's duration,
and the device span from union start through update completion. Python event
intervals are not reported as device kernel time. Incomplete/ambiguous task
counts or ordering cause an error instead of silently using host timings.

All mutable state and raw top-k are restored on **every iteration**, including
warmup. Reset-copy tasks and host submission gaps are excluded from the kernel
duration sum. Otherwise a miss-heavy fixture would quietly converge to all-hit
residency. Profiling runs each variant separately; repeat runs/reverse the
variant list to check clock/thermal drift. Profiler overhead is not zero, and
this is a kernel experiment rather than a serving latency estimate.

Results report device-task mean/p50/p95 in microseconds, speedup and exact miss
counts. JSON records build identity, runtime versions, parameters and trace
paths. Traces go to a unique directory under `profiles/` (override using
`--trace-dir`). If the parser rejects your CANN trace format, retain that trace;
do not interpret a missing task as a zero-duration kernel.

`--stage all` additionally measures stages in isolation; baseline union/finalize
prepare identical valid operands. The full captured chain is the primary
comparison. Legacy `--mode graph` / `--mode eager` retain event intervals for
diagnosis only: their approximately 120 us submission floor can hide gains.

`--overlap 0` selects 4096 unique tokens for query width 2; `--overlap 1024`
(default) selects 3072; `--overlap 2048` selects 2048. Test all three overlap
levels rather than treating 4096 input entries as 4096 unique selections.

## Exact intersection with unchanged retrieval

`vector_intersection` replaces only the scalar resident intersection in union.
It preserves the ordered unique misses, hit slots and eviction candidates. The
baseline finalize/update kernels and registered-source retrieval are reused.
Both counts must fit 2048 lanes; larger/skewed shards execute the original scalar
merge. It reuses existing UB scratch and adds no global buffers, payload copies,
kernel launches, runtime serving knobs or serving-path checks (the compile flag
defaults to zero). Native performance remains to be measured on the target NPU.

Build and test the standalone kernels:

```bash
EXP=benchmarks/ops/resident_kernel_experiment
BUILD="$EXP/build-910b3-intersection"
bash "$EXP/build.sh" ascend910b3 "$BUILD" &&
python -m pytest --confcutdir="$EXP/tests" -o addopts= "$EXP/tests" \
  --resident-build-dir "$BUILD" -xq
```

Profile the original/new kernels, including their unchanged finalize and update:

```bash
python "$EXP/benchmark.py" --build-dir "$BUILD" --requests 8 --mtp 2 \
  --shards-per-row 4 --hit-rates 0.9 --variants baseline vector_intersection \
  --stage all --iterations 30 --warmup 20 --json "$EXP/intersection-results.json"
```

Measure the same evolving workload with actual registered CPU chunks and paged
BF16 512+64 retrieval. Both methods use the Original adapter: same miss-only loads,
same transfer call count, same hit-in-place behavior. The runner aborts on different
valid metadata, miss/target arrays or source counts before profiling. Only the
standalone original-kernel library is needed; no redesign native rebuild is needed.

```bash
RED="$EXP/redesign"
python "$RED/matched.py" --original-build-dir "$BUILD" \
  --methods original vector_intersection --lmcache-ascend-dir /workspace/sqh/LMCache-Ascend \
  --output-dir "$RED/sweeps/intersection-r8" --requests 8 --steps 4 --repeats 4 \
  --scenario rank_shift
```

This last command uses eight profiler sessions. To reuse an earlier input sequence,
add `--trace "$RED/sweeps/matched-r8-shift/r8-inputs.pt"`; that saved synthetic trace
is not a production trace. Real captured **pre-remap** top-k sequences can use the
same tensor schema in [redesign/MATCHED_COMPARISON.md](redesign/MATCHED_COMPARISON.md).
Repeat with requests 1/8/16 and other selection patterns before promoting the variant.
The report separates union, complete three-kernel planning, and total preparation.
The prepared-per-request backend is shared by both; this is not full-model TPOT.

## Isolate the union prefix cost

The additional `union_sort` and `union_dedup` stages compile the same kernel
with an early return after sorting or after deduplication/mapping. They are
diagnostics only: they do not publish a usable resident load plan and cannot
be followed by finalize/update. Baseline and vector variants have distinct
symbols. No timers or phase checks are inserted into the serving build.

```bash
python "$EXP/benchmark.py" --build-dir "$BUILD" --requests 8 --mtp 2 --hit-rates 0.9 --variants baseline vector_union --stage union_sort --json "$EXP/union_sort.json"
python "$EXP/benchmark.py" --build-dir "$BUILD" --requests 8 --mtp 2 --hit-rates 0.9 --variants baseline vector_union --stage union_dedup --json "$EXP/union_dedup.json"
python "$EXP/benchmark.py" --build-dir "$BUILD" --requests 8 --mtp 2 --hit-rates 0.9 --variants baseline vector_union --stage union --json "$EXP/union_full.json"
```

These are cumulative prefix probes, not exact in-kernel phase timestamps. Each
probe writes results to keep its computation observable; code generation and
writeback differ at the cutoff. Use the differences to identify the dominant
region, then judge benefit using the full union and full three-kernel chain.
If through-sort already consumes most of the union latency, vectorizing only
deduplication/mapping cannot deliver a large end-to-end gain.

Additional comparisons:

```bash
# Unchanged-state opportunity despite misses confined to one value shard.
python "$EXP/benchmark.py" --build-dir "$BUILD" --requests 8 --mtp 2 --scenario one_shard_miss --hit-rates 1
# Empty resident state: deliberately exercise the fallback work, not a warm hit.
python "$EXP/benchmark.py" --build-dir "$BUILD" --requests 8 --mtp 2 --scenario cold --hit-rates 0 --overlap 0
# Shard-count sensitivity at the same request count/query width.
python "$EXP/benchmark.py" --build-dir "$BUILD" --requests 8 --mtp 2 --shards-per-row 1 2 4 --hit-rates 0.9 1
```

Use `--json FILE` to retain separate runs, `--stage full` for just the full chain,
and `--device N` to choose a device. Run on an otherwise idle NPU and repeat.
Input restoration also creates a controlled, warm-memory microbenchmark rather
than reproducing all cache contention of a serving workload.

The new variants target partial hits but can lose performance from extra index
arithmetic or duplicated reads. No speedup is claimed until these measurements
pass correctness and run on the target hardware.
