# Resident kernel experiment

Branch: `perf/resident-kernel-optimization`, based on production `origin/sparse`
at `58cbdbbc15a9fd1afcc31223b9fe018105dd0f1f`.

This is an isolated old/new kernel comparison. It does not enable a new serving
path, change the serving operator registry, or require model weights, vLLM, or
LMCache imports. Do not use this benchmark to claim a serving TPOT/TTFT gain.

## DCP free-slot-first comparison

This separate experiment is based on `feat/decoder-dcp-sharded-h2d` at
`451b4571b`. It leaves production sources and serving dispatch unchanged.
The `dcp` build profile generates four entry points from the production source:
the existing DCP union and membership update, the original finalizer, and a
finalizer that consumes unused slots before evicting unselected residents.
Source substitutions must match exactly; source drift stops generation.
Existing vector limits and scalar fallbacks are preserved.

Both policies use **4,096 scratch slots per request per simulated DCP rank**,
two query rows, Top-K 2,048, block/interleave size 128, and eight internal shards.
The candidate preserves current hits, grows the dense occupied prefix using
unused slots, then uses the existing shard-major eviction order when full.
It adds no planner kernel, workspace, or extra KV capacity. It can make the
existing union/update more expensive by retaining more history.

Build only the standalone extension, with the same SoC and Python/CANN
environment as the serving build (for example `ascend910b3` on the 910B3 host):

```bash
EXP=benchmarks/ops/resident_kernel_experiment
BUILD="$EXP/build-dcp"
bash "$EXP/build.sh" "$SOC_VERSION" "$BUILD" dcp
python3 -m pytest --confcutdir="$EXP/tests" -o addopts= -q \
  "$EXP/tests/test_dcp_free_slots.py" --resident-build-dir "$BUILD"
python3 "$EXP/benchmark_dcp.py" --build-dir "$BUILD" \
  --requests 1 2 4 8 --steps 128 --overlap 0.9 --query-overlap 0.5 \
  --dcp-size 8 --ranks 0 --output "$BUILD/temporal-0.9.json"
```

Use an idle NPU. This runs sequentially on one device, without `torchrun`, HCCL,
model weights, or a server. `--device` selects that NPU; `--ranks 0 1 2 3 4 5 6 7`
simulates all DCP ranks sequentially. The rank ownership predicate is
`(global_token // 128) % dcp_size`. Uniform traces are identical across ranks
and both policies. DCP sizes 2, 4 and 8 are supported.

`--overlap` accepts any value in `[0, 1]` and controls the intersection of each
query's consecutive Top-K selections. The default `0.9` retains
`floor(2048 * 0.9) = 1843` tokens per query per step. It is **not** the cache hit
rate. `--query-overlap` independently controls overlap between target/MTP rows
within each step. The JSON reports realized global and rank-local overlap,
trace hashes, miss counts, occupancy and maximum internal-shard occupancy at
every step. Input token order is shuffled. Non-null block-table pages are
deliberately reordered.

The default `reuse` workload draws from a fixed `--working-set 32768` within
`--history 65536`, allowing previously dropped selections to return. Identical
input traces go to both policies. A `--workload fresh` control never reintroduces
a dropped token; it can reveal increased metadata cost without miss savings.
Increase `--history` when a long fresh trace would exhaust unique tokens.
Invalid or impossible inputs fail explicitly.

For skew and boundary stress, use `--distribution owner-skew` to put all
selections on the simulated rank, or `--distribution shard-skew` to put them in
internal shard zero. Both need a sufficiently large eligible working set, e.g.
`--history 262144 --working-set 32768`. Native regressions additionally cover
2,047/2,048/2,049 resident entries in one shard, full 4,096-entry state/union,
mixed fresh allocation plus eviction, generation reset, and padded rows.

Correctness validation captures an instrumented multi-step graph, checks each
step's remapped queries/miss destinations/counts against the CPU oracle, and
checks the complete final resident map. Timing uses separate graphs without
those diagnostic copies. Each policy starts cold; state persists across all
`--steps` in a trace. State resets happen outside timing, once per replay of the
whole trace. `--repeats` defaults to 10, `--warmup` to 2; timing order alternates.
Reported event time is microseconds per **batch planner step**, including the
input D2D copy needed to restore the in-place Top-K input.

For actual Union/Finalize/Update device-task times, add
`--profile-dir "$BUILD/profiles-0.9"`. The parser requires exactly one device
task per stage per step and refuses to substitute host spans. Profiling and
correctness work are excluded from event timing. Use new output/trace paths
for another run; existing results are not overwritten.

This measures the **three-kernel planner**, not H2D, attention or serving TPOT.
It reserves the same BF16 latent-plus-RoPE scratch allocation for both policies,
but performs no KV transfer. Estimated H2D bytes use misses × 576 × 2 for one
latent layer; these are not measured bandwidth or transfer savings. The trace
covers historical scratch selections; live-tail attention integration is
outside this standalone benchmark. No serving configuration change or package
installation is needed.

The workload and oracle can also run on a CPU-only development host:

```bash
python3 "$EXP/benchmark_dcp.py" --cpu-only --requests 1 --steps 128 \
  --overlap 0.9 --output cpu-reference.json
```

CPU results contain no kernel timings and do not qualify native correctness or
performance. Run the native checks and benchmark on the target NPU before
considering integration.

## What changes

Select variants with `--variants` and a reference with `--compare-to`. The full
build defaults to the original baseline comparison.

| Variant | Finalize | Update/remap |
|---|---|---|
| `baseline` | Original | Original |
| `optimized` | Earlier all-hit shortcut | Earlier unchanged-state shortcut |
| `compact_remap` | Original | Compact slot gather; original state merge |
| `sharded_finalize` | Existing experimental sharded worker | Original |
| `combined` | Sharded worker | Compact slot gather; original state merge |
| `vector_union` | Original | Original; vectorized union deduplication/mapping |
| `vector_intersection` | Original | Original sorting/dedup; exact vector resident intersection only |
| `vector_state_update` | Original | Exact vector state merge; original union/finalize/remap |
| `exact_combined` | Original | Vector intersection and vector state merge; original finalize/remap |
| `query_membership` | Original | Exact intersection/state merge; vector union tags and row assembly |

## Fast query membership comparison

On the 910B3 host, activate the deployment's Torch/Torch-NPU environment and
source its CANN `set_env.sh`. Use the same SoC value as the serving build.

```bash
EXP=benchmarks/ops/resident_kernel_experiment
BUILD="$EXP/build-membership"
bash "$EXP/build.sh" ascend910b3 "$BUILD" membership
python3 "$EXP/benchmark.py" --build-dir "$BUILD" --requests 1 8 16 \
  --mtp 2 --hit-rates 0.9 1 --live-tokens 64 --json "$BUILD/results.json"
```

The `membership` profile builds five AIV entry points (two unions, two updates,
and one shared finalize) and the small Torch-NPU binding. It does not build or
install vLLM-Ascend, LMCache, or the model's attention kernels. Rerun the same
build command for an incremental rebuild. The optional last argument `full`
builds the older experimental variants as well.

The fast profile defaults to `exact_combined` vs. `query_membership`. It checks
both against the CPU oracle before timing. The default `--stage full --mode
profile` reports individual Union/Finalize/Update device times, their sum, and
the span across the three launches. Use `--stage update` for an isolated Update
comparison; each variant's workspace is prepared by its own Union and Finalize.
Try `--overlap 0`, `--overlap 2048`, and `--scenario skewed` to test disjoint
rows, high overlap, and one heavily loaded shard. The hit fraction in the JSON
is computed from the historical union; live entries are not cache requests.

```bash
python3 -m pytest --confcutdir="$EXP/tests" -o addopts= -q \
  "$EXP/tests/test_query_membership.py" "$EXP/tests/test_kernels.py" \
  --resident-build-dir "$BUILD" -k membership
```

The new path sorts the same shard-local candidates, carrying a one-hot query tag
instead of the original column. Adjacent comparisons compact the union and the
OR of each pair's tags. The active prefix of the existing int16 mapping
allocation now holds one tag per unique token. Finalize does not read that
allocation. Union also publishes row validity and split boundaries in unused
fields of shard zero's count record. Update retains one state task per shard
and adds one task per query row in the same launch. Row tasks read finalized slots, compact aligned UB
segments, and write a single fixed-width row. They do not read the state that
sibling tasks are updating. No scan over original Top-K positions or extra
kernel launch is needed.

The experiment accepts the existing unique-row contract for one or two query
rows: valid indices occupy a contiguous nonnegative prefix, followed by `-1`;
invalid graph rows are zeroed separately. It returns shard-ordered historical
slots followed by the original live indices. A stage-only caller must pair
matching workspace producers and consumers; the new tags are not position
ranks. Legacy remap/state probes are unavailable for the membership variant.

These timings cover the planner, not H2D retrieval, sparse attention, or TPOT.
Before serving integration, compare actual attention outputs and timings for
the reordered indices, and check model output and MTP acceptance on NPU.

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

On A2, integer `Compare` supports EQ only. The experimental searches implement
`a < b` as `NOT(min(a,b) == b)` and integer NE by inverting equality masks; no
floating-point conversion of token IDs is needed. Compare widths are padded to
64 four-byte elements (256 bytes), including small shards. See the
[CANN Compare contract](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0066.html).

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

## Exact state-update experiment

`vector_state_update` preserves the exact eviction prefix and sorted token/slot
state. It compacts old tokens absent from current, discards precisely the prefix
selected by finalize, then uses parallel merge-path searches to merge the retained
old entries with current tokens. Finalized current hit slots equal their old slots;
misses keep the slots assigned by the unchanged finalizer. No token IDs are cast
to float, and A2 comparisons use integer Min/EQ/Not with 64-lane padding.

The guarded path requires `oldCount + currentCount <= 2048`. Larger/skewed shards
use the original scalar merge. The 4096-capacity/single-shard geometry also stays
scalar. Vector-capable geometries reserve 48 KiB additional UB (no extra GM scratch
or payload banks); calculated explicit UB allocation is at most 164 KiB. The
merged state buffers remain disjoint from remap scratch, and the vector-to-MTE3
dependency is completed before original publication/remap code runs.

`exact_combined` adds the previously measured vector intersection. Both variants
remain experiment-only compile flags, disabled in the serving build. They add no
retrieval calls or payload copies. Correctness and speed still require NPU testing.

```bash
EXP=benchmarks/ops/resident_kernel_experiment
RED="$EXP/redesign"
BUILD="$EXP/build-910b3-state-update"
bash "$EXP/build.sh" ascend910b3 "$BUILD" &&
python -m pytest --confcutdir="$EXP/tests" -o addopts= "$EXP/tests" \
  --resident-build-dir "$BUILD" -xq

python "$EXP/benchmark.py" --build-dir "$BUILD" --requests 8 --mtp 2 \
  --shards-per-row 4 --hit-rates 0.9 \
  --variants baseline vector_intersection vector_state_update exact_combined \
  --stage full --iterations 30 --warmup 20 --json "$EXP/state-full-results.json"

for STAGE in state_update remap; do
  python "$EXP/benchmark.py" --build-dir "$BUILD" --requests 8 --mtp 2 \
    --shards-per-row 4 --hit-rates 0.9 --variants baseline vector_state_update \
    --stage "$STAGE" --iterations 30 --warmup 20 --json "$EXP/$STAGE-results.json" || break
done

python "$RED/matched.py" --original-build-dir "$BUILD" \
  --methods original vector_intersection vector_state_update exact_combined \
  --lmcache-ascend-dir /workspace/sqh/LMCache-Ascend \
  --trace "$RED/sweeps/matched-r8-shift/r8-inputs.pt" \
  --output-dir "$RED/sweeps/state-update-r8" --repeats 4
```

State-only and remap-only probes start from the same baseline union/finalize output.
They are diagnostic components, not exact additive phase timers: separate kernels
have different initialization and expose different writeback overlap. Judge the
full three-kernel and matched retrieval path. The matched runner verifies exact
state, remapped indices, miss/target arrays and source counts before timing every
exact variant. Use `--requests 1 8 16` without `--trace` for a synthetic size sweep;
for serving inputs use the documented pre-remap tensor trace schema.

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

## Serving selection

The serving extension builds the baseline, exact, and query membership planners.
`VLLM_ASCEND_DSA_RESIDENT_QUERY_MEMBERSHIP=0` (default) keeps the existing
`VLLM_ASCEND_DSA_RESIDENT_EXACT_KERNELS` selection: `1` for exact, `0` for baseline.
Set `VLLM_ASCEND_DSA_RESIDENT_QUERY_MEMBERSHIP=1` on the decoder workers to use
the bitmap planner on the existing fused sorted-resident decode route. The
bitmap variant uses exact resident matching and state update regardless of
the legacy selector. Split remap and diagnostic operators still use position
ranks. Nonresident execution retains its existing path.

The worker selects the matching Union and Finalize/Update operators before
capture. Shared planner consumers still reuse the producer's plan. There is no
extra tensor, kernel launch, environment read on replay, or change to LMCache
miss transfers. Output indices for a valid row may be reordered.

Rebuild the **serving** extension once (the standalone experiment build is insufficient):

```bash
SOC_VERSION=ascend910b3 COMPILE_CUSTOM_KERNELS=1 python -m pip install -e . --no-build-isolation --no-deps
export VLLM_ASCEND_DSA_RESIDENT_QUERY_MEMBERSHIP=1
# Launch the existing decoder deployment with the flag on all workers.
```

Use your deployment's actual `SOC_VERSION`. Restart the decoder workers and
recapture their graphs after changing the flag. Missing bitmap operators fail
at initialization when enabled. No LMCache YAML or graph-size change is needed.

For a device check of the rebuilt extension:

```bash
python -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_resident_sorted_cache.py \
  -k 'query_membership or full_path_supports_graph_replay or zeroes_staged_graph_padding'
```

For a serving A/B, keep the exact selector at `1` and toggle only membership.
The standalone planner speedup does not measure attention, TPOT, or throughput.
