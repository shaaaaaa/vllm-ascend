# Resident-cache algorithm redesign experiments

Target branch: `shaaaaaa/vllm-ascend:perf/resident-kernel-optimization`.
Base inspected: `a845e93995d1e3ed85b7a521fcb02412e27153e3`.
Target workload: Ascend 910B3, top-k 2048, **two query rows for one speculative token**.

## Status: experimental, not serving-qualified

This directory adds executable candidates and independent correctness tests. It
**does not replace production serving operators, change `sfa_v1.py`, or enable a
new serving path**. Existing experiments and the production fallback are untouched.

The authoring runtime had CPU Torch, but no `torch_npu`, CANN compiler or NPU.
Consequently, **the AscendC sources have not been compiled or run on a 910B3**.
Native build/registration, actual SFA equivalence, production integration,
PCIe-transfer behavior, and the combined **<10 us** target remain unverified.
Do not interpret host test passes as native validation or a performance result.

| Candidate | Executable implementation | Remaining work |
|---|---|---|
| Reload every selected occurrence | Torch reference; native AIV lookup and fused HBM copy | Real offloaded-source comparison |
| Fixed-position tag check | Torch reference; native contiguous AIV lookup and fused HBM copy | Device qualification and serving adapter |
| Bounded positional search across both rows | Torch reference; native AIV lookup and fused HBM copy | Radius/tiling/recall tuning on traces |
| Immutable bounded hash snapshot | Torch reference; native single-writer table builder + AIV lookup/copy | Real device tuning; collisions are safe misses |
| Direct token-to-slot directory | Torch reference; native directory builder + UB-window lookup/copy | Measure full-directory traffic; not assumed faster |
| One-bit membership plus slot directory | Torch reference with explicit packed bitmap | Custom native bitmap implementation |
| Sorted snapshot lookup | Torch reference with exact integer pair keys | Native sort/search or maintained snapshot implementation |
| Signed-bit matrix equality join | Tiled Torch FP16 matmul reference, token+version exact | Fused native Cube/Vector reduction and tuning |
| Full HBM token-addressable cache | Functional correctness baseline (`full_hbm`) | Real allocation policy/capacity and SFA integration |
| Shared-indexer group plan | Functional `shared_plan`; verifies every layer's identity/readiness | Group lifecycle, physical block translation, serving wiring |
| Hit/miss split softmax | Functional `split_attention` and empty-subset tests | Native SFA partial statistics, asynchronous overlap |
| Retrieval fusion | Native `resident_resolve_copy_redesign` with normalized HBM source | **Not yet the actual LMCache host-source kernel** |
| Lightning-indexer epilogue fusion | Not implemented | Requires editing actual final top-k native implementation and its synchronization |

This is deliberately an accurate status table, not a claim that all proposed
optimizations have already become production-ready native kernels.

## Changed contract

There is no requirement to reproduce the old union order, miss order, eviction
policy, or sorted state. A selection is a request-local token position plus a
KV **version**. Request-slot incarnation is a separate 64-bit epoch.

The snapshot is immutable while a plan/transfer/SFA consumer is in flight. It
contains one slot per selected occurrence and tags only become reusable after
the associated KV is ready. The next bank has one private destination per
occurrence, in original query order. Repeating a token across query rows may
cause extra copies; it never adds or removes a contribution within a query.

A plan's `source[B,Q*K]` is:

- `>=0`: verified source slot in the old bank;
- `-1`: offloaded-prefix miss;
- `-2`: invalid or inactive selection, zeroed in the normalized test payload;
- `-3`: live-tail selection, not an LMCache offload miss;
- `-4`: full-HBM reference only, directly sourced by token address.

Every hit checks full token identity, full int32 KV version, readiness, and
request epoch. Hash collisions/overflows cause **false misses, never false hits**.
Metadata with inconsistent boundaries/lengths invalidates its whole query row.
A selected token is valid only below that row's causal length. The split boundary
separates offloaded prefix from live tail. Version/epoch reuse after wraparound
requires explicit invalidation; counters cannot silently alias old identities.

`assert_safe` uses an independent CPU dictionary oracle, rather than treating
another candidate as ground truth. `materialize` compares every selected KV row
with the complete current truth. Attention tests preserve selection order and
multiplicity and compare with a separate softmax attention oracle. This is not
an invocation of the deployed sparse-flash-attention operator.

`Transaction` is CPU-only and tests publication-before-completion and failure
handling. It is **not** a substitute for native events or a serving transaction.
An asynchronous integration must keep all owners and old banks alive until their
last consumer completes; Python cancellation is not a memory-reuse fence.

## Native implementation

`resident_snapshot_build_redesign` owns disjoint, aligned 256-cell table slices.
It vector-filters candidate indices, compacts them locally, then resolves the
slice's collisions with a deterministic last-slot rule. There are no global
atomics, shared write cachelines or cross-core spin waits. The scalar insertion
loop can be expensive under skew; it is included in timing, not hidden as setup.

`resident_lookup_redesign` works on 256 selected positions per tile. The fixed
mode reads only the corresponding old-tag/version/readiness tile. Bounded search
loads the old snapshot into UB and performs bounded exact comparisons. Hash
lookup loads a small table; a large direct directory streams through UB windows.
The directory is a comparison candidate, not a guaranteed low-latency path.

`resident_resolve_copy_redesign` fuses lookup with two-bank materialization. It
copies byte-aligned payloads without dtype conversion, including FP16/BF16 and
raw byte rows. The copy loop is a correctness-first per-occurrence DMA path;
it still needs transfer tiling/double-buffering optimization on hardware.

Native limits: B=1..1024; Q=1 or 2; K a multiple of 256 up to 2048; universe up
to 1,048,576 positions; KV row size a multiple of 32 bytes up to 8192; hash table
a power of two from 256 through 8192. Each native buffer must be contiguous,
on the same NPU, and nonoverlapping. In-place old/new bank aliases are rejected.

The fused fixture source is **complete, normalized, token-major HBM KV**, including
live-tail values. It does not exercise registered CPU pointers, page planes,
physical block tables, quantization metadata layouts, or LMCache request routing.
The host wrapper uses `OpCommand` and the current torch-npu stream, retains tensor
owners through queued submission, and never publishes state. Keep the `NativeCase`
alive until device completion; cross-stream callers must establish dependencies.

## Build and tests

From the repository root, activate the deployment's Torch/torch-npu environment
and source its CANN `set_env.sh`. Pass the actual SOC_VERSION used by that build.

```bash
EXP=benchmarks/ops/resident_kernel_experiment/redesign
python "$EXP/build.py" --soc "$SOC_VERSION" --build-dir "$EXP/build-910b3"
python -m pytest --confcutdir="$EXP/tests" -o addopts= "$EXP/tests" \
  --redesign-build-dir "$EXP/build-910b3" -q
```

Builds are isolated from `vllm_ascend_C` and the existing experiment library.
One explicitly named AIV entry is emitted per translation unit, following the
working branch's registration packaging. A source digest rejects stale binaries.
A build/registration/test failure is fatal, with **no functional fallback**.
Only absent NPU support/hardware causes hardware tests to skip.

Host-only correctness:

```bash
python -m pytest --confcutdir="$EXP/tests" -o addopts= "$EXP/tests/test_candidates.py" "$EXP/tests/test_profile.py" -q
python "$EXP/benchmark.py" --backend reference --topk 64 --universe 257 \
  --overlap 32 --json "$EXP/reference-smoke.json"
```

The test suite includes current/old mismatches, empty/cold/partial hits, full
int32 identity, collisions, invalid/causal bounds, graph padding, epoch changes,
KV-version changes, duplicate selections, both query widths, real top-k 2048,
failed publication, multi-step transitions, shared-group disagreement, and
empty hit/miss softmax subsets. Hardware tests add fixed-address queued graph
replays, dtype-byte-copy checks, and alias rejection.

## Benchmark honestly

```bash
python "$EXP/benchmark.py" --backend native --build-dir "$EXP/build-910b3" \
  --variants reload fixed_position bounded_position hash_snapshot direct_directory \
  --requests 1 --query-rows 2 --topk 2048 --universe 131072 --overlap 1024 \
  --hit-rate 0.9 --scenario rank_shift --iterations 30 --warmup 10 \
  --json "$EXP/native-r1.json"
```

Repeat with request counts 8/16, stable/permuted/rank-shift/cold scenarios,
multiple hit rates, overlap 0/1024/2048, different bounded-search radii, and
multiple hash sizes. Use the actual deployed KV row byte size before comparing
transfer costs. Defaults are fixture geometry, not GLM model facts.

Native benchmarks correctness-gate BOTH lookup and fused outputs. They capture
fixed-address graphs and parse actual hardware kernel tasks, with Decimal
arithmetic for large timestamps. They report table-build-inclusive kernel sum
and dependency span, plus lookup+HBM-materialization timings. Missing tasks,
wrong ordering, ambiguous pairing, and invalid durations are errors, not zeros.
Inputs/snapshot stay immutable, so warmup cannot silently change misses to hits.
Profiler overhead and warm-memory conditions still apply.

The functional backend is a correctness sweep, not a timing claim. Matrix,
bitmap, sorting and directory preprocessing live inside their candidate calls;
none is represented as zero-cost setup. Native functional tests can expose
unsupported Torch-NPU primitives; such failures require real implementation
work rather than a silent CPU fallback.

**Do not claim <10 us/layer from lookup-only time.** The acceptance metric must
include required state publication/maintenance, any work moved into indexer or
retrieval, additional copied bytes, and SFA's changed address-access cost.
Current reports explicitly say `serving_qualified: false`. `native_validated`
means only that the requested normalized fixtures passed, not that serving is
qualified. Compare against the sibling original-three-kernel benchmark as well.

## Required serving integration before enabling anything

1. Map the native source descriptors into separate latent/no-PE/PE/scale planes,
   actual physical block tables, and LMCache's prepared source objects. Prefix
   misses must use ready LMCache sources; live-tail entries must use current
   native KV, never be sent blindly to the prefix retriever.
2. Allocate two-bank scratch with explicit ownership and consumer completion.
   Materializing every selected row into the next bank prevents indefinite old
   arena references. A pointer-only alternative needs a separate retirement proof.
3. For shared-indexer groups, coordinate slot policy and lifecycle across exactly
   the actual producer/consumer layers. Do not confuse these with DSA KV groups.
   Group planning may conservatively reject a slot unless every layer agrees.
4. Extend final indexer output only after global top-k completion; preserve raw
   indices for consumers. No unproved cross-core epilogue barrier is inserted here.
5. Compare actual SFA outputs/logits and speculative acceptance with the baseline
   across layouts, request reuse/preemption, padding, stream overlap and failures.
   The standalone CPU attention oracle is necessary but not sufficient.

Optimization objective: fewer launches, no exact union, no globally packed miss
list, no in-place eviction dependency, and no false hits. Selecting the fastest
candidate requires 910B3 measurements of the complete path. No winner or measured
speedup is claimed by this patch.
# Batched materialization experiment

`--copy-mode row` preserves the original row-at-a-time implementation (default).
`--copy-mode batched` uses the same lookup and source selection, gathers up to
16 rows into a 16 KiB UB buffer, and writes each group contiguously. Groups shrink
for larger rows; the final partial group is bounded. Invalid rows are zeroed;
hits and misses still copy into a disjoint output bank. Completion fences protect
the gather-to-output dependency and buffer reuse. This groups transfers and
fences; it does not yet overlap separate double-buffered groups.

Rebuild and run the native tests before benchmarking. Both implementations are
in the same library and tested against the same oracle, including graph replay.
Use a fresh build directory when adding the new translation unit:

```bash
RED=benchmarks/ops/resident_kernel_experiment/redesign
RBUILD="$RED/build-910b3-batched"
python "$RED/build.py" --soc ascend910b3 --build-dir "$RBUILD" &&
python -m pytest --confcutdir="$RED/tests" -o addopts= "$RED/tests" --redesign-build-dir "$RBUILD" -xq
for COPY in row batched; do
  python "$RED/benchmark.py" --backend native --build-dir "$RBUILD" \
    --variants reload bounded_position hash_snapshot --copy-mode "$COPY" \
    --requests 8 --query-rows 2 --topk 2048 --universe 131072 \
    --overlap 1024 --hit-rate 0.9 --scenario rank_shift \
    --kv-width 576 --dtype bfloat16 --iterations 30 --warmup 10 \
    --json "$RED/materialize-576-$COPY.json" || break
done
```

576 BF16 elements model the latent row's byte width only: this remains a
contiguous token-major HBM fixture, not production paged layout or CPU-cache DMA.
The JSON reports row bytes and total materialized bytes separately from miss
bytes. Repeat with `stable`, `permuted`, and `cold`, and request counts 1/8/16.
Use width 64 and float32 to compare with the earlier reported fixture. Actual
production top-k traces, registered CPU sources, and original metadata-plus-load
chain timing remain necessary before claiming a serving speedup.
