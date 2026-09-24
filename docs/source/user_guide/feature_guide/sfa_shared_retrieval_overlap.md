# Experimental shared-layer retrieval overlap

Branch: `feat/shared-layer-retrieval-overlap`, based on
`integration/decode-full-graph-production` at `1611dc8f`.

This is a **default-off Python scheduling experiment**, not a new KV transfer
kernel. It retrieves the next shared-indexer layer's historical Group-0 misses
while the current layer runs value/output projections and MLP. Newly generated
KV still comes from that layer's ordinary pre-compute. Top-k, resident eviction,
Group-1 loading, Mooncake placement and request admission are unchanged.

## Enable and compare

Use the matching full-graph vLLM/LMCache deployment and existing full-graph
configuration. On every target worker:

```bash
export VLLM_ASCEND_SFA_FULL_GRAPH=1
export VLLM_ASCEND_SFA_SHARED_RESIDENT_PLAN=1
export VLLM_ASCEND_SFA_SHARED_RETRIEVAL_OVERLAP=1
```

Use `VLLM_ASCEND_SFA_SHARED_RETRIEVAL_OVERLAP=0` for the serial baseline. Restart
workers between settings; capture topology cannot change in a running worker.
Keep `VLLM_ASCEND_DSA_RESIDENT_EXACT_KERNELS`, model, TP/DP, capture sizes, GC,
performance logging and request workload identical between runs.

These changes require **no additional native rebuild** over the matching
`1611dc8f` serving extensions. The resident-kernel integration in that base still
requires its own previously documented vLLM-Ascend build. No LMCache YAML changes
or new LMCache-Ascend native symbols are introduced here.

## Schedule and integration

For an eligible edge `L0 -> L1` in an active shared resident group:

```text
main:      retrieve L0 -> SFA L0 -> record start -> V projection/O projection/MLP -> wait done -> pre L1
transfer:                              wait start -> retrieve L1 -> record done
```

The consumer waits before *any* pre-compute/cache access. Its later serial
retrieval is suppressed. The first group member still retrieves normally;
the final member launches no prefetch. Edges never cross group boundaries.

Eligibility is established at startup for each graph key: matching prepared
transfers, active shared planning, FP16/BF16 warmed query and destination layout,
and the existing cube-only value-projection path (including its token limit).
Ineligible edges retain serial retrieval. Native, staged and eager fallback
retain serial retrieval even when the switch is enabled. A model with no
eligible shared edges gets no transfer stream and no overlap benefit.

The owner holds one transfer stream and start/done events per eligible edge/key.
It captures dependencies inside the root graph. Warm replay executes no new
Python per-layer work, host wait, event polling, metadata copy or request lookup.
It does execute extra device event dependencies; zero device overhead is not
claimed, particularly on all-hit steps.

Layer cache destinations are checked for overlapping byte ranges before capture;
disjoint views of the same storage remain legal. The prefetch-capable post
operator is selected from the startup flag BEFORE initial model compilation.
It resolves the current prepared edge and its fixed cache destinations inside
the opaque operator, just as the existing retrieval operator resolves its
cache state by layer. Its output mutation preserves the model's computation
order; consumer pre-compute also explicitly waits for the transfer. Cache
buffers must not be traced as late-created Python tensor attributes: vLLM's
initial profile compiles before KV initialization and cached callables can
bypass guards afterward. Resident plan read/write tensors remain explicit,
with their existing AOT buffer-reuse handling unchanged. Payload arguments are
retained through synchronized graph cleanup. No extra KV bank is allocated.

Every edge joins before its consumer, so root completion still covers all KV
transfers and protects source leases/pointer-table replacement. Uncertain
capture or replay failure is fail-stop; cleanup synchronizes before releasing
the stream, events or payload owners. Ownership does not rely on Python GC.

## Native qualification first

Enabling overlap runs a small startup capture probe using the same fork/join
helper. Failure aborts preparation; it does not silently revert after partial
submission. This probe verifies dependency replay, **not useful concurrency**.

Run the standalone probe before loading the model:

```bash
timeout 180s python -m pytest --confcutdir=tests/e2e/singlecard -o addopts= \
  tests/e2e/singlecard/test_sfa_retrieval_overlap.py -k native_capture_contract -q
```

Then qualify registered CPU sources, layer-specific K/PE values, changing
selections, zero/full misses, partial chunks, idle lanes, source replacement,
consumer-boundary visibility, alternating graphs, Q1/Q2 and GC disabled:

```bash
export LMCACHE_ASCEND_SOURCE_DIR=/workspace/sqh/LMCache-Ascend
python -m pytest --confcutdir=tests/e2e/singlecard -o addopts= \
  tests/e2e/singlecard/test_sfa_retrieval_overlap.py -q
```

The native suite requires the deployed vLLM-Ascend and LMCache-Ascend extensions.
It does not substitute an HBM-only copy for the real registered-host transfer.
It covers the transfer contract; model output, actual preemption/resumption,
multi-rank collectives and source-lease retirement still need serving parity
qualification using the existing full-graph harness.

## Initial performance experiment

```bash
python tools/sfa_retrieval_overlap_benchmark.py \
  --lmcache-ascend-dir /workspace/sqh/LMCache-Ascend \
  --requests 8 --query-rows 2 --misses 410 \
  --iterations 100 --repeats 6 \
  --profile-dir overlap-traces --json overlap-r8.json
```

This alternates serial/overlap measurement order, first checks exact transfer
contents, and measures four layers of the real graph retrieval plus the serving
`batch_matmul_transpose` cube projection. It reports median/p95 chain durations,
copied bytes and host enqueue time including timing events. It is **not** a
full-forward or serving benchmark: SFA, O-projection, normalization and MoE are
absent. Event timings may include launch starvation; inspect the exported trace
to confirm side-stream retrieval actually overlaps cube execution. Do not sum
overlapping waits and kernels as if they were serial latency.

Repeat with request counts 1 and 16 and misses 0 and 4096. Then compare actual
serving with the switch off/on, including graph fallback and preemption cases.
Measure full-step median/p95, retrieval waits, projection/MLP slowdown, TPOT,
TTFT, throughput, acceptance rate and copied rows. Keep the feature off unless
paired runs show a repeatable overall gain without correctness or tail-latency
regressions. Memory bandwidth contention can offset the overlap benefit.

## Local validation limits

CPU tests exercise real forward methods, strict event ordering, cleanup,
fixed destinations under both AOT functionalization modes, disjoint shared
storage, immutable payload ownership and GC-disabled lifetime. Native tests and
the benchmark are provided but cannot establish NPU correctness or performance
on a CPU-only development host. No measured serving speedup is claimed here.

## Independent audit — 2026-09-24

**Fixed, reproduced defect:** graph-memory profiling cleanup left temporary KV
alive through the overlap topology (and the original per-layer destination
lists, subsequently removed by the capture dispatch fix). The
regression executes the actual Ascend runner cleanup method and checks tensor
weak references with GC disabled; it failed before the fix. Profiling cleanup
now releases those references after synchronized graph cleanup and before the
parent discards temporary KV. Graph-only resets still retain the real layout
for recapture. Release-before-clear and subsequent layout replacement are tested.

The source and ordering audit confirmed:

- The wrapper resolves `layer.mla_attn.impl`; the new post operator uses the
  Ascend implementation and its existing layer-specific `SparseGraphTransfer`.
  Native retrieval reads the current NPU stream when creating its OpCommand.
- LMCache `prepare_sparse_graph_step` resolves the whole ordered target-layer
  source prefix before replay. Overlap does not advance layerwise generators,
  bypass source readiness, or select another LocalCPU/Mooncake loading policy.
- Existing `SFASourceLease` retains actual allocator objects, not only tensor
  views. Consumer joins preserve the existing root completion/retirement order.
- Group-1 reservation/materialization, remote-fill admission, async stores,
  request scheduling, recovery gates and generation invalidation are unchanged.
  This is an audit of integration with those contracts, not exhaustive native
  qualification of every storage/deployment combination.
- Prepared destinations are resolved inside the opaque operator, including
  after initial compilation without KV. Both AOT functionalization versions,
  payload ownership, inactive/full-graph gating, duplicate suppression,
  partial-capture fail-stop and GC-disabled cleanup have CPU regression coverage.
- Disabled captured execution keeps its original device operator sequence.
  Eager Python does have small new attribute/optional-prefetch checks; zero
  eager overhead is not established. Enabled replay adds device event nodes and
  may contend for memory bandwidth. Full-serving speedup remains unmeasured.

Validation after the fix: **1,025 compilation/graph tests passed, 72 skipped**;
**145 standalone recovery tests passed**; **39 focused overlap cases passed**
(the focused cases are included in the graph suite, not additional passes).
Two compilation modules needing an installed vLLM runtime and the Windows Gloo
device tests were excluded. NPU tests remain unrun on this host. Targeted Ruff,
syntax and whitespace checks passed. Matching sibling source snapshots were
used for CPU tests that inspect the vLLM/LMCache integration contracts.

## Capture dispatch correction — 2026-09-24

A startup log showed `rtStreamWaitEvent` error 107024: a captured consumer wait
had no corresponding event record. The original MLA forward selected the new
operator only when a destination list populated during KV initialization was
nonempty. Initial model profiling runs before that initialization; a cached
compiled callable therefore retained the serial post operator even after edges
were prepared. Consumer pre-compute could then wait for a prefetch never launched.

A regression traces the actual MLA forward before KV setup and executes the
cached graph after setup. It failed on the original dispatch. The choice is now
fixed from the startup knob in the MLA constructor, and destinations are resolved
at opaque-operator execution. There are no KV tensor attributes frozen into the
compiled post call. AOT tests exercise both modes with edges installed only
AFTER initial compilation. Startup/capture bookkeeping also rejects a join with
no matching launch before issuing a native wait; priming events outside capture
is not treated as a record in the captured graph. This bookkeeping never runs
on graph replay. Native recapture still requires qualification on the NPU host.
