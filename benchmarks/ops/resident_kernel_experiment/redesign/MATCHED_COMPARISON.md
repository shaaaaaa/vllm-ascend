# Matched cache-preparation comparison

This runner compares the original union/finalize/update chain with bounded-position,
hash-snapshot and wide-directory lookup. It uses one input trace and initial cache
for every method. Source identity, valid lengths, initial resident tokens and
physical page permutations are shared. Caches evolve over consecutive steps using
each method's own retention policy; subsequent miss counts need not be equal.

For the exact union-only optimization use `--methods original vector_intersection`.
Both methods then use the original cache/transfer adapter, and every step's valid
resident state and miss/target arrays must match before timing. `--build-dir` is
unnecessary for this pair; `--original-build-dir` must contain the newly built
intersection variant. `union_us_per_step` and `planning_three_us_per_step` report
the old/new metadata costs separately from retrieval. See the parent README for
complete build and benchmark commands.

## Measured scope

- BF16 Group-0 MLA_LATENT: separate 512-element latent and 64-element rotary planes,
  `[blocks,128,1,dimension]` destinations and stacked LMCache CPU chunks.
- CPU chunks come from the production PinMemoryAllocator and its device-pointer
  registration helper. Missing registration or unavailable native ops fails.
- All retrieval uses `sparse_mla_dsa_batched_direct_kv_transfer_prepared`, including
  its count-aware fixed rows. The original path loads only unique misses; resident
  hits remain in place. Its three original kernels are timed separately as well.
  This adapter makes R prepared transfer calls per original step and 3R per
  replacement step (including zero-count calls). It is not the newer serving
  `SparseGraphCopy` kernel that batches requests in one launch. The report names
  this backend explicitly; its winner is not yet a final full-graph serving winner.
- The replacement runs lookup, a native source-descriptor packer, prepared loads
  for CPU misses, HBM resident hits and live tails, then publishes the next snapshot.
  Two disjoint paged payload banks avoid overwriting a source still being consumed.
  Every one of those operations is included in complete-chain timing.
- There is **no full-prefix HBM source** in the replacement. Its earlier synthetic
  fused copy kernel is not used: this is a production-transfer adapter comparison,
  and its packing/extra launches are deliberately measured, not assumed free.

Both paths produce bit-identical selected planes for every eager step and the final
graph-replay output. Input selections may contain padding and duplicate occurrences;
the live tail is read from NPU memory, never from offloaded CPU prefix chunks.
Request epochs and KV versions remain fixed within this benchmark. It does not
qualify quantized/C8 layouts, preemption, admission, Mooncake fetching, SFA itself,
or end-to-end serving performance.

## Build and hardware qualification

Use the installed production LMCache/LMCache-Ascend environment and source CANN's
environment as usual. These commands build only the standalone experiment libraries.

```bash
EXP=benchmarks/ops/resident_kernel_experiment
RED="$EXP/redesign"
OBUILD="$EXP/build-910b3-matched-original"
RBUILD="$RED/build-910b3-matched"
LMA=/workspace/sqh/LMCache-Ascend
bash "$EXP/build.sh" ascend910b3 "$OBUILD" &&
python "$RED/build.py" --soc ascend910b3 --build-dir "$RBUILD" &&
python -m pytest --confcutdir="$RED/tests" -o addopts= "$RED/tests" \
  --redesign-build-dir "$RBUILD" --matched-original-build-dir "$OBUILD" \
  --matched-lmcache-ascend-dir "$LMA" -xq
```

Start with eight requests. Each method is profiled once per repetition over an
entire four-step graph; resets happen outside the trace. Unlike the old sweep,
this default performs sixteen profiler sessions, not ninety.

```bash
python "$RED/matched.py" --original-build-dir "$OBUILD" --build-dir "$RBUILD" \
  --lmcache-ascend-dir "$LMA" --output-dir "$RED/sweeps/matched-r8-shift" \
  --requests 8 --steps 4 --repeats 4 --scenario rank_shift
```

Then repeat with `--requests 1 8 16`, and `stable`, `permuted`, `cold`, using new
output directories. For closer results increase repeats (e.g. 8). Order rotates
so each method occupies every position once per four-repeat block; alternate blocks
reverse the order. CPU registration/allocation is deliberately
outside measurement; prepared objects and graph buffers remain alive until all
work completes. The CPU footprint is approximately `R * prefix * 1152` bytes,
plus allocator alignment; the allocator also has a per-request minimum capacity.

## Real top-k traces

`--trace trace.pt` overrides the synthetic sequence. Use a tensor-only dictionary
loadable with `torch.load(..., weights_only=True)`:

| Key | CPU tensor |
| --- | --- |
| `initial_tokens` | int32 `[R,4096]`, unique resident-prefix identities per request |
| `initial_ready` | bool `[R,4096]`, initially reusable entries |
| `steps` | int32 `[S,R,2,2048]`, original token positions, `-1` for padding |
| `boundary` | int32 `[R]`, fixed offloaded-prefix length, chunk aligned |
| `lengths` | int32 `[R]`, valid length including live NPU tail |

The initial cache must describe the same request lifetime as the sequence. Export
selections before resident remapping; do not use scratch slot IDs. No production
trace collection has been enabled or added to the decode path.

## Report interpretation

`report.json` includes input hashes, loaded transfer file hashes, original/new
build manifests, device identity, per-kernel timing and:

- Original three-kernel time, excluding retrieval.
- Complete graph device span and summed task duration, including descriptor
  adaptation, retrieval, metadata input copies and snapshot publication.
  Timing schema 2 separates `NOTIFY_WAIT`, `NOTIFY_RECORD` and `MODEL_EXECUTE`
  into `control_breakdown_us_per_step`: they can overlap the work they coordinate
  and must not be added to independent kernel cost. `raw_task_sum_us_per_step`
  preserves the unfiltered sum; complete span still covers the entire trace.
- Median/p95 across repeated **rollout averages per step**, not per-request TPOT
  or the p95 of individual decode steps. Overlapping hardware work can make summed
  task duration exceed the wall-clock device span.
- CPU rows/bytes fetched per step, resident-hit and live-tail occurrences, and
  extra HBM payload copies. Original misses are unique tokens; replacement misses
  are occurrences, so compare bytes as well as counts.
- Resident payload footprint and Torch-tracked device allocation including graph
  and reset buffers (the latter is a harness footprint, not production memory).
- A measured winner for each input geometry; not an automatic serving default.

If an adapter dominates, the per-kernel breakdown identifies that integration work.
Do not substitute the earlier 56-us synthetic result for a slower complete result,
or subtract adapter work that the replacement currently requires.
