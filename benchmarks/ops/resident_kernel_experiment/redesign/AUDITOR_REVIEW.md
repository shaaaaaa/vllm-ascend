# Auditor review of the September 23 kernel measurements

Reviewed the supplied auditor notes against the experiment code and the separate
full-graph LMCache-Ascend worktree. No output-correctness failure is established
by the timing observations. The principal findings concern measurement scope,
repeatability and repeated metadata work.

## Confirmed findings

- `sweep.py` reference is redesign batched-16, not production union/finalize/update.
  The sweep already includes table construction where applicable and payload
  materialization. Adding lookup-only time would double count work.
- Every valid occurrence is copied into the next bank, even on a hit. With
  R=8, Q=2, K=2048 and 256 bytes/row, that is 8 MiB output. The old HBM fixture's
  miss bytes do not measure CPU-to-NPU traffic.
- Bounded lookup currently reads 4096 tags, versions and ready values per tile:
  48 KiB, or 6 MiB logical GM-to-UB input across 128 tiles. The rank-shift fixture
  rolls by two and uses radius two, so its high recall is not evidence about real
  indexer rank movement.
- Wide construction reduces directory slices by eight. Construction requests
  128 MiB versus 16 MiB of tag/readiness reads at the stated geometry; these are
  logical transfer requests, not measured physical HBM transactions. Directory
  lookup still scans sixteen 8192-cell windows per tile (64 MiB logical reads).
- `CopyTuned` overlaps copy-in/copy-out within a pair of banks. It drains the pair
  before reuse and does not overlap the next tile's lookup with current copying.
- Pipeline64/reload and combined/reload select the same native entry and copy
  geometry. Mode zero skips table construction and bounded lookup. Their 26%
  discrepancy remains unexplained; attributing it to directory/interior work is
  unsupported. Equal algorithmic work does not prove identical runtime timing.
- Lookup-only and fused lookup+copy measurements cannot quantify fusion's benefit.
  There is no matching standalone copy-only kernel in that benchmark.
- Dividing an eight-request kernel time by eight gives amortized throughput cost,
  not critical-path layer latency. The sub-10-us target has not been demonstrated.

## Corrections to earlier conclusions

The 56.49 versus 57.42 us bounded-position results do not justify ruling out
pipeline64. Keep both until repeated paired measurements are available. Larger
reported regressions also need controls rather than an automatic implementation
rollback. The matching reload configurations are a useful null comparison.

The new matched harness improves source realism and includes adapter/state costs,
but it invokes the prepared transfer API per request. Full-graph production has
`SparseGraphCopy::Run`, which consumes per-request source tables, counts, limits
and separate K/PE pointers and balances work over active packed entries in one
launch. R versus 3R prepared calls in the current harness cannot determine the
winner for that different serving backend. The new report labels this explicitly.

The matched harness also does not compare the exact same bounded planner under
both separate retrieval and fused registered-host retrieval. That remains a
separate controlled integration experiment. Recomputing lookup in every layer
would be unfair when the deployed indexer group already shares one plan.

## Highest-priority algorithmic experiment after measurement controls

Load bounded metadata only for the positional window used by each 256-entry tile.
For radius two and two query rows the ideal coverage is 520 positions, versus
4096 today. Actual buffer/transfer sizes must account for alignment and clipped
row edges; 520 is not an unconditional DMA size. First retain the existing local
gather/comparison semantics with smaller windows. Do not assume shifted unaligned
vector operands are supported merely because their logical ranges are contiguous.

Preserve full token/version/readiness and 64-bit epoch checks, global source-slot
indices, row-edge clipping and the existing last-match selection order. Test
radii 0/2/32, first/last tiles, duplicates, stale metadata and graph replay before
timing. This is a code-grounded reduction in requested metadata, not a promised
eightfold kernel speedup. No new bounded-window implementation is included here.

## Repeatability control added

`repeatability.py` alternates the two reload configurations in one process on the
same NativeCase/tensor addresses, preserves the exact differing tuning flags,
checks payloads and reports paired deltas. It records best-effort `npu-smi info`
outside timing. A default run uses eight profiler sessions. No kernel rebuild is
needed beyond the current experiment build.

```bash
python "$RED/repeatability.py" --build-dir "$RBUILD" \
  --output-dir "$RED/sweeps/reload-control-1" --repeats 4
```

Repeat in a new process/output directory if a persistent difference remains.
Small within-run p95 values do not replace this between-run check. This control
does not establish a serving winner or directly compare batched-16 to pipeline64.

## Test order

1. Run the same-address reload control; investigate persistent unexplained drift.
2. Qualify and run the registered-source matched harness with shifted and permuted
   selections, then deployment traces and R=1/8/16. Interpret results for its
   explicitly named prepared backend.
3. Use its breakdown to separate lookup, descriptor packing and transfer cost.
4. Implement aligned bounded windows as one independent variant; avoid another
   broad matrix of buffering flags.
5. Before choosing a serving replacement, compare original batched graph retrieval,
   bounded planning plus that same retrieval backend, and fused bounded retrieval
   with equal lookup sharing, source/destination layout and cache-lifetime rules.
