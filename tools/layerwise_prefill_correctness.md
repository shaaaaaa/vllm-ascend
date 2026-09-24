# Prefill layerwise numerical validation

Run this on the Ascend server from the `vllm-ascend` checkout:

```bash
set -o pipefail
python tools/layerwise_prefill_correctness.py 2>&1 | tee log.log
```

The default checkpoint is `/workspace/models/GLM-5.2-w4a8c8-0723`, shared with
`layerwise_prefill_profile.py`. Pass `--model /actual/checkpoint` to override it.
Startup prints the checkpoint path and its `indexer_types` producer/shared
schedule, and writes `model_info.json`. Shared layers consume a producer's top-k
result and are not expected to run their own lightning indexer.

## What runs

The launcher runs the complete model twice in separate processes: OFF first,
then ON, unless `--off-dir` selects a saved baseline. Both receive exactly the
same saved prompt token IDs. The default
10,000-token target crosses multiple 4,096-token prefill chunks, including a
partial final chunk. It uses TP8, DP1, FlashComm1, eager execution and MTP1, with
the established profile memory settings. All configuration is in the Python
launcher; no external LMCache configuration file is loaded.

OFF saves complete tensors to files. ON loads the matching OFF tensor at each
probe during inference, calculates numerical statistics, then releases the CPU
buffers. `--save-on-tensors` also archives complete ON tensors. There are no
tensor fingerprints, sampled-value comparisons, stage switches, or shortened
model layers.

Files retain the full intermediate tensors, including TP padding. Numerical
statistics use only valid token rows, determined from the actual TP context;
unused padding cannot dominate the error report. KV is saved in logical token
order, so different physical bank addresses do not count as value differences.

The probes cover every TP rank and main-model prefill layer: decoder inputs and
outputs, attention intermediates, logical KV in token order at its consumption
point, indexer inputs and top-k indices. Producer and shared-indexer layers are
checked against the actual model configuration. The generated first output
token is also compared. MTP/decode intermediate tensors and remote Mooncake
transport are outside this test's scope.
With MTP1, a final chunk containing only one or two tokens uses decode KV
remapping. Such prompts are rejected before model loading; choose a different
`--prompt-tokens` target. Other partial chunks, including TP padding, are covered.

Both cases use the real local shared CPU allocator with merged layer pages.
Because the production layout predicates normally require a Mooncake URL, an
explicit tools-only selector enables that layout without starting a remote
backend. It does not replace allocation, DMA, cache publication, or attention.
The tool rejects incompatible storage configurations and verifies the merged
objects actually used for H2D loads. It does not independently validate the last
chunk's stored pages after request completion. The original OFF local path
rejects asynchronous storage, so OFF uses
synchronous store and ON exercises asynchronous store; the feature flag and
store mode are the only environment differences between the cases.

## Reuse a completed OFF run

```bash
set -o pipefail
python tools/layerwise_prefill_correctness.py \
    --off-dir ./kv-check-10k \
    --run-dir ./kv-check-on-next 2>&1 | tee log.log
```

`--off-dir` accepts either the previous run directory or its `off/` subdirectory.
Only ON is launched. The old ON may have failed or may be absent; only the OFF
run must have completed with full coverage and all required tensor files.
This requires a correctness archive, not a performance profile directory.

The tool reuses the exact saved prompt token IDs without tokenizing again.
Unspecified model, device list, CPU cache capacity and prompt target inherit
the OFF settings; an 80k baseline therefore does not need `--prompt-tokens`
again. Explicit overrides are checked for compatibility. `--prompt-file` cannot
be combined with `--off-dir`, and a different prompt target requires a new OFF.

Before model loading, it checks the checkpoint configuration, engine options,
runtime environment, completed OFF result, rank/layer coverage and tensor file
existence. The checkpoint at the model path should remain the same. Code commits
may differ so that updated ON implementations can use the established baseline.

The new run directory must be empty. It stores a small `off_reference.json`
pointing to the absolute OFF path; the tensor files are neither copied nor
modified. Keep the original OFF archive and its parent `model_info.json`
available. New logs, ON statistics and reports are written to the new directory.
`--compare-only ./kv-check-on-next` automatically follows the saved reference.

OFF can retain historical KV on the device, so it need not perform a CPU-to-NPU
reload between prefill chunks. Its historical KV is still captured and compared
at the attention/indexer consumers. ON must additionally demonstrate actual
merged-page H2D reloads because it reuses two banks across layers.

Older tools incorrectly required H2D evidence from OFF too. An old OFF summary
whose only error is `No actual merged-page H2D source was observed`, with zero
merged and legacy reload sources, is accepted after all tensor, file, model and
layout checks pass. This correction is read-only; the saved archive is not
rewritten. Other coverage errors remain failures and their first concrete causes
are printed. Fresh runs check worker completion before declaring a case complete
or proceeding from OFF to ON; producing an output token alone is insufficient.

## RPC deadline and progress

The correctness launcher sets `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` to 1800
seconds for both cases. Use the CLI option to override it; the launcher selects
its own environment, so exporting that variable outside the script is not an
override:

```bash
set -o pipefail
python tools/layerwise_prefill_correctness.py \
    --rpc-timeout-seconds 3600 2>&1 | tee log.log
```

`sample_tokens` is vLLM's next-token selection RPC, not tensor subsampling.
Its deadline starts when it is enqueued, including time waiting for the preceding
forward pass, CPU tensor copies, file writes and online comparisons. The normal
300-second deadline can expire during this instrumented workload. A longer
deadline allows slow work to complete; it cannot resolve a deadlock. Performance
profile and serving defaults are unchanged.

After warmup, each worker writes its latest progress to
`off/tensors/rankN/progress.json` or `on/tensors/rankN/progress.json`, refreshed
every 30 seconds. It records the RPC operation, step, layer, tensor name, current
phase, phase duration, and completed record/file/byte counts. Phases distinguish
`compute`, `copy_cpu`, `stats`, `save`, `load_off`, `compare` and `manifest`.
Rank 0 prints a short `[PFC]` heartbeat; other ranks print only when stalled.
If one operation remains unchanged for 90 seconds, that rank writes all Python
thread stacks to `stacks.txt` beside its progress file, once for that operation.
The file retains the latest such snapshot. This CPU monitor does not touch NPU
streams or add device synchronization. A Python thread cannot report if another
thread holds the GIL indefinitely, and Python stacks do not expose native device
execution details.

Increasing counts/changing layers show forward progress even if the RPC is
slow. An unchanged phase and its stack locate the next investigation: device
readback, filesystem I/O, CPU comparison, or model/communication execution.
Inspect all ranks because a waiting collective on one rank may be caused by
another rank's slow probe. Completed OFF archives remain reusable when only the
deadline changes, including archives recorded before this option existed.

## Results

The launcher prints its result directory. It contains OFF tensor archives,
per-case logs and coverage records, and the numerical comparison report.
Per-tensor statistics describe each run's value distribution and differences:
mean, standard deviation, range, RMS, absolute errors, RMSE, relative L2 error,
and error relative to the OFF standard deviation. Nonfinite values and integer
index mismatches are reported separately.

Floating-point differences alone do not fail a comparison: inspect their scale
relative to the original values. Incomplete coverage, mismatched shapes/dtypes,
new nonfinite values or changed output tokens are reported as failures. A
successful diagnostic does not mean that every floating-point bit is identical.

```bash
# Keep artifacts at an explicit, initially empty directory.
python tools/layerwise_prefill_correctness.py --run-dir ./kv-check-10k 2>&1 | tee log.log

# Exercise a longer prefix using the existing fixed long article.
python tools/layerwise_prefill_correctness.py --prompt-tokens 80000 2>&1 | tee log.log

# Rebuild the report on a CPU machine without loading the model.
python tools/layerwise_prefill_correctness.py --compare-only ./kv-check-10k
```

Full tensor archives can be large, especially for 80k prompts. The tool keeps
only each probe's working tensors in CPU memory, but preserves all OFF files.
Tensor readback and file I/O perturb scheduling and timing; use
`layerwise_prefill_profile.py` for performance measurements. This diagnostic
does not establish that uninstrumented asynchronous execution is race-free.
