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
then ON. Both receive exactly the same saved prompt token IDs. The default
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
