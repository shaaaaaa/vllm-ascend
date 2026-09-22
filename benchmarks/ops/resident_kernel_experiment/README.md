# Resident kernel experiment

Branch: `perf/resident-kernel-optimization`, based on production `origin/sparse`
at `58cbdbbc15a9fd1afcc31223b9fe018105dd0f1f`.

This is an isolated old/new kernel comparison. It does not enable a new serving
path, change the serving operator registry, or require model weights, vLLM, or
LMCache imports. Do not use this benchmark to claim a serving TPOT/TTFT gain.

## What changes

The union/intersection algorithm is identical in both variants. This first
experiment optimizes `finalize` and fused `update + remap`:

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
zero. Only the generated optimized sources enable it. There is no new runtime environment knob
or serving dispatch check. The baseline/optimized symbols have separate suffixes
to avoid interposition with an installed serving extension.

Both variants reuse the same source rather than maintaining a second large
kernel copy. At CMake configuration, `generate_sources.py` emits six translation
units, each with one explicitly named AIV entry point: old/new union, finalize,
and update. Helper/class bodies and entry-point bodies come verbatim from the
resident source; entry names are literal, not preprocessor aliases. The host
dispatcher preserves the same launch order. This packaging avoids the former
multi-entry-point/include-and-macro build, which failed binary registration on
CANN 8.5.1 with `finalize ... get kernel type failed`. Native confirmation of
the replacement packaging is still required.

The standalone build compiles only these six resident entry points and a small PyTorch-NPU binding.
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
bash "$EXP/build.sh" ascend910b3 "$EXP/build-910b3-single-entry"
python -m pytest --confcutdir="$EXP/tests" -o addopts= "$EXP/tests/test_kernels.py" --resident-build-dir "$EXP/build-910b3-single-entry" -k 'normal-1-1' -xq
```

If registration fails, do not benchmark: successful Python loading alone does
not prove that the AscendC-generated registration stub accepted its binary.

## Correctness tests

```bash
python -m pytest --confcutdir="$EXP/tests" -o addopts= "$EXP/tests" -q
```

For a custom build directory/device, append
`--resident-build-dir /path/to/build --resident-device 0`.

The native tests compare both compiled variants with an independent CPU
set/dictionary oracle, not merely against each other. They cover query widths
1/2, 1/2/4 shards per query row, cold/mixed/all hits, full capacity, subsets,
cross-shard eviction, stale generations, inactive/dummy requests, padding,
negative/out-of-boundary indices, skewed shards, grid-stride execution, and
fragmented physical block tables. Graph tests change inputs/generations/padding
at fixed addresses and queue multiple replays before a final host fence. Launch
shape/dtype/alias errors are rejected before a kernel is submitted.

Only valid prefixes of variable-length payloads are compared; unused workspace
tails have no defined value. State counts/generation padding, remapped top-k,
valid resident slot bijections, exact miss order, and physical destinations are
checked. The fixture poisons previous miss/eviction counters to detect stale
metadata in the new fast path.

Host-only execution requires CPU Torch and pytest; native tests skip if
torch-npu/NPU is unavailable. **Host passes do not validate compiled kernels.**
Initial local validation on Windows: 65 host tests passed, 79 native tests
skipped. Native compilation and execution could not be performed on that host.

## Benchmark

First run a small graph comparison:

```bash
python "$EXP/benchmark.py" --requests 1 8 16 --mtp 2 --shards-per-row 4 --hit-rates 0 0.9 1 --iterations 100 --warmup 20
```

`--mtp` means **query width**: `2` represents one speculative token plus the
verification row; it is not the number of speculative tokens alone.

The default measures the full chain and each stage independently. Baseline
union/finalize prepare identical inputs for isolated later-stage measurements.
All mutable state and raw top-k are restored outside the timed interval on
**every iteration**, including warmup. Otherwise a miss-heavy fixture would
quietly converge to all-hit residency. Old/new execution order alternates.

Results report NPU-event mean/p50/p95 in microseconds, speedup, exact miss count,
and the number of unchanged shards. `results.json` records build identity,
device/runtime versions, parameters and results. The full-chain measurement is
authoritative for the chain; summing independently measured stage timings can
include different event/launch effects.

Additional comparisons:

```bash
# Unchanged-state opportunity despite misses confined to one value shard.
python "$EXP/benchmark.py" --requests 8 --mtp 2 --scenario one_shard_miss --hit-rates 1
# Empty resident state: deliberately exercise the fallback work, not a warm hit.
python "$EXP/benchmark.py" --requests 8 --mtp 2 --scenario cold --hit-rates 0
# Shard-count sensitivity at the same request count/query width.
python "$EXP/benchmark.py" --requests 8 --mtp 2 --shards-per-row 1 2 4 --hit-rates 0.9 1
```

Use `--json FILE` to retain separate runs, `--stage full` for just the full chain,
and `--device N` to choose a device. Run on an otherwise idle NPU and repeat.
`--mode eager` is available, but its event intervals can include host enqueue
gaps; prefer the default graph mode for the production graph-kernel comparison.
Input restoration also creates a controlled, warm-memory microbenchmark rather
than reproducing all cache contention of a serving workload.

Performance gains are expected primarily for unchanged shards/all-hit requests.
The miss-heavy fallback may be neutral or slower because it evaluates an extra
device-side predicate. No speedup is claimed until these measurements pass
correctness and run on the target hardware.
