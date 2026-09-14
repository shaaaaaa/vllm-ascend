# GLM-5.2 layerwise prefill: single-node profile

Use the four matching `glm52-model-port-prefill-offload` checkouts, including the
load/store protocol fixes. Run inside the Ascend container after installing those
checkouts and their native extensions. This script does not install or rebuild
them. It needs eight free 64-GB NPUs and at least 4 GB of available `/dev/shm` for
the shared CPU cache.

From `vllm-ascend`, replace the model path with the actual local GLM-5.2 directory:

```bash
set -o pipefail
python tools/glm52_prefill_profile.py --model /path/to/GLM-5.2 2>&1 | tee log.log
```

No server or separate client is needed. Defaults:

- TP8, DP1, eight hidden layers, dummy weights, Ascend quantization.
- Preserve the checkpoint's first eight `indexer_types` and `index_topk_pattern`
  entries; do not turn shared-indexer layers into physical indexers. The usual
  GLM-5.2 layout has physical indexers at layers 0, 1, 2 and 6 (zero-based).
- One 30,000-token request, chunked prefill at 4,096 tokens, one output token.
  There is no second warmup request, MTP or decode loop.
- P role, two rotating KV banks, shared local CPU storage, synchronous store
  protocol, eager execution. The next chunk must reload earlier chunks' history.
- Ignore inherited vLLM/LMCache/Mooncake deployment settings **in this process**;
  keep the caller's shell untouched. No remote cache or D node is contacted.

Capture starts **before the request**, not after prefill. After the workers stop,
the script automatically parses all eight worker profiles. Open each worker's
`profile/**/ASCEND_PROFILER_OUTPUT/trace_view.json` in MindStudio. The raw profiler
directories are also retained. Look for CPU operator launches, RoPE operations,
NPU kernels, H2D/D2H and compute/transfer stream overlap across prefill chunks.
This is not a full-decode-graph audit.

An existing nonempty `profile` directory is never overwritten or deleted; use
`--profile-dir profile_next` for another capture. To measure the same request
without profiling overhead:

```bash
python tools/glm52_prefill_profile.py --model /path/to/GLM-5.2 --no-profile 2>&1 | tee log.log
```

`request_seconds` includes the entire chunked prefill, first-token sampling and
engine IPC, but excludes model startup and profiler parsing. The profiled time
is explicitly labeled as including profiler overhead; do not use it to claim a
speedup. For A/B measurements, use the same weights, prompt/chunk sizes and device
configuration, with profiling disabled in both runs. `--load-format auto` uses
real weights instead of dummy weights while retaining the eight-layer override.

## Metadata optimization

Within each P-node forward, each SFA attention group builds shared RoPE/sequence
metadata once. Each layer receives a separate shallow copy with its own latent
and physical-indexer bank views; shared-indexer layers have no indexer views.
The four `(KV group, bank)` table/slot views and their padding are prepared once
per forward instead of repeatedly per layer. Templates do not survive between
forwards. Other builders, graph capture, decode
and CP retain their original build path. Bank synchronization and transfer
completion fences are unchanged.

CPU tests compare the real builder's full metadata against the optimized runner
loop for 8/79/80 layers, changing banks and positions across steps, and check that
RoPE is computed once per group. These tests do not establish NPU speedup or
P-to-D correctness. This single-node profile exercises the P-side CPU-offload
path only; real PD transfer, D-side bootstrap and numerical/model quality still
need separate validation.
