# Offline graph-mode chunked-prefill benchmark

This is a single-machine offline hardware test. It does not start an HTTP
server, does not require P/D disaggregation, and does not use LMCache or KV
transfer. The script constructs exact token-ID inputs and calls vLLM's local
`LLM.generate()` API. Only the prefill model-forward steps are timed.

The default workload compares chunk sizes 4096 and 8192 using the same 65536
token prompt. Each chunk size runs in a fresh subprocess so graph and device
state cannot leak between cases. Data parallel and pipeline parallel sizes are
fixed to 1; tensor and expert parallelism can use the model's normal settings.

## Run on one 8-NPU machine

From the modified `vllm-ascend` checkout:

```bash
cd /workspace/sqh/vllm-ascend
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python benchmarks/chunked_prefill_graph/benchmark_chunk_sizes.py run \
  --model /workspace/models/GLM-5.1-w4a8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization ascend \
  --gpu-memory-utilization 0.93 \
  --prompt-tokens 65536 \
  --chunk-sizes 4096 8192 \
  --warmups 1 \
  --repeats 5 \
  --additional-config-json \
    '{"recompute_scheduler_enable":false,"multistream_overlap_shared_expert":false,"fuse_muls_add":true,"fuse_qknorm_rope":false,"enable_npugraph_ex":true,"layer_sharding":["q_b_proj"]}' \
  --output-dir /workspace/sqh/chunk-profile/offline-64k
```

Install the plotting dependency once if it is not already present:

```bash
pip install matplotlib
```

Do not run an existing `vllm serve` process on these eight NPUs at the same
time. The command itself initializes a local offline engine twice, first for
4096 and then for 8192. It deliberately does not accept DP, speculative
decoding, connector, or eager-mode arguments.

Use a new empty `--output-dir` for every run. If the directory already contains
JSONL profile data, the script refuses to mix the runs.

## Outputs

The output directory contains:

- `c4096/` and `c8192/`: raw per-chunk, per-rank NPU Event records and offline
  request wall times;
- `results.json`: every aggregated point and total;
- `plots/chunk_timeline.png`: each chunk's time and average per-layer time
  against its historical-context length;
- `plots/chunk_size_comparison.png`: median one-chunk time and total 64K
  prefill model-forward time against chunk size.

The TP aggregation takes the slowest rank because it determines layer
completion. Warmup requests are excluded. A measured request is rejected if
it captures a new graph or if any prefill step falls back to eager execution.

The per-layer value is the complete target-model forward divided by the known
hidden-layer count. In PIECEWISE graph mode, Python layer hooks do not execute
during graph replay and compiler partitions do not exactly equal original
decoder layers, so this is intentionally reported as an average rather than an
exact layer-by-layer trace.

## Re-summarize existing raw data

```bash
python benchmarks/chunked_prefill_graph/benchmark_chunk_sizes.py summarize \
  --input /workspace/sqh/chunk-profile/offline-64k \
  --prompt-tokens 65536 \
  --warmup-requests 1 \
  --output-json /workspace/sqh/chunk-profile/offline-64k/results.json \
  --plot-dir /workspace/sqh/chunk-profile/offline-64k/plots
```
