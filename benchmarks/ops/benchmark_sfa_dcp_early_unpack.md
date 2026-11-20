# Overlap DCP query unpack

The decoder can submit `_unpack_queries` to a side stream that waits for the
existing query all-gather. Native attention waits for the two contiguous query
planes. Both streams join on errors. The producer can overlap unpack with
Lightning Indexer and resident planning. A query-first consumer can overlap it
with current-KV normalization, RoPE and cache writes. On a prefetched consumer
the gather is submitted before the incoming H2D wait, and unpack forks after
that wait. If the gather outlasts current-KV work, unpack is still on the
critical path.

## Whole-group A/B

Run on an idle DCP group with the existing native operators. This command uses
eight NPUs.

```bash
cd /workspace/sqh/vllm-ascend
HCCL_OP_EXPANSION_MODE=AIV TASK_QUEUE_ENABLE=1 HCCL_BUFFSIZE=256 \
python3 -m torch.distributed.run \
  --rdzv-backend=static --rdzv-endpoint=127.0.0.1:29571 \
  --nnodes=1 --node-rank=0 --nproc-per-node=8 \
  benchmarks/ops/benchmark_sfa_dcp_early_unpack.py \
  --lmcache-ascend-dir /workspace/qzy/LMCache-Ascend \
  --requests 1 2 4 8 --layers 4 --unpack-placement producer \
  --unpack-cores 12 --json early-unpack-producer.json
```

Both arms gather early and use query-first consumers. The baseline unpacks
on the attention stream with the existing grid. The candidate moves only the
selected layers and caps the number of vector programs per launch. Repeat with
`--unpack-placement consumer` or `both` and a different output filename.
To isolate stream placement, set `--unpack-cores` to the device's vector-core
count so the two arms use the same grid. Use `--profile-dir` to inspect the
exposed gather tail, unpack and competing kernels by layer.

The default `--indexer c8` runs the native quantized Lightning Indexer on the
producer, with 64 index heads of width 128. Its output goes directly into
resident planning and real sparse H2D. Index queries and paged keys are
synthetic. The measured group includes BF16
projections, current-KV updates, DCP attention, output exchange and combine.
Indexer input projections and key updates, MoE, TP output reduction and server
scheduling are outside the measurement. Use `--indexer precomputed --overlap 0.9`
for controlled consecutive-step Top-K overlap. In C8 mode the indexer determines
the misses. The JSON reports them for rank zero.

Timing follows eager and repeated graph parity checks. Compare the slowest-rank
median in microseconds per **whole layer group**. Positive reduction means the
candidate was faster. Use the same `--prefetch-after-pack` setting as serving.

## Serving trial

Set these on every decoder rank and restart the workers.

```bash
export VLLM_ASCEND_SFA_DCP_EARLY_GATHER=1
export VLLM_ASCEND_SFA_DCP_QUERY_FIRST=1
export VLLM_ASCEND_SFA_DCP_EARLY_UNPACK=producer
export VLLM_ASCEND_SFA_DCP_UNPACK_CORES=12
```

`EARLY_UNPACK` defaults to `off`. It applies to full DCP target graphs with
an active shared resident plan. Consumer overlap also requires a qualified
query-first layer. The core setting limits the launch grid. Actual contention
and the best cap depend on the device and workload. This Python scheduling
change reuses the installed native attention and retrieval operators.
