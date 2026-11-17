# Direct DCP output exchange experiment

This branch prototypes removing `_pack` without changing native attention.
Serving still uses the packed path. The private `_direct_exchange=True`
argument is selected only by this benchmark and its tests; no server knob,
startup probe or native rebuild is added.

Native attention returns contiguous `[query_rows, global_heads, 512]` partials
and two separate FP32 `[query_rows, global_heads]` statistics. Each row contains
contiguous head-owner slices. The candidate exchanges those slices directly,
using one `all_to_all_single` per row and per plane. Receivers contain
`[query_row, source_rank, local_head, feature]`. The existing combine kernel reads
that layout and the separate statistics. It retains normalization before the
weighted sum, zero-count handling and full FP32 max/sum precision.

The candidate has no packet allocation, pack kernel, statistic byte encoding,
or unpacking of statistics. It still performs query gather/unpack and weighted
combination. It makes **3 × query_rows output collectives per layer**, versus one
in the baseline. Communication payload per head is 1,032 bytes instead of 1,056;
that modest reduction may be outweighed by additional collective latency.
This is a measurement candidate, not an established speedup.

Why not one batched send/receive call? The
[HCCL communication guide](https://www.hiascend.com/doc_center/source/en/canncommercial/800/hcclug/hcclug/hcclug_000011.html)
restricts one batch call to one memory segment per peer and direction. Repeating
the same peer for partial/max/sum or successive query rows is not a supported
substitute for packing. The candidate uses existing collectives and never
depends on repeated-peer descriptors or private torch-npu APIs.

## Run

Use idle NPUs: another server on the same devices can cause HCCL port conflicts
and invalid timing. Start with a short eight-rank run:

```bash
cd /workspace/sqh/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
HCCL_OP_EXPANSION_MODE=AIV TASK_QUEUE_ENABLE=1 HCCL_BUFFSIZE=256 \
python3 -m torch.distributed.run \
  --rdzv-backend=static --rdzv-endpoint=127.0.0.1:29571 \
  --nnodes=1 --node-rank=0 --nproc-per-node=8 \
  benchmarks/ops/benchmark_sfa_dcp_exchange.py \
  --requests 1 --steps 2 --trials 2 --iterations 2 \
  --json sfa-dcp-exchange-smoke.json
```

For comparison use `--requests 1 2 4 8 --steps 8 --trials 10 --iterations 10`.
`--profile-dir profiles/dcp-exchange` exports separate rank/variant traces.
Run HOST separately if needed. The existing `VLLM_ASCEND_DCP_HCCL_HOST` override
is honored and recorded, so unset it when comparing AIV against HOST.

Check `--layouts balanced skewed empty --padding 1` separately from the main
timing run. `--dtype float16` exercises FP16; BF16 is the default. Set
`--query-rows 1` for one query per request. `--local-kv-tokens` is rank-local KV
capacity, aligned to 128; 2,048 is the default. Reducing it clips the synthetic
selections. DCP sizes 2 and 4 use the corresponding torchrun rank counts.

The multicard qualification test uses the same complete path:

```bash
HCCL_OP_EXPANSION_MODE=AIV TASK_QUEUE_ENABLE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m torch.distributed.run \
  --rdzv-backend=static --rdzv-endpoint=127.0.0.1:29571 \
  --nnodes=1 --node-rank=0 --nproc-per-node=8 \
  --module pytest --noconftest -o addopts= -q -s \
  tests/e2e/multicard/test_sfa_dcp_exchange.py
```

## Measurement and correctness

Timing covers strided projected queries through packing, query all-gather,
query unpack, **native counted SparseFlashAttentionDcp**, output communication
and weighted combination. Both variants run identical inputs and the same
query path. This intentionally excludes projection/indexer compute, resident
planning, H2D retrieval, MoE, sampling and scheduling. It is a complete DCP
attention comparison, not server TPOT or an isolated `_pack` microbenchmark.

The graph contains multiple different selections/queries and the requested
number of layers per step. Device-event samples are divided by `--steps`, not
by layers. All samples and per-rank values are saved; the summary uses the
slowest rank of each corresponding sample, then computes its mean and median.
A positive `median_reduction_percent` means the direct candidate is faster.
Trials alternate variant order. Warmup, validation, input changes and Gloo
barriers are outside timing.

Before timing, direct eager output must match the packed baseline exactly.
Repeated graph replay changes query storage in place and compares both variants
with a freshly computed eager reference. Errors are agreed over Gloo before
the next phase. Graphs remain alive through the case; the communicator is not
reused after graph release. There is no temporary diagnostic graph.

Host tests additionally exercise DCP=2/4/8, one/multiple query rows, FP16/BF16,
empty shards, all-empty outputs, large equal maxima with unequal sums, and
large finite partial values. They check address arithmetic and source aliasing;
they do not substitute for Triton compilation or HCCL graph execution on NPUs.
