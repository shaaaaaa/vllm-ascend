# Native DCP packet I/O

This A2 experiment keeps the original fused QKV projection, HCCL collectives,
FP32 softmax merge, resident planner and host transfers. The new native attention
operator writes exactly the 528 halfword lanes expected by the single all-to-all.
Its query loader can read the rank-major gather buffer directly into the
existing Cube L1 tile.

| Stage | Query input | Output |
| --- | --- | --- |
| Baseline | Native RoPE, packet construction, gather, unpack | Native partials, FP32 stats, pack |
| `output` | Same as baseline | Native send packet |
| `direct` | Native RoPE, packet construction, gather, direct L1 load | Native send packet |
| `fused` | One RoPE/packet writer, gather, direct L1 load | Native send packet |

## Build and check packet parity

Use the serving Python, CANN environment and A2 `SOC_VERSION` from your normal
build. Rebuild vLLM-Ascend on each machine that will run the new operator.
The build includes both the ACLNN operator package and the PyTorch extension.

```bash
cd /workspace/sqh/vllm-ascend
git switch perf/sfa-dcp-overlap-unpack
git pull --ff-only
COMPILE_CUSTOM_KERNELS=1 python3 -m pip install -e . --no-build-isolation --no-deps
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest --noconftest -o addopts= -q \
  tests/e2e/singlecard/test_sfa_dcp_packet_io.py
```

The single-device test emulates DCP rank storage for sizes 2, 4 and 8 and runs
both native operators. It compares every bit of the RoPE/gather packet and the
send packet in FP16 and BF16, including empty shards, padded rows, reordered
Top-K, and changed inputs across graph replays.

## Measure complete decoder layer groups

Use eight idle local NPUs for DCP8, or four processes on four NPUs for DCP4.
This smoke run compares the original chain against all three candidates.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
HCCL_OP_EXPANSION_MODE=AIV TASK_QUEUE_ENABLE=1 HCCL_BUFFSIZE=256 \
python3 -m torch.distributed.run \
  --rdzv-backend=static --rdzv-endpoint=127.0.0.1:29571 \
  --nnodes=1 --node-rank=0 --nproc-per-node=8 \
  benchmarks/ops/benchmark_sfa_dcp_packet_io.py \
  --lmcache-ascend-dir /workspace/qzy/LMCache-Ascend \
  --requests 1 --layers 4 --steps 2 --warmup-steps 1 --trials 2 --iterations 2 \
  --json packet-io-smoke.json
```

For timing, use `--requests 1 2 4 8 --steps 16 --warmup-steps 4 --trials 10
--iterations 10 --json packet-io.json`. Each stage has its own paired baseline
and HCCL group. The benchmark checks eagerly and through repeated graph replay,
then alternates the order of timed arms. Each `RESULT` line gives the
slowest-rank median in microseconds per whole four-layer group and its percentage
reduction. Positive values mean the candidate ran faster.

To isolate the loader, use `--reference-stage output --stages direct`.
To isolate bridge fusion on the same communicator and inputs, use
`--reference-stage direct --stages fused`.

By default both arms use early gathering and query-first consumers. The group
includes the native C8 Lightning Indexer on the producer, real sparse H2D, KV
updates, DCP attention, exchange, combine and local output projection. Indexer
inputs and BF16 projection weights are synthetic. TP output reduction, MoE and
server scheduling are outside the timing interval.

Use `--unpack-placement producer` if the baseline being deployed overlaps
producer unpack. Only baseline and `output` execute unpack. Match serving's
`--prefetch-after-pack` setting. When it is off, H2D may already hide part of
the original pack. Native packet writes complete with attention, so the group
timing includes any change in that overlap window. To qualify padding use
`--layouts mixed --padding 1`. Use `--stages fused --profile-dir packet-profiles`
to inspect the final chain and native SFA duration.

## Serving

The startup setting `VLLM_ASCEND_SFA_DCP_PACKET_IO` defaults to `off`. After
qualification, set it to `output`, `direct` or `fused` on all decoder ranks and
restart. It applies to full target DCP graphs using a shared resident plan.
The existing gather and H2D settings still select when communication starts.
The fused bridge combines latent packing with the documented
[InterleaveRope](https://gitcode.com/cann/ops-transformer/tree/9.0.0-beta.1/posembedding/interleave_rope)
arithmetic and preserves the original FP32 multiply/add rounding sequence.
