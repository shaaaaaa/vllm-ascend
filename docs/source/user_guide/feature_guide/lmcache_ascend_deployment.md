# LMCache-Ascend Deployment Guide

## Overview

LMCache-Ascend is a community maintained plugin for running LMCache on the Ascend NPU.

We provide a simple deployment guide here. For further info about deployment notes, please refer to [LMCache-Ascend doc](https://github.com/LMCache/LMCache-Ascend/blob/main/README.md)

## Getting Started

### Clone LMCache-Ascend Repo

Our repo contains a kvcache ops submodule for ease of maintenance, therefore we recommend cloning the repo with submodules.

```bash
cd /workspace
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
```

### Docker

```bash
cd /workspace/LMCache-Ascend
docker build -f docker/Dockerfile.a2.openEuler -t lmcache-ascend:v0.3.12-vllm-ascend-v0.11.0-openeuler .
```

Once that is built, run it with the following cmd

```bash
DEVICE_LIST="0,1,2,3,4,5,6,7"
docker run -it \
    --privileged \
    --cap-add=SYS_RESOURCE \
    --cap-add=IPC_LOCK \
    -p 8000:8000 \
    -p 8001:8001 \
    --name lmcache-ascend-dev \
    -e ASCEND_VISIBLE_DEVICES=${DEVICE_LIST} \
    -e ASCEND_RT_VISIBLE_DEVICES=${DEVICE_LIST} \
    -e ASCEND_TOTAL_MEMORY_GB=32 \
    -e VLLM_TARGET_DEVICE=npu \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/localtime:/etc/localtime \
    -v /var/log/npu:/var/log/npu \
    -v /dev/davinci_manager:/dev/davinci_manager \
    -v /dev/devmm_svm:/dev/devmm_svm \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/hccn.conf:/etc/hccn.conf \
    lmcache-ascend:v0.3.12-vllm-ascend-v0.11.0-openeuler \
    /bin/bash
```

### Manual Installation

Assuming your working directory is ```/workspace``` and vllm/vllm-ascend have already been installed.

1. Install LMCache Repo

```bash
NO_CUDA_EXT=1 pip install lmcache==0.3.12
```

2. Install LMCache-Ascend Repo

```bash
cd /workspace/LMCache-Ascend
python3 -m pip install -v --no-build-isolation -e .
```

### Usage

We introduce a dynamic KVConnector via LMCacheAscendConnectorV1Dynamic, therefore LMCache-Ascend Connector can be used via the kv transfer config in the two following setting.

#### Online serving

```bash
python \
    -m vllm.entrypoints.openai.api_server \
    --port 8100 \
    --model /data/models/Qwen/Qwen3-32B \
    --trust-remote-code \
    --disable-log-requests \
    --block-size 128 \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both"}'
```

#### Offline

```python
ktc = KVTransferConfig(
        kv_connector="LMCacheAscendConnector",
        kv_role="kv_both"
)
```

### Experimental shared-indexer resident planning

For structural `indexer_types` groups, set
`VLLM_ASCEND_SFA_SHARED_RESIDENT_PLAN=1` on the decoder before startup to run
the existing resident planner once per producer/consumer group. The default
is `0`; changing it requires a worker restart. Each layer still retrieves its
own KV and retains its private staged bridge copies.

This requires the existing two-group compact-scratch configuration
(`VLLM_ASCEND_DSA_TWO_GROUPS=1`, `VLLM_ASCEND_DSA_UNBUNDLE=1`,
`VLLM_ASCEND_DSA_SHRINK_LATENT=2`, resident caching enabled), top-k 2048,
at most one speculative token, and a common latent metadata/slot layout.
It supports fixed-width native decode and staged PIECEWISE decode under the
existing staged-SFA configuration restrictions, with eager graph mode `NONE`
also accepted. Draft layers and runtime-only IndexCache patterns are excluded.
Incompatible structural group layouts fail startup. Mixed/prefill fallback
uses the existing ordinary planner and invalidates shared residency.

Startup logs identify each `[SFA_SHARED_PLAN]` producer and its members.
Preemption invalidates shared residency even if a resumed request receives
the same physical block IDs. The Ascend fusion compiler keeps the group's
resident buffers in place instead of cloning them during functionalization.
After a partial forward failure, restart the worker: metadata may describe
KV fills that did not complete. Group state lives with the runner; graph
profiling/capture uses the existing private dummy state rows, and capacity
views never create separate real-request residency.

The host regression suite is
`tests/ut/distributed/kv_transfer/test_shared_resident_plan.py`. NPU planner
parity, repeated graph replay, and distinct layer-specific K/PE reachability
are covered by
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_shared_resident_plan.py`.
Before serving rollout, additionally compare deployed SFA/model outputs and
matched latency runs with the flag disabled/enabled. Device traces must show
one union/finalize/update sequence per eligible group, while per-layer KV
transfers remain. Measure bridge copies separately. This switch does not
enable the separate full-target graph integration or change planner kernels.
