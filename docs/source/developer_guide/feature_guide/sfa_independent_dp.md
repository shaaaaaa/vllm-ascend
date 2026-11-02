# Independent DP execution for full SFA

This experimental mode removes the cross-DP **host metadata/preparation vote**
from live target and draft forwards. Each DP selects its own existing graph bucket
or compatible native MC2 path. TP-local checks, EP dispatch/combine, startup capture
coordination and the EngineCore finish check every 32 steps remain.

It is default off. It does not establish that a slower DP can no longer delay
peers inside expert communication, or guarantee a throughput improvement.

## Deployment

Install matching modified `vllm` and `vllm-ascend` Python sources on every decoder
host. These changes do not require rebuilding the C++ extensions or changing the
installed Mooncake library. Restart the whole decoder DP/EP cohort together.

```bash
export VLLM_ASCEND_SFA_INDEPENDENT_DP=1
export VLLM_ASCEND_SFA_SINGLE_AGREEMENT=0
# Optional guardian health lease; default 30 seconds, valid range [2, 86400].
# This does not replace the existing model-execution/startup timeout settings.
export VLLM_ASCEND_SFA_GUARD_TIMEOUT_S=30
```

Keep the existing full-SFA, KV, graph and MTP configuration. The flag must match
on all DP peers. Startup logs `[SFA independent DP] active` only after every TP
worker installs the common qualified mode. Static ineligibility chooses the
established protocol for the whole cohort; mixed flags/contracts fail startup.

The initial qualification scope is Linux MP execution, one host per TP group,
fixed EP membership, PP=PCP=DCP=1, no DBO, regular MC2 V2 on A2/A3, and either no
speculation or the existing one-token MTP schedule. Recovery must remain within
the existing MC2 token bound. Other configurations retain the established path.
Different draft TP subgroups and fused/hierarchical MC2 need separate qualification.

CANN 8.5.1 remains the target stack: unequal-token dispatch/combine is documented
in [CANN 8.5](https://www.hiascend.com/document/detail/en/canncommercial/850/API/aolapi/context/ops-transformer/aclnnMoeDistributeDispatchV2.md),
and the implementation retains the existing default algorithm
arguments. It does not copy newer A3 algorithm names or require a CANN upgrade.
Installed torch-npu, CANN/driver versions and native binaries still need to be
recorded and tested together.

Startup compares the loaded dispatcher, Python sources and content fingerprints
of mapped torch-npu, HCCL, runtime and custom-op libraries. File device/inode checks
reject a replaced mapping. Unavailable identity selects the common legacy path;
differing identities fail activation. Hashing runs once after capture and reads
the native files on each worker, so it adds startup I/O. It does not hash weights
or run during decoding, and it does not prove driver or NPU-kernel compatibility.

The guardian TCP endpoint uses vLLM's existing host-IP selection. On multi-NIC
hosts, keep `VLLM_HOST_IP` reachable from every decoder host. No per-step worker
traffic uses that endpoint.

## Ownership and failure

One CPU-only guardian per executor supervises the registered TP workers. It does
not load torch/NPU libraries, own model/cache data, or send per-step worker messages.
The executor supplies fixed-rate control and observes existing RPC deadlines.
Workers send only registration and fatal-error notifications.
Only the executor's current outstanding-operation set decides RPC expiry. The
guardian enforces its health lease; it does not infer a timeout from an older
heartbeat after the executor may already have consumed the successful response.

Faults terminate the cohort; they do not trigger an independent-rank restart or
switch a live rank back to CPU agreement. Worker/process death is not proof of DMA
completion. Existing LMCache owners retain uncertain leases and quiesce on parent
loss; the guardian does not delete slabs or fabricate successful retirement.

Epoch markers live under `${TMPDIR:-/tmp}/vllm-dp-guardian/`. Successful common
fallback or completed group drain removes its marker. A failed epoch leaves it
in place. Before removing a failed marker and restarting, establish that the
original resource owners have drained, or perform the necessary device/transport
fencing. The marker is a restart barrier for this deployment, not a machine-wide
device fencing service.

## Qualification

Desktop CPU tests cover the actual control protocol, subprocess/TCP lifecycle,
selected runner methods and failure ordering. They do not qualify CANN graph
replay, DMA, full-server correctness or performance.

On both test hosts, run the installed-library probe on otherwise available NPUs:

```bash
# Use a reachable rendezvous address and a different node rank (0 or 1) per host.
torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29617 --module pytest \
  --noconftest -q tests/e2e/multicard/test_sfa_independent_mc2.py
```

Preserve the deployment's HCCL interface configuration. This probe compares
unequal local inputs and mixed native/captured dispatch/combine sequences with
identity expert computation, including communication quantization. It does not
exercise the full SFA attention/cache path or prove server performance.

Then qualify TP4/DP4 serving with identical trace, graph-size list, cache state,
model and MTP settings. Include unequal/idle DPs, Q1/Q2, native recovery, cold
loading, two-group C8 KV, local/Mooncake placement, checkpoint preemption/resumption,
GC disabled and failures during preparation/replay/draft/retirement. Verify all
participants stop on failure and uncertain ownership remains retained. Compare
outputs, finish reasons, MTP acceptance, TTFT, TPOT, throughput, per-rank gaps and
peak/retired memory together. Keep the mode off until that qualification passes.
