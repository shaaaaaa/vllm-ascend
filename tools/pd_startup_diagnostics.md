# PD startup diagnostics

KV-transfer workers emit `[PD_INIT]` startup boundaries automatically. LMCache
emits nested `[LMCACHE_INIT]` boundaries. Update both repositories and restart
all P/D worker processes; no additional environment variable is required.
Each line includes a hostname and PID. PD lines also identify DP/TP rank and,
after distributed initialization, the global rank. Lines are flushed before
entering each operation, followed by `end` plus elapsed milliseconds or `error`
plus the exception type. These probes add no collective, device wait, background
thread, or timeout change, and do not run in request execution.

## Collect a compact summary

For the supplied TP4-DP4 launcher, run this on each machine after starting the
service, including machines whose workers appear to have initialized normally:

```bash
cd /workspace/sqh/vllm-ascend
python3 tools/pd_startup_summary.py tp4_dp4_logs/*.log
```

Use the original per-DP logs from one startup attempt. The command only reads
files. It prints one row per observed worker, capped below 400 output lines:

```text
h=nodeA p=15010 d=0 t=0 g=0 waiting open=PD.ep_barrier
h=nodeB p=17010 d=2 t=0 g=8 waiting open=LM.mooncake_setup
```

The innermost open phase is an operation whose `begin` has no matching end in
the supplied logs. It identifies where to investigate, not proof of a deadlock.
The first nested error is retained even when enclosing phases also report it.
`complete` means worker warmup ended, not that the HTTP service is ready.
`partial` means the logs contain completed subphases but no warmup completion.
Workers with no markers are absent from the summary: compare the inventory with
the expected local worker count and check process-launch logs for missing ranks.

For detailed markers from one machine:

```bash
grep -hE '\[(PD_INIT|LMCACHE_INIT)\]' tp4_dp4_logs/*.log | tail -n 200
```

## Interpret phases

| Open phase | Operation being entered |
| --- | --- |
| `PD.device_init` / `PD.dist_init` | Device and distributed-group initialization |
| `PD.model_load` / `PD.memory_profile` | Weight loading or memory profiling |
| `PD.kv_connector` / `LM.engine_create` | Connector and LMCache engine construction |
| `PD.kv_alloc` / `PD.kv_reshape` | NPU KV allocation and tensor views |
| `PD.attn_backend` / `PD.draft_attn` / `PD.latent_init` | Attention, drafter and latent-cache initialization |
| `PD.kv_register` | Connector registration, including LMCache post-init |
| `LM.backend_cpu` / shared-slab phases | CPU allocator, publication, receive or attach |
| `LM.mooncake_shared_engine` | Process-wide Mooncake transfer-engine acquisition |
| `LM.mooncake_setup` | Native Mooncake store setup |
| `LM.mooncake_register_cpu` | Registration of the CPU cache buffer |
| `PD.ep_barrier` | Existing EP startup Gloo barrier |
| `PD.warmup_dummy` / `PD.graph_capture` / `PD.atb_warmup` | Computation warmup or graph capture |

At TP4-DP4, each P or D instance has an EP group of 16 workers. The existing
barrier uses the CPU/Gloo group and does not join P and D into a single group.
P with `kv_both` also enters it. A DP enters this barrier only after its local
TP workers finish KV initialization. If one DP waits here while another worker
is still in `kv_register` or a nested LMCache phase, investigate that earlier
phase first. If every expected member entered but none returned, inspect group
membership, process failures and Gloo communication. The EP group leader's
marker includes the member ranks.

The existing native slab allocation markers further distinguish reserve,
first-touch/populate and host registration. Correlate their PID with these outer
markers when the open phase points at the shared CPU allocator.
