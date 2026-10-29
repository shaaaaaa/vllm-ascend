# PD serving performance and diagnostics

Set `PD_SERVING_PERF` before starting the proxy, prefiller and decoder processes.
The setting is read at startup (or the first native diagnostic use); changing
the environment of an already running process is not supported.

| Value | Performance capability |
| --- | --- |
| Unset, empty, `0`, `false`, `no`, `off` | Optional performance timing disabled |
| `1` | Host timing and request/transfer/graph-route correlation |
| `detail` | Host timing plus additional per-layer detail where implemented |
| `device` | Detail plus explicit NPU event timing where implemented |

Values ignore surrounding whitespace and case. Other nonfalse values retain
host timing behavior. Use the documented modes in deployments.

`PD_SERVING_PERF` replaces `LMCACHE_COLD_START_PERF`; the old environment variable
is no longer read. Update all four repositories together, rebuild the changed
native extensions, and update every launcher or container environment. The
standalone proxy and tokenizer use a device-independent local gate.

Existing log markers such as `[LMCACHE_COLD_PERF]` and `[ENGINE_STEP_TIMING]`,
event names and fields are preserved for existing extraction tools. A
`decoder_*` event can originate from common worker code used on either side;
identify the process role from its deployment, host and rank. These events
measure nested or overlapping intervals: do not sum them as independent costs.
`includes_model_compute` and the clock-domain fields are significant. Compare
monotonic timestamps only within their clock domain.

Host modes do not enable NPU event timing or content readback. Device timing
keeps its bounded queue, queries completion and does not add a host synchronize.
Measure normal throughput with performance tracing disabled; use separate
diagnostic runs for attribution.

`prefiller_sample_slow` samples TP0 producer prefill batches at most once every
five seconds and logs completed calls taking at least 500 ms. It separates
grammar/sampling, bookkeeping, MTP proposal/readback, connector finalization and
output-tail host intervals. These intervals include any existing device waits;
they are not measurements of pure NPU compute time. This probe adds no device
events or readbacks. Both roles may use `kv_both`, so it selects prefill-sized
batches; ordinary decoder batches are excluded. It can also describe actual
prefill recomputation on a decoder. `sample_started_monotonic_ms` locates the
call relative to the worker's execution-return timestamp.

The existing `remote_fill_producer_fence_decision` event also reports
`pending_sync_wait_ms`, the nested wait for pending synchronous stores. Correlate
it by request and time; do not add it to the encompassing connector interval.

Content fingerprints (`enable_npu_content_diagnostics`), MTP/target crash dumps
and existing tensor-trace controls remain separate. They can read device data,
write files or intentionally fence execution, so they are not throughput runs.
Operational warnings, fatal ownership diagnostics, admission accounting and
native timeout enforcement remain active when performance tracing is disabled.

The implementation keeps timing/configuration helpers local to each package.
The vLLM-Ascend content callback bridge avoids importing LMCache into model
modules. Shared crash snapshot helpers live in `vllm_ascend/diagnostic_utils.py`;
model-runner device timing lives in `vllm_ascend/worker/serving_perf.py`. Serving
operations and the locations of timing measurements remain in the runner.

When adding an event, guard diagnostic payload construction at the call site.
The logger's internal guard cannot prevent Python from first building keyword
arguments, scanning request/page collections or taking a metrics snapshot.
Keep diagnostic clocks and temporary containers behind the same startup gate.
Do not gate required synchronization, capacity accounting, deadlines or cleanup.
If content diagnostics also need a timing value, collect it when either mode
is enabled. Test both modes independently, including failure cleanup.

## Enhanced proxy workload balancing

`load_balance_proxy_server_enhanced.py` balances P by unfinished work and D by
request count. For D, `--request-count-slack 1` is the default. When the largest and smallest active counts
differ by at most this value, the existing affinity, token-score and fair
tie-breaking policy applies. Otherwise, only least-count destinations compete
under that policy. `0` gives request count strict priority; `-1` disables the
D count filter. No new LMCache or worker configuration is required.

P reserves an estimated work score at selection, including queued requests,
until the prefill response or failure. Unknown prefixes use the original full
prompt score. A current learned prefix discounts only the prompt-dependent
part of that score: uncached suffix tokens count fully and known prefix tokens
count at `--prefill-cache-hit-cost-ratio 0.25` by default. Fixed request cost is
unchanged. This ratio is a tunable heuristic for loading and cache-miss risk,
not a measured speed ratio; valid values are greater than zero and at most one.
Use `1` to disable the discount. Byte-based routing without tokenizer analysis
uses the full score, since its byte lengths cannot be subtracted from token hints.

P chooses the least unfinished work, using active request count, retained KV
and fair rotation to break ties. A known prefix's P affinity may win when its
extra backlog is no greater than both the estimated reuse benefit and the
incoming request's estimated work. A long uncached suffix therefore does not
create a large affinity queue allowance. P's
affinity decision is independent of D load. Completed prefills' KV holds remain
accounted for until their original release point, but no longer make an idle P
look compute-busy. This policy does not migrate running requests or query cache
contents synchronously. P endpoints should identify individual DP instances;
the proxy does not select P ranks hidden behind a shared API endpoint.

Counts represent assignments owned by this proxy, including queued work, not
instantaneous backend scheduler batch sizes. P is counted from selection through
prefill completion. With RemoteFill enabled, D is reserved before prefill starts
and counted until decode completion or request cleanup. This includes admissions
still waiting for P. Selection and reservation happen under the same state lock;
the D endpoint/rank and KV destination then remain fixed for that attempt.
Without RemoteFill, the existing P-then-D selection order is preserved.
Per-rank count filtering also applies to discovered ranks behind one D endpoint;
endpoint selection still uses each endpoint's total assignments.

`GET /stats` exposes `active_requests` in `prefiller_token_load` and
`decoder_token_load`, P's `active_prefill_work`, and D's `rank_active_requests`.
The work score is in the existing proxy score units, not milliseconds.
These counters require
no metrics polling or per-token worker work. They cover traffic through this
single proxy process; independent proxies and requests sent directly to backends
are not included. Compare these counts alongside throughput, TTFT and TPOT.

With `--enable-prefix-affinity-routing --enable-remote-lmcache-store`, a prefix's
first learned placement is retained rather than overwritten after rerouting.
New prefixes are learned after the existing persistence confirmation. The table
remains bounded, and endpoint/epoch validation and discovery refresh remain
enabled. Affinity is a hint, not proof of current cache residency; cache misses
continue through the existing load path. No cache data is moved by this policy.
