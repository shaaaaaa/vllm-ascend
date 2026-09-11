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
