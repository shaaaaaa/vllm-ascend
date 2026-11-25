# Prefill replay

Reconstruct the original follow-up from `multi-turn-concurrent`, read the previous
raw snapshot up to the original cache boundary, and send it through the running
PD proxy. The request completes through D and saves the generated output.

See [file replay commands, routing and verification](kv_replay_files.md).
