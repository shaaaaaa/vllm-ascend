# SPDX-License-Identifier: Apache-2.0
"""Startup performance gate for standalone proxy and tokenizer processes.

Keep this module independent of vLLM/LMCache imports: proxies can run without
the device runtime. Use the same PD_SERVING_PERF mode contract as the workers.
"""

import os

_MODE = os.environ.get("PD_SERVING_PERF", "0").strip().lower()
_ENABLED = _MODE not in ("", "0", "false", "no", "off")


def serving_perf_enabled() -> bool:
    """Return the host timing gate selected before process startup."""
    return _ENABLED
