# SPDX-License-Identifier: Apache-2.0
"""Selected ONLY by the two-holder test; install routing before LMCache init."""

from layerwise_prefill_mooncake_routing import install_routing

install_routing()


class MooncakeRoutingWorker:
    """No tensor probes, model overrides, or normal-serving hooks."""
