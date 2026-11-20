# SPDX-License-Identifier: Apache-2.0
"""Time the original and packet-native DCP attention chains through a complete shared-index group."""

from benchmark_sfa_dcp_early_unpack import UnpackPipeline
from benchmark_sfa_dcp_query_first import main


def packet_pipeline(args, inputs, sources, group, candidate):
    mode = args.stage if candidate else args.reference_stage
    early_unpack = args.unpack_placement != "off" and mode in ("off", "output")
    pipeline = UnpackPipeline(args, inputs, sources, group, early_unpack)
    pipeline.packet_mode = mode
    return pipeline


if __name__ == "__main__":
    main(comparison="packet-io", pipeline_factory=packet_pipeline, description=__doc__)
