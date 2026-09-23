# SPDX-License-Identifier: Apache-2.0
"""Build only the redesign experiment, in the serving torch-npu/CANN environment."""
import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--soc', required=True, help='exact SOC_VERSION used by the deployed extension')
    parser.add_argument('--build-dir', type=Path, default=Path(__file__).resolve().parent / 'build')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    import torch
    import torch_npu
    from native import HERE, source_digest
    home = os.environ.get('ASCEND_HOME_PATH')
    if not home:
        raise RuntimeError('source the deployed CANN set_env.sh; ASCEND_HOME_PATH is required')
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    stamp = build / 'build-info.json'
    # A failed rebuild must not leave a manifest endorsing an older binary.
    stamp.unlink(missing_ok=True)
    subprocess.run(['cmake', '-S', str(HERE / 'native'), '-B', str(build),
                    f'-DSOC_VERSION={args.soc}', f'-DASCEND_HOME_PATH={home}',
                    f'-DTORCH_NPU_PATH={Path(torch_npu.__file__).resolve().parent}',
                    f'-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}'], check=True)
    subprocess.run(['cmake', '--build', str(build), '-j', str(args.jobs)], check=True)
    stamp.write_text(json.dumps({'source_sha256': source_digest(), 'soc': args.soc,
                                 'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
                                 'native_correctness': 'not established by compilation'}, indent=2) + '\n')
    print(stamp)


if __name__ == '__main__':
    main()
