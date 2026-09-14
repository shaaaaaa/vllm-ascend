#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash $0 SOC_VERSION [BUILD_DIR]" >&2
    echo "Use the same SOC_VERSION as your existing vLLM-Ascend build." >&2
    exit 2
fi
: "${ASCEND_HOME_PATH:?Source the installed CANN set_env.sh first}"
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${2:-${source_dir}/build}"
torch_prefix="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
torch_npu_path="$(python3 -c 'from pathlib import Path; import torch_npu; print(Path(torch_npu.__file__).resolve().parent)')"
cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release -DRUN_MODE=npu -DSOC_VERSION="$1" \
    -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DTORCH_NPU_PATH="$torch_npu_path" \
    -DCMAKE_PREFIX_PATH="$torch_prefix"
cmake --build "$build_dir" --target resident_experiment_ops --parallel 2
python3 "$source_dir/resident_experiment.py" "$build_dir" "$1"
echo "Built standalone old/new resident kernels in: $build_dir"
