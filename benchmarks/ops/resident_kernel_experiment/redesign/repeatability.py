# SPDX-License-Identifier: Apache-2.0
"""Same-process/address control for pipeline64/reload versus combined/reload."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess

import torch

from benchmark import profile, workload
from candidates import gather_dense
from native import NativeCase, load_library


def select_control(case, name):
    if case.mode != 0 or name not in ('pipeline64', 'combined'):
        raise ValueError('this control is defined only for the two reload configurations')
    case.copy_mode = 'pipelined'
    case.tuning = 64 | 128 | (256 if name == 'combined' else 0)
    case.wide = name == 'combined'
    # mode=0 skips all table work and bounded lookup; both use the same copy kernel.
    assert case.profile_symbols(True) == ['resident_tuned_copy_redesign']


def paired_summary(records):
    pairs = {}
    for record in records:
        pair = pairs.setdefault(record['repeat'], {})
        if record['config'] in pair:
            raise ValueError('duplicate measurement in pair')
        pair[record['config']] = record['timing']['chain_span']['mean_us']
    deltas, ratios = [], []
    for pair in pairs.values():
        if set(pair) != {'pipeline64', 'combined'}:
            raise ValueError('incomplete pair')
        first, second = pair['pipeline64'], pair['combined']
        deltas.append(second-first)
        ratios.append(second/first)
    return {'pairs': len(pairs), 'paired_delta_us': deltas, 'paired_ratio': ratios,
            'median_delta_us': statistics.median(deltas), 'median_ratio': statistics.median(ratios)}


def device_snapshot():
    try:
        result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True,
                                errors='replace', timeout=10)
        return {'time': datetime.now().isoformat(), 'returncode': result.returncode,
                'output': result.stdout + result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'time': datetime.now().isoformat(), 'unavailable': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--requests', type=int, default=8)
    parser.add_argument('--query-rows', type=int, choices=(1, 2), default=2)
    parser.add_argument('--topk', type=int, default=2048)
    parser.add_argument('--universe', type=int, default=131072)
    parser.add_argument('--overlap', type=int, default=1024)
    parser.add_argument('--hit-rate', type=float, default=.9)
    parser.add_argument('--scenario', choices=('stable', 'rank_shift', 'permuted', 'cold'), default='rank_shift')
    parser.add_argument('--kv-width', type=int, default=64)
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='float32')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=4)
    args = parser.parse_args()
    if min(args.requests, args.topk, args.kv_width, args.iterations, args.warmup, args.repeats) < 1:
        parser.error('sizes and repetition counts must be positive')
    if args.universe < args.query_rows*args.topk or not 0 <= args.overlap <= args.topk or not 0 <= args.hit_rate <= 1:
        parser.error('invalid universe, overlap or hit rate')
    import torch_npu  # noqa: F401
    torch.npu.set_device(args.device)
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    build = load_library(args.build_dir)
    query, snapshot, dense = workload(args)
    expected = gather_dense(query, dense)
    device = torch.device('npu', args.device)
    case = NativeCase(query.to(device), snapshot.to(device), dense.to(device), 'reload', copy_mode='pipelined', copy_rows=64)
    addresses = [tensor.data_ptr() for tensor in case.tensors]
    report = {'scope': 'repeatability control only; configurations have identical effective reload work',
              'build': build, 'device': torch.npu.get_device_name(args.device),
              'parameters': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              'addresses': addresses, 'records': [], 'conditions': []}
    for repeat in range(args.repeats):
        report['conditions'].append(device_snapshot())
        order = ('pipeline64', 'combined') if repeat % 2 == 0 else ('combined', 'pipeline64')
        for name in order:
            select_control(case, name)
            assert addresses == [tensor.data_ptr() for tensor in case.tensors]
            case.run(True)
            torch.npu.synchronize()
            torch.testing.assert_close(case.payload.cpu(), expected, rtol=0, atol=0)
            path = args.output_dir / f'{repeat}-{name}.json'
            timing = profile(case, True, args.iterations, args.warmup, path)
            torch.testing.assert_close(case.payload.cpu(), expected, rtol=0, atol=0)
            report['records'].append({'repeat': repeat, 'config': name, 'timing': timing})
            print(f"repeat={repeat+1} {name}: {timing['chain_span']['mean_us']:.2f} us", flush=True)
        report['paired'] = paired_summary(report['records'])
        (args.output_dir / 'control.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['paired'], indent=2))


if __name__ == '__main__':
    main()
