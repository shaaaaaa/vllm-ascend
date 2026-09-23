# SPDX-License-Identifier: Apache-2.0
"""Matched cache-preparation comparison with real registered CPU sources.

Requires installed LMCache-Ascend native ops. No HBM fallback for CPU misses.
Times actual original three kernels, prepared transfers, and replacement adapters.
"""
import argparse
from decimal import Decimal
import gc
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from resident_experiment import load_library as load_original
from native import load_library
from matched_data import make_trace, trace_digest, validate_trace
from matched_runtime import Original, Replacement, Sources, load_transfer_api

METHODS = ('original', 'bounded_position', 'hash_snapshot', 'direct_directory')
ORIGINAL_SYMBOLS = ('dsa_resident_sharded_union_kernel_baseline',
                    'dsa_resident_sorted_finalize_kernel_baseline',
                    'dsa_resident_sorted_update_kernel_baseline')


def device_timing(document, method, steps):
    events = document if isinstance(document, list) else document['traceEvents']
    hardware = {e['pid'] for e in events if e.get('ph') == 'M' and e.get('name') == 'process_name'
                and 'ascend hardware' in str(e.get('args', {}).get('name', '')).lower()}
    tasks = [e for e in events if e.get('ph') == 'X' and 'ts' in e and 'dur' in e
             and (e.get('pid') in hardware or str(e.get('cat', '')).lower() in ('kernel', 'aicore', 'ai_core'))]
    if not tasks:
        raise RuntimeError('missing device tasks; inspect trace')
    tasks.sort(key=lambda e: Decimal(str(e['ts'])))
    start = [Decimal(str(e['ts'])) for e in tasks]
    duration = [Decimal(str(e['dur'])) for e in tasks]
    if any(not x.is_finite() for x in start + duration) or any(x < 0 for x in duration):
        raise RuntimeError('invalid hardware durations')
    end = [s+d for s, d in zip(start, duration, strict=True)]
    names = [str(e.get('name', '')) for e in tasks]
    symbols = ORIGINAL_SYMBOLS if method == 'original' else ('resident_pack_sources_redesign',)
    for symbol in symbols:
        if sum(symbol in name for name in names) != steps:
            raise RuntimeError(f'expected {steps} {symbol} tasks; inspect trace')
    original = sum((d for name, d in zip(names, duration, strict=True)
                    if any(s in name for s in ORIGINAL_SYMBOLS)), Decimal(0))
    breakdown = {}
    for name, d in zip(names, duration, strict=True):
        breakdown[name] = breakdown.get(name, 0.) + float(d) / steps
    return {'complete_sum_us_per_step': float(sum(duration)) / steps,
            'complete_span_us_per_step': float(max(end)-min(start)) / steps,
            'original_three_us_per_step': float(original) / steps,
            'kernel_breakdown_us_per_step': breakdown, 'device_tasks': len(tasks),
            'hardware_tracks': len({(e.get('pid'), e.get('tid')) for e in tasks})}


def stats(values):
    ordered = sorted(values)
    return {'n': len(values), 'mean': statistics.fmean(values), 'median': statistics.median(values),
            'p95': ordered[min(len(ordered)-1, int(.95*len(ordered)))]}


def measurement_order(repeat):
    # Every four repeats gives every method each position exactly once.
    order = list(METHODS[repeat % len(METHODS):] + METHODS[:repeat % len(METHODS)])
    return order[::-1] if (repeat // len(METHODS)) % 2 else order


def capture(case):
    case.reset()
    counts = []
    for i in range(case.s):
        case.step(i)
        torch.npu.synchronize()
        case.check(i)
        counts.append({'cpu_rows': case.last_misses, 'hit_occurrences': case.last_hits, 'live_occurrences': case.last_tail})
    case.reset()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for i in range(case.s):
            case.step(i)
    torch.npu.synchronize()
    case.reset()
    graph.replay()
    torch.npu.synchronize()
    case.check(case.s-1)
    return graph, counts


def measure(case, graph, path):
    import torch_npu
    # Reset and its synchronization are OUTSIDE the profile/graph. Each replay
    # sees the same initial cache, while all consecutive steps evolve naturally.
    case.reset()
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False, profile_memory=False, with_stack=False,
    ) as profiler:
        graph.replay()
        profiler.step()
        torch.npu.synchronize()
    profiler.export_chrome_trace(str(path.resolve()))
    result = device_timing(json.loads(path.read_text(), parse_float=Decimal), case.name, case.s)
    case.check(case.s-1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-build-dir', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--lmcache-ascend-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--trace', type=Path, help='tensor-only .pt: initial_tokens, initial_ready, steps, boundary, lengths')
    parser.add_argument('--requests', type=int, nargs='+', default=[1, 8, 16])
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--scenario', choices=('stable', 'rank_shift', 'permuted', 'cold'), default='rank_shift')
    parser.add_argument('--prefix', type=int, default=131072)
    parser.add_argument('--tail', type=int, default=128)
    parser.add_argument('--chunk-size', type=int, default=1024)
    parser.add_argument('--hit-rate', type=float, default=.9)
    parser.add_argument('--overlap', type=int, default=1024)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if min(*args.requests, args.steps, args.repeats, args.prefix, args.chunk_size) < 1 or args.tail < 0:
        parser.error('invalid geometry/repetition count')
    if args.prefix < 4096 or args.prefix % args.chunk_size:
        parser.error('prefix must fit 4096 resident entries and be chunk aligned')
    import torch_npu  # noqa: F401
    torch.npu.set_device(args.device)
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    build = {'original': load_original(args.original_build_dir), 'redesign': load_library(args.build_dir)}
    api = load_transfer_api(args.lmcache_ascend_dir)
    backend_files = [Path(api.__file__)] + [Path(importlib.import_module(name).__file__)
                     for name in ('lmcache_ascend.v1.npu_connector.utils', 'lmcache_ascend.c_ops')]
    build['loaded_transfer_file_hashes'] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                          for path in backend_files}
    trace_file = validate_trace(torch.load(args.trace, map_location='cpu', weights_only=True)) if args.trace else None
    requests = [trace_file['steps'].shape[1]] if trace_file is not None else args.requests
    report = {'scope': 'BF16 MLA_LATENT 512+64, block=128, registered stacked CPU chunks; cache preparation only',
              'parameters': {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
              'includes': ['input metadata copies', 'lookup/state maintenance', 'source packing',
                           'prepared CPU/HBM transfers', 'snapshot publication'],
              'excludes': ['source registration and allocation', 'fixture reset', 'attention/model execution',
                           'Mooncake fetching and admission', 'request preemption and version changes'],
              'limitations': 'replacement HBM hit/tail copies use the prepared transfer adapter, not the earlier fused synthetic copier',
              'serving_qualified': False, 'device': torch.npu.get_device_name(args.device),
              'statistical_unit': 'one rollout replay averaged over its steps; p95 is across rollout repeats',
              'build': build, 'results': []}
    for r in requests:
        trace = trace_file if trace_file is not None else make_trace(r, args.steps, prefix=args.prefix, tail=args.tail,
                overlap=args.overlap, hit_rate=args.hit_rate, scenario=args.scenario)
        if trace['steps'].shape[2:] != (2, 2048) or torch.any(trace['boundary'] % args.chunk_size):
            raise ValueError('matched native run requires Q=2, K=2048 and chunk-aligned fixed prefixes')
        torch.save(trace, args.output_dir / f'r{r}-inputs.pt')
        source = Sources(api, trace, torch.device('npu', args.device), args.chunk_size)
        cases, graphs, misses, samples, memory = {}, {}, {}, {}, {}
        try:
            for name in METHODS:
                before = torch.npu.memory_allocated()
                cases[name] = Original(source, trace) if name == 'original' else Replacement(source, trace, name)
                graphs[name], misses[name] = capture(cases[name])
                memory[name] = torch.npu.memory_allocated() - before
                samples[name] = []
            for repeat in range(args.repeats):
                order = measurement_order(repeat)
                for name in order:
                    print(f'R={r} repeat={repeat+1}/{args.repeats} {name}', flush=True)
                    path = args.output_dir / f'r{r}-{name}-{repeat}.json'
                    samples[name].append(measure(cases[name], graphs[name], path))
            summary = {}
            for name in METHODS:
                summary[name] = {'complete_sum_us_per_step': stats([x['complete_sum_us_per_step'] for x in samples[name]]),
                                 'complete_span_us_per_step': stats([x['complete_span_us_per_step'] for x in samples[name]]),
                                 'original_three_us_per_step': stats([x['original_three_us_per_step'] for x in samples[name]]),
                                 'source_counts_per_step': misses[name],
                                 'cpu_bytes_per_step': [m['cpu_rows']*1152 for m in misses[name]],
                                 'extra_hbm_copy_bytes_per_step': [0 if name == 'original' else
                                     (m['hit_occurrences']+m['live_occurrences'])*1152 for m in misses[name]],
                                 'resident_payload_bytes': r*4096*1152*(1 if name == 'original' else 2),
                                 'torch_device_bytes_including_graph_and_reset_buffers': memory[name], 'samples': samples[name]}
            base = summary['original']['complete_span_us_per_step']['median']
            for name in METHODS:
                value = summary[name]['complete_span_us_per_step']
                summary[name]['speedup'] = base / value['median']
                print(f"R={r} {name:18} median={value['median']:.2f}us p95={value['p95']:.2f}us "
                      f"speedup={base/value['median']:.3f}x CPU_rows={[m['cpu_rows'] for m in misses[name]]}")
            print(f"R={r} original three kernels only: {summary['original']['original_three_us_per_step']['median']:.2f} us/step")
            winner = min(METHODS, key=lambda name: summary[name]['complete_span_us_per_step']['median'])
            report['results'].append({'requests': r, 'trace_sha256': trace_digest(trace),
                                      'measured_winner': winner, 'methods': summary,
                                      'registered_payload_bytes': source.registered_payload_bytes})
            (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        finally:
            torch.npu.synchronize()
            graphs.clear()
            cases.clear()
            gc.collect()
            source.close()
    print(args.output_dir/'report.json')


if __name__ == '__main__':
    main()
