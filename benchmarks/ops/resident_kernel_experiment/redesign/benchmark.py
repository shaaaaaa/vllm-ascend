# SPDX-License-Identifier: Apache-2.0
"""Correctness sweep and strict device-task profiling of native redesigns."""
import argparse
import json
import statistics
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import torch
from candidates import (VARIANTS, Query, Snapshot, assert_safe, attention,
                        gather_dense, lookup, materialize, oracle_sources)
from native import HERE, MODES, NativeCase, load_library


def stats(values):
    ordered = sorted(values)
    return {'n': len(ordered), 'mean_us': statistics.fmean(ordered),
            'p50_us': statistics.median(ordered),
            'p95_us': ordered[min(len(ordered)-1, int(.95 * len(ordered)))]}


def parse_native_trace(document, symbols, iterations):
    events = document if isinstance(document, list) else document.get('traceEvents', [])
    hardware = {e.get('pid') for e in events if e.get('ph') == 'M' and e.get('name') == 'process_name'
                and 'ascend hardware' in str(e.get('args', {}).get('name', '')).lower()}
    grouped = {s: [] for s in symbols}
    for event in events:
        if event.get('ph') != 'X' or 'ts' not in event or 'dur' not in event:
            continue
        args = event.get('args', {})
        category = str(event.get('cat', '')).lower()
        task = str(args.get('Task Type', args.get('task_type', ''))).upper()
        if not (event.get('pid') in hardware or category in ('kernel', 'aicore', 'ai_core')
                or task in ('AI_CORE', 'AI_VECTOR_CORE', 'AIV', 'AIC')):
            continue
        labels = [str(event.get('name', ''))] + [str(v) for k, v in args.items() if 'kernel' in k.lower() and 'name' in k.lower()]
        matches = [s for s in symbols if any(s in label for label in labels)]
        if len(matches) > 1:
            raise RuntimeError('ambiguous device kernel label')
        if matches:
            grouped[matches[0]].append(event)
    counts = {s: len(v) for s, v in grouped.items()}
    if any(c != iterations for c in counts.values()):
        raise RuntimeError(f'expected {iterations} device tasks per symbol, got {counts}; inspect the trace')
    for values in grouped.values():
        values.sort(key=lambda e: Decimal(str(e['ts'])))
    sums, spans = [], []
    previous_end = None
    for i in range(iterations):
        tasks = [grouped[s][i] for s in symbols]
        if len({(t.get('pid'), t.get('tid')) for t in tasks}) != 1:
            raise RuntimeError('candidate stages are not on one device stream')
        starts = [Decimal(str(t['ts'])) for t in tasks]
        durations = [Decimal(str(t['dur'])) for t in tasks]
        if any(not v.is_finite() for v in starts + durations) or any(v <= 0 for v in durations):
            raise RuntimeError('invalid device timing')
        ends = [t+d for t, d in zip(starts, durations, strict=True)]
        if any(ends[j] > starts[j+1] + Decimal('.01') for j in range(len(tasks)-1)):
            raise RuntimeError('overlapping/out-of-order dependency stages')
        if previous_end is not None and starts[0] < previous_end - Decimal('.01'):
            raise RuntimeError('candidate iterations overlap or stage pairing is ambiguous')
        previous_end = ends[-1]
        sums.append(float(sum(durations)))
        spans.append(float(ends[-1] - starts[0]))
    return {'kernel_sum': stats(sums), 'chain_span': stats(spans),
            'kernels': {s: stats([float(e['dur']) for e in values]) for s, values in grouped.items()}}


def profile(case, fused, iterations, warmup, path):
    import torch_npu
    for _ in range(warmup):
        case.run(fused)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        case.run(fused)
    torch.npu.synchronize()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    # Inputs and prior snapshot are immutable; repeated calls cannot turn misses
    # into hits. Table work is rebuilt inside the captured graph on EVERY replay.
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False, profile_memory=False, with_stack=False,
    ) as prof:
        for _ in range(iterations):
            graph.replay()
            prof.step()
        torch.npu.synchronize()
    prof.export_chrome_trace(str(path.resolve()))
    symbols = (['resident_snapshot_build_redesign'] if case.mode >= 3 else [])
    symbols += [('resident_batched_copy_redesign' if case.copy_mode == 'batched'
                 else 'resident_resolve_copy_redesign') if fused else 'resident_lookup_redesign']
    result = parse_native_trace(json.loads(path.read_text(), parse_float=Decimal), symbols, iterations)
    result['trace'] = str(path.resolve())
    return result


def workload(args):
    rng = torch.Generator().manual_seed(args.seed)
    b, q, k, u = args.requests, args.query_rows, args.topk, args.universe
    tokens = torch.stack([torch.randperm(u, generator=rng)[:q*k].reshape(q, k) for _ in range(b)]).int()
    if q == 2 and args.overlap:
        tokens[:, 1, :args.overlap] = tokens[:, 0, :args.overlap]
    previous = tokens.flatten(1).clone()
    if args.scenario == 'permuted':
        for r in range(b):
            for row in range(q):
                tokens[r, row] = tokens[r, row, torch.randperm(k, generator=rng)]
    elif args.scenario == 'rank_shift':
        tokens = tokens.roll(2, dims=-1)
    # Nominal replacement rate; actual recoverable reuse is calculated below.
    replacements = torch.rand(b, q, k, generator=rng) > args.hit_rate
    new = torch.randint(u, (b,q,k), generator=rng, dtype=torch.int32)
    tokens = torch.where(replacements, new, tokens)
    query = Query(tokens, torch.zeros_like(tokens), torch.ones(b,q,dtype=torch.bool),
                  torch.full((b,q), u, dtype=torch.int32), torch.full((b,q), u, dtype=torch.int32),
                  torch.arange(b, dtype=torch.int64)+1, u)
    dense = torch.randn(b,u,args.kv_width,generator=rng).to(getattr(torch, args.dtype))
    prior = dense.gather(1, previous.long()[...,None].expand(b,q*k,args.kv_width)).clone()
    snap = Snapshot(previous, torch.zeros_like(previous), torch.full_like(previous, args.scenario != 'cold', dtype=torch.bool),
                    query.epochs.clone(), prior)
    return query, snap, dense


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('reference','native'), default='reference')
    parser.add_argument('--variants', choices=VARIANTS, nargs='+', default=None)
    parser.add_argument('--build-dir', type=Path, default=HERE / 'build')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--requests', type=int, default=1)
    parser.add_argument('--query-rows', type=int, choices=(1,2), default=2)
    parser.add_argument('--topk', type=int, default=2048)
    parser.add_argument('--universe', type=int, default=131072)
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--hit-rate', type=float, default=.9)
    parser.add_argument('--scenario', choices=('stable','permuted','rank_shift','cold'), default='rank_shift')
    parser.add_argument('--kv-width', type=int, default=64, help='normalized fixture elements, not assumed model geometry')
    parser.add_argument('--dtype', choices=('float16','bfloat16','float32'), default='float32')
    parser.add_argument('--copy-mode', choices=('row', 'batched'), default='row')
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--buckets', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--json', type=Path, default=HERE / 'results.json')
    args = parser.parse_args()
    if min(args.requests,args.topk,args.kv_width,args.iterations,args.warmup) < 1 or args.universe < args.query_rows*args.topk:
        parser.error('positive sizes and universe >= query_rows*topk required')
    if not 0 <= args.hit_rate <= 1 or not 0 <= args.overlap <= args.topk or not 0 <= args.radius <= 32 or args.buckets < 1:
        parser.error('invalid hit rate/overlap/radius/buckets')
    torch.set_num_threads(1)
    variants = args.variants or (list(MODES) if args.backend == 'native' else list(VARIANTS))
    if args.backend == 'native' and any(v not in MODES for v in variants):
        parser.error('some requested variants have functional implementations only; no silent fallback')
    query, snap, dense = workload(args)
    expected = gather_dense(query, dense)
    oracle = oracle_sources(query, snap)
    report = {'backend': args.backend, 'torch': torch.__version__,
              'scope': 'isolated normalized HBM fixture; NOT production SFA/LMCache or serving latency',
              'serving_qualified': False,
              'native_validated': False,
              'parameters': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              'oracle_occurrence_hits': int((oracle >= 0).sum()), 'results': []}
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.backend == 'native':
        report['build'] = load_library(args.build_dir)
        torch.npu.set_device(args.device)
        device = torch.device('npu', args.device)
        report['device'] = torch.npu.get_device_name(args.device)
    for variant in variants:
        if args.backend == 'reference':
            plan = lookup(query, snap, variant, radius=args.radius, buckets=args.buckets)
            actual = materialize(query, snap, plan, dense)
            timing = {'not_measured': 'CPU correctness run; no native or serving performance claim'}
        else:
            case = NativeCase(query.to(device), snap.to(device), dense.to(device), variant,
                              radius=args.radius, buckets=args.buckets, copy_mode=args.copy_mode)
            # Both unfused and fused outputs are correctness-gated before timing.
            case.run(False)
            torch.npu.synchronize()
            assert_safe(query, snap, case.plan)
            case.run(True)
            torch.npu.synchronize()
            plan, actual = case.plan, case.payload.cpu()
            timing = None
        assert_safe(query, snap, plan)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Byte-identical selected KV is the stronger representation invariant.
        # This CPU attention oracle also detects multiplicity/order mistakes.
        if args.kv_width % 2 == 0:
            vectors = torch.randn(args.requests,args.query_rows,2,args.kv_width//2)
            valid = query.masks()[0].reshape_as(query.tokens)
            shape = (*query.tokens.shape, args.kv_width)
            torch.testing.assert_close(attention(vectors,actual.float().reshape(shape),valid),
                                       attention(vectors,expected.float().reshape(shape),valid),rtol=0,atol=0)
        if args.backend == 'native':
            timing = {}
            for fused in (False, True):
                label = 'lookup_plus_hbm_materialize' if fused else 'lookup_including_table_build'
                timing[label] = profile(case, fused, args.iterations, args.warmup,
                                        HERE / 'profiles' / stamp / f'{variant}_{label}.json')
        source = plan.source.cpu()
        misses, hits = int((source == -1).sum()), int((source >= 0).sum())
        record = {'variant': variant, 'correctness': 'selected_kv_and_reference_attention_pass',
                  'copy_mode': args.copy_mode, 'row_bytes': dense.element_size() * args.kv_width,
                  'materialized_bytes': actual.numel() * actual.element_size(),
                  'occurrence_hits': hits, 'offload_occurrence_misses': misses,
                  'miss_payload_bytes': misses * dense.element_size() * args.kv_width,
                  'timing': timing}
        report['results'].append(record)
        print(json.dumps(record), flush=True)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + '\n')
    # This flag means only this fixture passed native checks, not broad coverage.
    report['native_validated'] = args.backend == 'native'
    args.json.write_text(json.dumps(report, indent=2) + '\n')
    print(args.json.resolve())


if __name__ == '__main__':
    main()
