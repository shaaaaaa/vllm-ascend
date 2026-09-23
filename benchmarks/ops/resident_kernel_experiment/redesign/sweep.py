# SPDX-License-Identifier: Apache-2.0
"""Compare independent kernel optimizations with the validated batched-16 path."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

CONFIGS = {
    'reference': [],
    'batch32': ['--copy-rows', '32'],
    'batch64': ['--copy-rows', '64'],
    'pipeline16': ['--copy-mode', 'pipelined'],
    'pipeline32': ['--copy-mode', 'pipelined', '--copy-rows', '32'],
    'pipeline64': ['--copy-mode', 'pipelined', '--copy-rows', '64'],
    'interior': ['--lookup-mode', 'interior'],
    'wide': ['--table-mode', 'wide'],
    'combined': ['--copy-mode', 'pipelined', '--copy-rows', '64',
                 '--lookup-mode', 'interior', '--table-mode', 'wide'],
}


def summarize(reports):
    result = []
    baseline = {r['variant']: r['timing']['lookup_plus_hbm_materialize']['chain_span']['mean_us']
                for r in reports['reference']['results']}
    for config, report in reports.items():
        for row in report['results']:
            timing = row['timing']['lookup_plus_hbm_materialize']['chain_span']
            result.append({'config': config, 'variant': row['variant'], 'mean_us': timing['mean_us'],
                           'p95_us': timing['p95_us'], 'speedup': baseline[row['variant']] / timing['mean_us'],
                           'hits': row['occurrence_hits'], 'misses': row['offload_occurrence_misses']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--configs', nargs='+', choices=CONFIGS, default=list(CONFIGS))
    args, benchmark_args = parser.parse_known_args()
    if benchmark_args[:1] == ['--']:
        benchmark_args = benchmark_args[1:]
    if any(arg.split('=')[0] in ('--copy-mode', '--copy-rows', '--lookup-mode', '--table-mode', '--json', '--backend')
           for arg in benchmark_args):
        parser.error('sweep owns backend, tuning and output options; pass geometry/build options only')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reports = {}
    for config in dict.fromkeys(['reference', *args.configs]):
        output = args.output_dir / f'{config}.json'
        log = output.with_suffix('.log')
        command = [sys.executable, str(Path(__file__).with_name('benchmark.py')),
                   '--backend', 'native', *benchmark_args, '--copy-mode', 'batched',
                   *CONFIGS[config], '--json', str(output)]
        print(f'Running {config}; log: {log}', flush=True)
        with log.open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        reports[config] = json.loads(output.read_text())
    summary = summarize(reports)
    for row in summary:
        print(f"{row['config']:12} {row['variant']:18} {row['mean_us']:8.2f} us "
              f"p95={row['p95_us']:.2f} speedup={row['speedup']:.3f}x hits={row['hits']} misses={row['misses']}")
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
