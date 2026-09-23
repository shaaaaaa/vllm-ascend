# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from native import NativeCase
from test_candidates import case


@pytest.mark.parametrize('options,op,tuning,wide,symbol', (
    ({}, 'run_', 0, False, 'resident_resolve_copy_redesign'),
    ({'copy_mode': 'batched'}, 'run_batched_', 0, False, 'resident_batched_copy_redesign'),
    ({'copy_mode': 'pipelined'}, 'run_tuned_', 144, False, 'resident_tuned_copy_redesign'),
    ({'copy_mode': 'batched', 'copy_rows': 64}, 'run_tuned_', 64, False, 'resident_tuned_copy_redesign'),
    ({'copy_mode': 'batched', 'lookup_mode': 'interior'}, 'run_tuned_', 272, False, 'resident_tuned_copy_redesign'),
    ({'table_mode': 'wide'}, 'run_tuned_', 0, True, 'resident_resolve_copy_redesign'),
))
def test_dispatch_and_profile_symbols(monkeypatch, options, op, tuning, wide, symbol):
    query, snap, dense = case()
    native = NativeCase(query, snap, dense, 'hash_snapshot', **options)
    calls = []
    namespace = SimpleNamespace(**{name: (lambda *args, name=name: calls.append((name, args)))
                                  for name in ('run_', 'run_batched_', 'run_tuned_')})
    monkeypatch.setattr(torch.ops, 'resident_redesign', namespace)
    native.run(True)
    assert calls[0][0] == op
    if op == 'run_tuned_':
        assert calls[0][1][-2:] == (tuning, wide)
    assert native.profile_symbols(True) == [
        'resident_wide_build_redesign' if wide else 'resident_snapshot_build_redesign', symbol]


@pytest.mark.parametrize('options', ({'copy_rows': 0}, {'copy_rows': 65}, {'copy_mode': 'bad'},
                                   {'lookup_mode': 'bad'}, {'table_mode': 'bad'},
                                   {'copy_rows': 64}, {'lookup_mode': 'interior'}))
def test_reject_invalid_tuning(options):
    query, snap, dense = case()
    with pytest.raises(ValueError):
        NativeCase(query, snap, dense, 'reload', **options)


def test_sweep_compares_each_variant_against_its_own_reference():
    from sweep import summarize
    def report(times):
        return {'results': [{'variant': variant, 'occurrence_hits': 10, 'offload_occurrence_misses': 2,
                             'timing': {'lookup_plus_hbm_materialize': {
                                 'chain_span': {'mean_us': time, 'p95_us': time + 1}}}}
                            for variant, time in times.items()]}
    rows = summarize({'reference': report({'reload': 40, 'bounded_position': 60}),
                      'combined': report({'bounded_position': 30, 'reload': 50})})
    assert [row['speedup'] for row in rows] == [1, 1, 2, .8]
