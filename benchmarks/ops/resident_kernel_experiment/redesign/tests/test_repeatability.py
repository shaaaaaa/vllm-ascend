# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from native import NativeCase
from repeatability import paired_summary, select_control


def test_reload_controls_keep_effective_parameters_and_symbols():
    case = SimpleNamespace(mode=0)
    case.profile_symbols = lambda fused: NativeCase.profile_symbols(case, fused)
    for name, flags in (('pipeline64', 192), ('combined', 448)):
        select_control(case, name)
        assert case.tuning == flags
        assert case.tuning & 127 == 64
        assert case.tuning & 128
    case.mode = 2
    with pytest.raises(ValueError):
        select_control(case, 'combined')


def test_summary_pairs_by_repeat_not_measurement_order():
    def row(repeat, config, duration):
        return {'repeat': repeat, 'config': config, 'timing': {'chain_span': {'mean_us': duration}}}
    records = [row(0, 'pipeline64', 40), row(0, 'combined', 50),
               row(1, 'combined', 39), row(1, 'pipeline64', 40)]
    result = paired_summary(records)
    assert result['paired_delta_us'] == [10, -1]
    assert result['median_delta_us'] == 4.5
    with pytest.raises(ValueError, match='incomplete'):
        paired_summary(records[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        paired_summary(records + [records[0]])
