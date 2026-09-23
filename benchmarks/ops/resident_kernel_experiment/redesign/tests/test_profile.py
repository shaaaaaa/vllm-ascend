# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
from decimal import Decimal

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import parse_native_trace


def event(name, ts, dur='2.25', cat='kernel', tid=3):
    return {'ph':'X','name':name,'ts':Decimal(ts),'dur':Decimal(dur),'cat':cat,'pid':1,'tid':tid}


def test_large_timestamps_are_subtracted_exactly():
    trace = [event('build','18000000000000000.00'), event('lookup','18000000000000002.25')]
    result = parse_native_trace(trace, ['build','lookup'], 1)
    assert result['chain_span']['mean_us'] == 4.5
    assert result['kernel_sum']['mean_us'] == 4.5


@pytest.mark.parametrize('trace', [
    [event('lookup','0',cat='cpu_op')],
    [event('lookup','0',dur='0')],
    [event('lookup','0'),event('lookup','1')],
    [event('lookup','NaN')],
])
def test_rejects_missing_invalid_and_ambiguous_traces(trace):
    with pytest.raises(RuntimeError):
        parse_native_trace(trace, ['lookup'], 1)


def test_dependency_order_and_stream_are_checked():
    for second in (event('lookup','1'), event('lookup','3',tid=4)):
        with pytest.raises(RuntimeError):
            parse_native_trace([event('build','0'), second], ['build','lookup'], 1)


def test_cross_iteration_pairing_is_checked():
    with pytest.raises(RuntimeError):
        parse_native_trace([event('lookup','0'),event('lookup','1')], ['lookup'], 2)
