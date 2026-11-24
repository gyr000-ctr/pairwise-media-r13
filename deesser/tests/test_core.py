import math

import pytest

import deesser

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _d():
    return deesser.DeEsser(48000, 0.3, 4.0)

CASES = [
    ("length", lambda: len(_d().process([0.1] * 100)) == 100),
    ("initial_gain", lambda: abs(_d().gain() - 1.0) < 1e-9),
    ("bad_rate", lambda: _raises(lambda: deesser.DeEsser(0, 0.3, 4.0))),
    ("bad_ratio", lambda: _raises(lambda: deesser.DeEsser(48000, 0.3, 0.5))),
    ("bad_threshold", lambda: _raises(lambda: deesser.DeEsser(48000, 1.5, 4.0))),
    ("empty", lambda: _d().process([]) == []),
    ("type", lambda: isinstance(_d().process([0.1]), list)),
    ("det", lambda: _d().process([0.1] * 10) == _d().process([0.1] * 10)),
    ("range", lambda: all(abs(v) <= 1.0 for v in _d().process([0.9] * 100))),
    ("no_jump", lambda: max(abs(a - b) for a, b in zip(_d().process([0.9] * 50), _d().process([0.9] * 50)[1:])) < 1.0),
    ("state", lambda: isinstance(_d().process([0.5] * 1000), list)),
    ("nan", lambda: _raises(lambda: _d().process([float('nan')]))),
    ("gain_range", lambda: 0.0 < _d().gain() <= 1.0),
    ("quiet", lambda: all(abs(v) < 0.01 for v in _d().process([0.0] * 10))),
    ("bad_buffer", lambda: _raises(lambda: _d().process('abc'))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
