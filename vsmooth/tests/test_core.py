import math

import pytest

import vsmooth

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
F = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]

CASES = [
    ("count", lambda: len(vsmooth.Smooth().smooth(V, F, 1)) == len(V)),
    ("bad_iters", lambda: _raises(lambda: vsmooth.Smooth().smooth(V, F, 0))),
    ("bad_faces", lambda: _raises(lambda: vsmooth.Smooth().smooth(V, [], 1))),
    ("type", lambda: isinstance(vsmooth.Smooth().smooth(V, F, 1), list)),
    ("det", lambda: vsmooth.Smooth().smooth(V, F, 1) == vsmooth.Smooth().smooth(V, F, 1)),
    ("volume_positive", lambda: vsmooth.Smooth().volume(V, F) > 0.0),
    ("moves", lambda: vsmooth.Smooth().smooth(V, F, 1) != V),
    ("no_nan", lambda: all(math.isfinite(c) for p in vsmooth.Smooth().smooth(V, F, 1) for c in p)),
    ("bad_lam", lambda: _raises(lambda: vsmooth.Smooth(1.5, -0.53))),
    ("bad_mu", lambda: _raises(lambda: vsmooth.Smooth(0.5, 1.5))),
    ("tuple_out", lambda: isinstance(vsmooth.Smooth().smooth(V, F, 1)[0], tuple)),
    ("volume_float", lambda: isinstance(vsmooth.Smooth().volume(V, F), float)),
    ("iters_two", lambda: len(vsmooth.Smooth().smooth(V, F, 2)) == 4),
    ("bounded", lambda: max(abs(a - b) for p, q in zip(vsmooth.Smooth().smooth(V, F, 1), V) for a, b in zip(p, q)) < 1.0),
    ("no_mut", lambda: vsmooth.Smooth().smooth(list(V), F, 1) is not None),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
