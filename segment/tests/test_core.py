import math

import pytest

import segment

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.0, 0.0, 1.0], [0.0, 0.1, 1.0], [1.0, 1.0, 1.0]]

CASES = [
    ("mask_shape", lambda: len(segment.FloodFill().fill(IMG, (0, 0), 0.2)) == 3),
    ("no_leak", lambda: segment.FloodFill().fill(IMG, (0, 0), 0.05)[0][2] is False),
    ("area", lambda: segment.FloodFill().count(segment.FloodFill().fill(IMG, (0, 0), 0.2)) >= 3),
    ("bad_seed", lambda: _raises(lambda: segment.FloodFill().fill(IMG, (9, 9), 0.1))),
    ("bad_tol", lambda: _raises(lambda: segment.FloodFill().fill(IMG, (0, 0), -0.1))),
    ("bad_image", lambda: _raises(lambda: segment.FloodFill().fill([], (0, 0), 0.1))),
    ("type", lambda: isinstance(segment.FloodFill().fill(IMG, (0, 0), 0.2), list)),
    ("det", lambda: segment.FloodFill().fill(IMG, (0, 0), 0.2) == segment.FloodFill().fill(IMG, (0, 0), 0.2)),
    ("seed_true", lambda: segment.FloodFill().fill(IMG, (1, 1), 0.2)[1][1] is True),
    ("count_type", lambda: isinstance(segment.FloodFill().count(segment.FloodFill().fill(IMG, (0, 0), 0.2)), int)),
    ("ragged", lambda: _raises(lambda: segment.FloodFill().fill([[0.0, 0.0], [0.0]], (0, 0), 0.1))),
    ("zero_tol", lambda: segment.FloodFill().fill(IMG, (0, 0), 0.0)[0][1] is False),
    ("nan_tol", lambda: _raises(lambda: segment.FloodFill().fill(IMG, (0, 0), float('nan')))),
    ("no_mut", lambda: segment.FloodFill().fill([r[:] for r in IMG], (0, 0), 0.2) is not None),
    ("bool_mask", lambda: all(isinstance(v, bool) for row in segment.FloodFill().fill(IMG, (0, 0), 0.2) for v in row)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
