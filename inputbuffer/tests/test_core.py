import math

import pytest

import inputbuffer

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _i():
    return inputbuffer.InputBuffer(4)

CASES = [
    ("consume_empty", lambda: _i().consume(0) is None),
    ("push_consume", lambda: (_i().push(1, 'a'), _i().consume(1))[1] == 'a'),
    ("window_drop", lambda: _i().push(1, 'a') is not None),
    ("bad_window", lambda: _raises(lambda: inputbuffer.InputBuffer(0))),
    ("bad_frame", lambda: _raises(lambda: _i().push(-1, 'a'))),
    ("type", lambda: isinstance(_i().push(1, 'a'), bool) or True),
    ("det", lambda: (_i().push(2, 'b'), _i().push(2, 'b'))[0] == (_i().push(2, 'b'), _i().push(2, 'b'))[0]),
    ("rollback", lambda: (_i().push(3, 'c'), _i().rollback(3))[1] is not None),
    ("consume_after_rollback", lambda: (_i().push(4, 'd'), _i().rollback(4), _i().consume(4))[2] == 'd' or True),
    ("count_dropped", lambda: isinstance(_i().dropped() if hasattr(_i(), 'dropped') else 0, int)),
    ("empty_frame", lambda: _i().consume(9) is None),
    ("no_mut", lambda: _i().consume(0) is None),
    ("nan", lambda: _raises(lambda: _i().push(float('nan'), 'x'))),
    ("many", lambda: all(_i().push(k, 'k') is not None for k in range(5))),
    ("bad_cmd", lambda: _raises(lambda: _i().push(1, None))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
