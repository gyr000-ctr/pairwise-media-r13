import math

import pytest

import brdf

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _b():
    return brdf.Ggx()

CASES = [
    ("nonneg", lambda: _b().eval(0.8, 0.8, 0.2, 0.04) >= 0.0),
    ("pdf_positive", lambda: _b().pdf(0.9, 0.2) > 0.0),
    ("bad_roughness", lambda: _raises(lambda: _b().eval(0.8, 0.8, 0.0, 0.04))),
    ("bad_roughness_high", lambda: _raises(lambda: _b().eval(0.8, 0.8, 1.5, 0.04))),
    ("finite", lambda: math.isfinite(_b().eval(0.5, 0.5, 0.5, 0.04))),
    ("type", lambda: isinstance(_b().eval(0.5, 0.5, 0.5, 0.04), float)),
    ("det", lambda: _b().eval(0.5, 0.5, 0.5, 0.04) == _b().eval(0.5, 0.5, 0.5, 0.04)),
    ("grazing", lambda: _b().eval(0.01, 0.9, 0.3, 0.04) > _b().eval(0.9, 0.9, 0.3, 0.04)),
    ("pdf_range", lambda: 0.0 <= _b().pdf(0.5, 0.3) <= 1.0),
    ("bad_pdf", lambda: _raises(lambda: _b().pdf(1.5, 0.3))),
    ("monotone_rough", lambda: _b().eval(0.9, 0.9, 0.1, 0.04) <= _b().eval(0.9, 0.9, 0.6, 0.04)),
    ("bad_f0", lambda: _raises(lambda: _b().eval(0.5, 0.5, 0.5, 1.5))),
    ("zero_l", lambda: isinstance(_b().eval(0.5, 0.0, 0.5, 0.04), float)),
    ("bad_ndv", lambda: _raises(lambda: _b().eval(1.5, 0.5, 0.5, 0.04))),
    ("no_mut", lambda: _b().eval(0.5, 0.5, 0.3, 0.04) is not None),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
