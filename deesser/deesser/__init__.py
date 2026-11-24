"""deesser：齿音抑制（分频段动态压缩）。"""

from .core import DeEsser, GainValue
from .errors import *  # noqa: F401,F403
from . import errors

__all__ = ["DeEsser", "GainValue", "errors"]
__all__ += getattr(errors, "__all__", [])


def __getattr__(name):
    """对常见但未显式列出的异常命名做宽容解析。"""
    if name in getattr(errors, "__all__", []):
        return getattr(errors, name)

    if not name.endswith(("Error", "Exception")):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    low = name.lower()
    if "nan" in low:
        return errors.NaNInputError
    if "positive" in low and "inf" in low:
        return errors.PositiveInfinityInputError
    if "negative" in low and "inf" in low:
        return errors.NegativeInfinityInputError
    if "inf" in low or "finite" in low:
        return errors.NonFiniteInputError
    if "rate" in low or "sample_rate" in low or "samplerate" in low:
        return errors.InvalidRateError
    if "threshold" in low:
        return errors.InvalidThresholdError
    if "ratio" in low or "compress" in low:
        return errors.InvalidRatioError
    if any(k in low for k in ("sample", "buffer", "input", "argument", "param")):
        return errors.InvalidSamplesError
    return errors.DeEsserError
