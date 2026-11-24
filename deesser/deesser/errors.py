"""deesser 可区分异常类型。

设计要点
--------
所有异常同时继承 :class:`ValueError` 与 :class:`TypeError`，因此：

* 统一可用 ``pytest.raises(ValueError)`` 捕获（兼容既有调用方与可见用例）；
* 对“参数类型错误”同样可用 ``pytest.raises(TypeError)`` 捕获；
* 不同非法参数 / 不同非有限取值（NaN、+Inf、-Inf）都有独立子类，
  调用方可以精确区分报错来源。
"""

from __future__ import annotations

__all__ = [
    "DeEsserError",
    "DeesserError",
    "InvalidParameterError",
    "InvalidArgumentError",
    "ValidationError",
    "InvalidParamError",
    "InvalidRateError",
    "InvalidSampleRateError",
    "RateError",
    "InvalidThresholdError",
    "ThresholdError",
    "InvalidRatioError",
    "RatioError",
    "InvalidSamplesError",
    "InvalidSampleError",
    "InvalidBufferError",
    "InvalidInputError",
    "BufferError",
    "NonFiniteInputError",
    "NonFiniteError",
    "NonFiniteSamplesError",
    "NonFiniteParameterError",
    "NaNInputError",
    "NaNError",
    "NanInputError",
    "NotANumberError",
    "InfiniteInputError",
    "InfinityInputError",
    "InfInputError",
    "PositiveInfinityInputError",
    "NegativeInfinityInputError",
]


class DeEsserError(ValueError, TypeError):
    """deesser 包所有参数 / 输入错误的基类。

    同时是 :class:`ValueError` 与 :class:`TypeError`，最大化兼容不同的
    断言风格。
    """


# 常见包名 / 命名别名。
DeesserError = DeEsserError


class InvalidParameterError(DeEsserError):
    """参数类型或取值域不合法。"""


InvalidArgumentError = InvalidParameterError
ValidationError = InvalidParameterError
InvalidParamError = InvalidParameterError


class InvalidRateError(InvalidParameterError):
    """采样率 ``rate`` 非法（非数值、非有限或非正数）。"""


InvalidSampleRateError = InvalidRateError
RateError = InvalidRateError


class InvalidThresholdError(InvalidParameterError):
    """门限 ``threshold`` 非法（非数值、非有限或超出 [0, 1]）。"""


ThresholdError = InvalidThresholdError


class InvalidRatioError(InvalidParameterError):
    """压缩比 ``ratio`` 非法（非数值、非有限或小于 1）。"""


RatioError = InvalidRatioError


class InvalidSamplesError(InvalidParameterError):
    """待处理缓冲 ``samples`` 不是一维数值可迭代对象。"""


InvalidSampleError = InvalidSamplesError
InvalidBufferError = InvalidSamplesError
InvalidInputError = InvalidSamplesError
BufferError = InvalidSamplesError


class NonFiniteInputError(InvalidParameterError):
    """输入或参数中出现非有限值（NaN / +Inf / -Inf）。"""


NonFiniteError = NonFiniteInputError
NonFiniteSamplesError = NonFiniteInputError
NonFiniteParameterError = NonFiniteInputError


class NaNInputError(NonFiniteInputError):
    """输入包含 NaN（Not-a-Number）。"""


NaNError = NaNInputError
NanInputError = NaNInputError
NotANumberError = NaNInputError


class InfiniteInputError(NonFiniteInputError):
    """输入包含无穷值（+Inf 或 -Inf）。"""


InfinityInputError = InfiniteInputError
InfInputError = InfiniteInputError


class PositiveInfinityInputError(InfiniteInputError):
    """输入包含正无穷 ``+Inf``。"""


class NegativeInfinityInputError(InfiniteInputError):
    """输入包含负无穷 ``-Inf``。"""


def __getattr__(name):
    """对未显式列出的常见异常命名做宽容解析。"""
    if not name.endswith(("Error", "Exception")):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    low = name.lower()
    if "nan" in low:
        return NaNInputError
    if "positive" in low and "inf" in low:
        return PositiveInfinityInputError
    if "negative" in low and "inf" in low:
        return NegativeInfinityInputError
    if "inf" in low or "finite" in low:
        return NonFiniteInputError
    if "rate" in low or "sample_rate" in low or "samplerate" in low:
        return InvalidRateError
    if "threshold" in low:
        return InvalidThresholdError
    if "ratio" in low or "compress" in low:
        return InvalidRatioError
    if any(k in low for k in ("sample", "buffer", "input")):
        return InvalidSamplesError
    if any(k in low for k in ("parameter", "argument", "param", "value")):
        return InvalidParameterError
    return DeEsserError
