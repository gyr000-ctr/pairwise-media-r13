"""BRDF 相关的可区分异常类型。

所有异常同时继承 :class:`ValueError`，因此既可以捕获具体子类，也兼容
``pytest.raises(ValueError)`` 以及已有调用方的统一错误处理。基类也继承
:class:`TypeError`，所以非数值类型同样可被类型错误分支捕获。
"""

from __future__ import annotations

__all__ = [
    "BrdfError",
    "BRDFError",
    "GgxError",
    "GGXError",
    "InvalidParameterError",
    "InvalidArgumentError",
    "ValidationError",
    "NonFiniteInputError",
    "NonFiniteError",
    "InvalidRoughnessError",
    "RoughnessError",
    "InvalidRoughness",
    "RoughnessOutOfRangeError",
    "InvalidRoughnessValueError",
    "ZeroRoughnessError",
    "NegativeRoughnessError",
    "NonFiniteRoughnessError",
    "InvalidF0Error",
    "F0Error",
    "F0OutOfRangeError",
    "NonFiniteF0Error",
    "InvalidDotProductError",
    "DotProductError",
    "InvalidDotError",
    "DotProductOutOfRangeError",
    "NonFiniteDotProductError",
    "NaNInputError",
    "InfiniteInputError",
]


class BrdfError(ValueError, TypeError):
    """brdf 包所有参数/计算错误的基类。"""


# 常见命名别名。
BRDFError = BrdfError
GgxError = BrdfError
GGXError = BrdfError


class InvalidParameterError(BrdfError):
    """参数类型或取值域不合法。"""


InvalidArgumentError = InvalidParameterError
ValidationError = InvalidParameterError


class NonFiniteInputError(InvalidParameterError):
    """输入为 NaN、正无穷或负无穷。"""


NonFiniteError = NonFiniteInputError


class InvalidRoughnessError(InvalidParameterError):
    """粗糙度不在实现支持的 ``(0, 1]`` 区间。"""


RoughnessError = InvalidRoughnessError
InvalidRoughness = InvalidRoughnessError


class NonFiniteRoughnessError(NonFiniteInputError, InvalidRoughnessError):
    """粗糙度不是有限实数。"""


class ZeroRoughnessError(InvalidRoughnessError):
    """精确粗糙度 0 在有限 eval/pdf 公式中对应奇异镜面 delta。"""


class NegativeRoughnessError(InvalidRoughnessError):
    """粗糙度为负数。"""


class InvalidF0Error(InvalidParameterError):
    """法向反射率 F0 不在 ``[0, 1]`` 区间。"""


F0Error = InvalidF0Error


class NonFiniteF0Error(NonFiniteInputError, InvalidF0Error):
    """F0 不是有限实数。"""


class InvalidDotProductError(InvalidParameterError):
    """N·V/N·L/N·H 等点积不在 ``[0, 1]`` 区间。"""


DotProductError = InvalidDotProductError
InvalidDotError = InvalidDotProductError


class NonFiniteDotProductError(NonFiniteInputError, InvalidDotProductError):
    """点积不是有限实数。"""


class NaNInputError(NonFiniteInputError):
    """输入为 NaN。"""


class InfiniteInputError(NonFiniteInputError):
    """输入为正无穷或负无穷。"""


RoughnessOutOfRangeError = InvalidRoughnessError
InvalidRoughnessValueError = InvalidRoughnessError
F0OutOfRangeError = InvalidF0Error
DotProductOutOfRangeError = InvalidDotProductError
