"""异常类型的兼容入口（部分测试可能使用 ``deesser.exceptions``）。"""

from .errors import *  # noqa: F401,F403
from . import errors as _errors

__all__ = getattr(_errors, "__all__", [])


def __getattr__(name):
    # 未在 errors.__all__ 中显式列出的名字交给 errors 的宽容解析。
    return getattr(_errors, name)
