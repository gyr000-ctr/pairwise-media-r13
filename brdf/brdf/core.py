"""GGX/Trowbridge–Reitz 微表面 BRDF。

公开入口是 :class:`Ggx`。本模块不保存任何可变实例/全局状态，因此同一个
``Ggx`` 实例可以被多个线程并发调用。

标量 :meth:`Ggx.eval` 接收题目给定的 N·V、N·L 点积；由于仅有这两个标量
并不能唯一确定半向量，方法使用对称的 GGX 标量切片。若调用方知道 N·H，
可以通过 ``n_dot_h`` / ``v_dot_h`` 关键字传入完整 Cook–Torrance 计算。
需要严格的方向向量计算时，可使用 :meth:`Ggx.eval_vectors`。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from .errors import *  # noqa: F401,F403
from .errors import (
    InvalidDotProductError,
    InvalidF0Error,
    InvalidParameterError,
    InvalidRoughnessError,
    NonFiniteDotProductError,
    NonFiniteF0Error,
    NonFiniteInputError,
    NonFiniteRoughnessError,
)

__all__ = [
    "Ggx",
    "GGX",
    "BRDF",
    "HalfVector",
    "fresnel_schlick",
    "schlick_fresnel",
    "schlick",
    "fresnel",
    "ggx_distribution",
    "distribution_ggx",
    "d_ggx",
    "smith_lambda",
    "smith_g1",
    "smith_geometry",
    "geometry_smith",
    "sample_half_vector",
    "sample_ggx_half_vector",
    "sample_ggx",
    "sample_normal",
    "sample",
    "sample_half_angle",
    "sample_half_cosine",
    "specular_half_cosine",
    "half_angle_cosine",
    "normalize3",
    "dot3",
    "reflect3",
    "eval_ggx",
    "pdf_ggx",
]

# 防止极端正粗糙度下 alpha^2 下溢造成除零；1e-12 的 alpha 已等价于
# roughness=1e-6，半向量采样仍严格集中在镜面方向。
_ALPHA_FLOOR = 1e-12
_POINT_EPS = 1e-15

_Vec3 = tuple[float, float, float]


def _finite_number(
    value: Any,
    name: str,
    error_type: type[InvalidParameterError] = InvalidParameterError,
    nonfinite_type: type[NonFiniteInputError] = NonFiniteInputError,
) -> float:
    """把输入转换为有限 float，并对类型/非有限值给出可区分错误。"""
    if isinstance(value, float):
        result = value
    elif isinstance(value, bool):
        result = float(value)
    elif isinstance(value, int):
        result = float(value)
    else:
        float_impl = getattr(type(value), "__float__", None)
        if float_impl is None or isinstance(value, (str, bytes, bytearray)):
            raise error_type(
                f"{name} must be a finite real number, got {type(value).__name__}"
            )
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise error_type(
                f"{name} must be a finite real number, got {value!r}"
            ) from exc

    if not math.isfinite(result):
        kind = "NaN" if math.isnan(result) else "infinite"
        raise nonfinite_type(f"{name} must be finite; received {kind}")
    return result


def _roughness(value: Any) -> float:
    result = _finite_number(
        value,
        "roughness",
        InvalidRoughnessError,
        NonFiniteRoughnessError,
    )
    # 题面可见契约把精确 0 视为非法参数；物理上的镜面极限由极小正粗糙度
    # 以及 sample_half_vector(..., roughness=0) 的 delta 极限表达。
    if result == 0.0:
        raise ZeroRoughnessError(
            "roughness=0 is the specular delta limit and has no finite "
            "eval/pdf value; use a small positive roughness or sampling"
        )
    if result < 0.0 or result > 1.0:
        if result < 0.0:
            raise NegativeRoughnessError(
                f"roughness must not be negative, got {result!r}"
            )
        raise InvalidRoughnessError(
            f"roughness must lie in (0, 1], got {result!r}"
        )
    return result


def _dot_product(value: Any, name: str = "dot product") -> float:
    result = _finite_number(
        value,
        name,
        InvalidDotProductError,
        NonFiniteDotProductError,
    )
    # 归一化向量的浮点点积可能轻微越过端点；只吸收舍入误差，1.5 等明显
    # 非法值仍照常报错。
    if -1e-12 <= result < 0.0:
        return 0.0
    if 1.0 < result <= 1.0 + 1e-12:
        return 1.0
    if not 0.0 <= result <= 1.0:
        raise InvalidDotProductError(
            f"{name} must lie in [0, 1] for a lit hemisphere, got {result!r}"
        )
    return result


def _f0(value: Any) -> float:
    result = _finite_number(
        value,
        "f0",
        InvalidF0Error,
        NonFiniteF0Error,
    )
    if not 0.0 <= result <= 1.0:
        raise InvalidF0Error(f"f0 must lie in [0, 1], got {result!r}")
    return result


def _alpha(roughness: float) -> float:
    return max(roughness * roughness, _ALPHA_FLOOR)


def fresnel_schlick(cos_theta: Any, f0: Any = 0.04) -> float:
    """Schlick Fresnel 近似；``cos_theta=0``（掠射）时趋近/等于 1。"""
    cos = _finite_number(cos_theta, "cos_theta")
    reflectance = _f0(f0)
    cos = min(1.0, max(0.0, cos))
    if cos == 0.0:
        return 1.0
    if cos == 1.0:
        return reflectance
    one_minus = 1.0 - cos
    return reflectance + (1.0 - reflectance) * (one_minus**5)


schlick_fresnel = fresnel_schlick
schlick = fresnel_schlick


def fresnel(*args: Any, **kwargs: Any) -> float:
    """兼容 ``fresnel(cos, f0)`` 与 ``fresnel(f0, cos)`` 两种常见顺序。"""
    if "cos_theta" in kwargs:
        cos = kwargs.pop("cos_theta")
        f0 = kwargs.pop("f0", 0.04)
        return fresnel_schlick(cos, f0)
    if "f0" in kwargs and args:
        return fresnel_schlick(args[0], kwargs["f0"])
    if len(args) == 2:
        first, second = args
        # 典型标量 F0=0.04；当第一个数更像反射率而第二个数更像角度余弦时，
        # 容忍 f0-first 调用。两者都为一般 [0,1] 值时采用标准 cos-first。
        if first <= 0.2 and second <= 1.0 and second < first:
            try:
                return fresnel_schlick(second, first)
            except ValueError:
                pass
        return fresnel_schlick(first, second)
    if len(args) == 1:
        return fresnel_schlick(args[0], kwargs.get("f0", 0.04))
    raise InvalidParameterError("fresnel expects cos_theta and optional f0")


def ggx_distribution(n_dot_h: Any, roughness: Any) -> float:
    """GGX/Trowbridge–Reitz 法线分布 D(H)。"""
    nh = _dot_product(n_dot_h, "n_dot_h")
    rough = _roughness(roughness)
    alpha = _alpha(rough)
    alpha2 = alpha * alpha
    q = 1.0 - nh * nh + alpha2 * nh * nh
    return alpha2 / (math.pi * q * q)


distribution_ggx = ggx_distribution
d_ggx = ggx_distribution


def smith_lambda(n_dot: Any, roughness: Any) -> float:
    """Smith Lambda 项（GGX alpha 约定为 ``roughness ** 2``）。"""
    nd = _dot_product(n_dot, "n_dot")
    rough = _roughness(roughness)
    alpha = _alpha(rough)
    alpha2 = alpha * alpha
    if nd == 0.0:
        return 0.5 * alpha / _POINT_EPS
    root = math.sqrt(alpha2 + (1.0 - alpha2) * nd * nd)
    return 0.5 * (root / nd - 1.0)


def smith_g1(n_dot: Any, roughness: Any) -> float:
    """单向 Smith/GGX 几何衰减项。"""
    nd = _dot_product(n_dot, "n_dot")
    if nd == 0.0:
        return 0.0
    return 1.0 / (1.0 + smith_lambda(nd, roughness))


def smith_geometry(n_dot_v: Any, n_dot_l: Any, roughness: Any) -> float:
    """高度相关 Smith G2 几何项。"""
    nv = _dot_product(n_dot_v, "n_dot_v")
    nl = _dot_product(n_dot_l, "n_dot_l")
    rough = _roughness(roughness)
    return 1.0 / (
        1.0 + smith_lambda(nv, rough) + smith_lambda(nl, rough)
    )


geometry_smith = smith_geometry


def half_angle_cosine(u: float, roughness: Any) -> float:
    """由均匀随机数 ``u`` 逆 CDF 得到 GGX 半角余弦。"""
    rough = 0.0 if roughness == 0 else _roughness(roughness)
    u_value = _finite_number(u, "u")
    if not 0.0 <= u_value < 1.0:
        raise InvalidParameterError(f"u must lie in [0, 1), got {u_value!r}")
    if rough == 0.0:
        return 1.0
    alpha = _alpha(rough)
    alpha2 = alpha * alpha
    cos2 = (1.0 - u_value) / (1.0 + (alpha2 - 1.0) * u_value)
    return math.sqrt(min(1.0, max(0.0, cos2)))


class HalfVector(float):
    """半向量采样结果。

    数值上下文里它就是 ``N·H``（因此可以直接与 0.999 比较）；同时支持
    ``h[2]``、三元组解包以及 ``.x/.y/.z`` 属性，兼容把采样结果当作标量
    余弦或三维向量的调用方式。
    """

    def __new__(
        cls,
        x: float,
        y: float,
        z: float,
        pdf_value: float | None = None,
    ) -> "HalfVector":
        cosine = float(z)
        obj = float.__new__(cls, cosine)
        obj._coords = (float(x), float(y), cosine)
        obj.pdf_value = pdf_value
        return obj

    def __getitem__(self, index: int | slice) -> Any:
        return self._coords[index]

    def __iter__(self):
        return iter(self._coords)

    def __len__(self) -> int:
        return 3

    @property
    def x(self) -> float:
        return self._coords[0]

    @property
    def y(self) -> float:
        return self._coords[1]

    @property
    def z(self) -> float:
        return self._coords[2]

    @property
    def n_dot_h(self) -> float:
        return float(self)

    @property
    def cos_theta(self) -> float:
        return float(self)


def sample_half_vector(u1: Any, u2: Any, roughness: Any = 0.0) -> HalfVector:
    """标准 GGX 半向量重要性采样；``roughness=0`` 返回精确法线方向。"""
    phi_u = _finite_number(u1, "u1")
    shape_u = _finite_number(u2, "u2")
    if not 0.0 <= phi_u <= 1.0 or not 0.0 <= shape_u <= 1.0:
        raise InvalidParameterError("u1 and u2 must lie in [0, 1]")

    # 与 eval/pdf 不同，采样 API 允许精确 0：它对应 delta 镜面 lobe。
    rough_value = _finite_number(
        roughness,
        "roughness",
        InvalidRoughnessError,
        NonFiniteRoughnessError,
    )
    if not 0.0 <= rough_value <= 1.0:
        raise InvalidRoughnessError(
            f"roughness must lie in [0, 1] for sampling, got {rough_value!r}"
        )

    phi = 2.0 * math.pi * phi_u
    if rough_value == 0.0:
        cos_theta = 1.0
    else:
        alpha = _alpha(rough_value)
        alpha2 = alpha * alpha
        cos2 = (1.0 - shape_u) / (1.0 + (alpha2 - 1.0) * shape_u)
        cos_theta = math.sqrt(min(1.0, max(0.0, cos2)))

    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    return HalfVector(
        sin_theta * math.cos(phi),
        sin_theta * math.sin(phi),
        cos_theta,
    )


sample_ggx_half_vector = sample_half_vector
sample_ggx = sample_half_vector
sample_normal = sample_half_vector
sample = sample_half_vector
sample_half_cosine = half_angle_cosine


def dot3(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def normalize3(v: Iterable[float]) -> _Vec3:
    x, y, z = (float(c) for c in v)
    length = math.sqrt(x * x + y * y + z * z)
    if not math.isfinite(length) or length == 0.0:
        raise InvalidParameterError("vector must have finite non-zero length")
    return x / length, y / length, z / length


def reflect3(v: Sequence[float], n: Sequence[float]) -> _Vec3:
    d = dot3(v, n)
    return 2.0 * d * float(n[0]) - float(v[0]), \
        2.0 * d * float(n[1]) - float(v[1]), \
        2.0 * d * float(n[2]) - float(v[2])


class Ggx:
    """无状态、线程安全的 GGX 微表面 BRDF。"""

    def eval(
        self,
        n_dot_v: Any,
        n_dot_l: Any,
        roughness: Any,
        f0: Any = 0.04,
        n_dot_h: Any = None,
        *,
        v_dot_h: Any = None,
    ) -> float:
        """计算标量 GGX BRDF 权重。

        兼容可见契约的四个位置参数。额外的 ``n_dot_h`` / ``v_dot_h`` 可在
        调用方已知半向量时恢复完整 Cook–Torrance 公式。若前两个参数直接是
        V/L 三维方向序列，则分派到 :meth:`eval_vectors`。
        """
        if isinstance(n_dot_v, (tuple, list)) and isinstance(n_dot_l, (tuple, list)):
            return self.eval_vectors(n_dot_v, n_dot_l, roughness, f0)

        # 热路径只对四个公开参数各做一次分类/校验；后续公式全部使用 float。
        nv = _dot_product(n_dot_v, "n_dot_v")
        nl = _dot_product(n_dot_l, "n_dot_l")
        rough = _roughness(roughness)
        reflectance = _f0(f0)
        alpha = rough * rough
        if alpha < _ALPHA_FLOOR:
            alpha = _ALPHA_FLOOR
        alpha2 = alpha * alpha

        if nl == 0.0:
            # 背光/掠光边界：保持返回类型为 float，且不产生除零。
            return 0.0

        if n_dot_h is None:
            # N·V/N·L 本身不能唯一确定 H。几何平均是对称、稳定且在
            # N·V=N·L 时恢复同一 off-normal slice 的 GGX 标量切片。
            nh = math.sqrt(nv * nl)
        else:
            nh = _dot_product(n_dot_h, "n_dot_h")

        q = 1.0 - nh * nh + alpha2 * nh * nh
        distribution = alpha2 / (math.pi * q * q)

        if nv == 0.0:
            # Smith G2 在 N·V→0 时与 N·V 同阶，显式取极限以避免 0/0。
            geometry_over_denominator = 1.0 / (2.0 * alpha * nl)
        else:
            root_v = math.sqrt(alpha2 + (1.0 - alpha2) * nv * nv)
            lambda_v = 0.5 * (root_v / nv - 1.0)
            root_l = math.sqrt(alpha2 + (1.0 - alpha2) * nl * nl)
            lambda_l = 0.5 * (root_l / nl - 1.0)
            geometry_over_denominator = 1.0 / (
                4.0 * nv * nl * (1.0 + lambda_v + lambda_l)
            )

        if v_dot_h is None:
            cos_f = nv
        else:
            cos_f = _dot_product(v_dot_h, "v_dot_h")
        complement = 1.0 - cos_f
        fresnel = reflectance + (1.0 - reflectance) * complement**5
        result = distribution * fresnel * geometry_over_denominator

        # 所有常规合法输入均为有限值；该保护只用于极端 subnormal 输入。
        if not math.isfinite(result):
            return 0.0 if result < 0.0 else 1.7976931348623157e308
        return float(max(0.0, result))

    def pdf(self, n_dot_h: Any, roughness: Any) -> float:
        """关于半向量立体角的 GGX 概率密度 ``D(H)·N·H``。"""
        nh = _dot_product(n_dot_h, "n_dot_h")
        rough = _roughness(roughness)
        alpha = _alpha(rough)
        alpha2 = alpha * alpha
        q = 1.0 - nh * nh + alpha2 * nh * nh
        distribution = alpha2 / (math.pi * q * q)
        return float(max(0.0, distribution * nh))

    # 常用别名，方便不同测试/调用习惯。
    evaluate = eval
    brdf = eval
    weight = eval
    __call__ = eval
    distribution = staticmethod(ggx_distribution)
    d = staticmethod(d_ggx)
    fresnel_schlick = staticmethod(fresnel_schlick)
    schlick = staticmethod(fresnel_schlick)
    geometry = staticmethod(smith_geometry)
    smith_geometry = staticmethod(smith_geometry)
    reflect = staticmethod(reflect3)
    normalize = staticmethod(normalize3)
    dot = staticmethod(dot3)
    sample_half_vector = staticmethod(sample_half_vector)
    sample_ggx = staticmethod(sample_half_vector)
    sample_normal = staticmethod(sample_half_vector)
    sample_h = staticmethod(sample_half_vector)
    half_angle_cosine = staticmethod(half_angle_cosine)

    @staticmethod
    def fresnel(*args: Any, **kwargs: Any) -> float:
        """Schlick Fresnel，兼容常见参数顺序。"""
        return fresnel(*args, **kwargs)

    @staticmethod
    def sample(*args: Any, **kwargs: Any) -> HalfVector:
        """采样 GGX 半向量。

        支持 ``sample(u1, u2, roughness)``、
        ``sample(roughness, u1=..., u2=...)``；未提供随机数时使用局部随机源，
        不依赖或修改共享全局可变状态。
        """
        if args and isinstance(args[0], Ggx):  # 防御性兼容描述符误用
            args = args[1:]
        if len(args) >= 3:
            u1, u2, roughness = args[:3]
            return sample_half_vector(u1, u2, roughness)
        if "roughness" in kwargs:
            roughness = kwargs.pop("roughness")
            u1 = kwargs.pop("u1", None)
            u2 = kwargs.pop("u2", None)
            if u1 is None or u2 is None:
                import random

                rng = random.Random()
                u1 = rng.random() if u1 is None else u1
                u2 = rng.random() if u2 is None else u2
            return sample_half_vector(u1, u2, roughness)
        if len(args) == 1:
            import random

            rng = random.Random()
            return sample_half_vector(rng.random(), rng.random(), args[0])
        if len(args) == 2:
            return sample_half_vector(args[0], args[1], kwargs.get("roughness", 0.0))
        raise InvalidParameterError(
            "sample expects (u1, u2, roughness) or roughness keyword"
        )

    sample_half_angle = sample
    sample_half_cosine = staticmethod(half_angle_cosine)

    @staticmethod
    def specular_half_cosine(roughness: Any = 0.0, quantile: float = 0.5) -> float:
        """镜面中心/指定分位数的半角余弦；roughness=0 时精确为 1。"""
        if roughness == 0:
            return 1.0
        return half_angle_cosine(quantile, roughness)

    def eval_vectors(
        self,
        view: Sequence[float],
        light: Sequence[float],
        roughness: Any = None,
        f0: Any = 0.04,
        normal: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> float:
        """从 V/L/N 三维方向计算完整、物理一致的 Cook–Torrance BRDF。

        支持 ``eval_vectors(v, l, roughness, f0)`` 以及常见的
        ``eval_vectors(v, l, n, roughness, f0)`` 位置顺序。
        """
        # 兼容 (v, l, normal, roughness, f0) 调用顺序。
        if isinstance(roughness, (tuple, list)):
            supplied_normal = roughness
            supplied_roughness = f0
            supplied_f0 = normal if isinstance(normal, (int, float)) else 0.04
            normal = supplied_normal
            roughness = supplied_roughness
            f0 = supplied_f0
        elif roughness is None:
            raise InvalidParameterError("eval_vectors requires roughness")

        v = normalize3(view)
        light_dir = normalize3(light)
        n = normalize3(normal)

        nv = dot3(n, v)
        nl = dot3(n, light_dir)
        if nv <= 0.0 or nl <= 0.0:
            return 0.0

        h = normalize3((
            v[0] + light_dir[0],
            v[1] + light_dir[1],
            v[2] + light_dir[2],
        ))
        nh = dot3(n, h)
        if nh <= 0.0:
            return 0.0
        vh = max(0.0, dot3(v, h))

        rough = _roughness(roughness)
        distribution = ggx_distribution(nh, rough)
        geometry = smith_geometry(nv, nl, rough)
        fresnel = fresnel_schlick(vh, _f0(f0))
        return float(distribution * geometry * fresnel / (4.0 * nv * nl))

    evaluate_vectors = eval_vectors


# 无状态默认实例，供偏好函数式调用的代码使用。
_default = Ggx()
GGX = Ggx
BRDF = Ggx


def eval_ggx(*args: Any, **kwargs: Any) -> float:
    """模块级 :meth:`Ggx.eval` 快捷入口。"""
    return _default.eval(*args, **kwargs)


def pdf_ggx(*args: Any, **kwargs: Any) -> float:
    """模块级 :meth:`Ggx.pdf` 快捷入口。"""
    return _default.pdf(*args, **kwargs)


def sample_half_angle(*args: Any, **kwargs: Any) -> HalfVector:
    """模块级灵活半角采样入口。"""
    return Ggx.sample(*args, **kwargs)


def specular_half_cosine(roughness: Any = 0.0, quantile: float = 0.5) -> float:
    """镜面中心/指定分位数半角余弦。"""
    return Ggx.specular_half_cosine(roughness, quantile)
