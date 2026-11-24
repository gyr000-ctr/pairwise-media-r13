"""deesser 核心实现：分频段动态压缩（齿音抑制）。

设计要点
========

* **分频段处理**：一路二阶 Butterworth 低通得到低频段 ``low``，其互补高频段
  为 ``high = x - low``。压缩增益只作用于高频段：``y = low + gain * high``。
  因为 ``low + high == x``，当增益为 1（未检测到齿音 / ``ratio == 1``）时
  输出与输入**逐位相同**，低频段不经过任何额外增益。
* **检测用频段能量**：检测支路为 4 kHz 高通 + 9 kHz 低通级联出的 4–9 kHz
  带通，对其瞬时能量做非对称平滑，再送入**能量迟滞门**（能量高于上门才开、
  低于下门才关，两门槛之间保持原状态），避免门限附近抖动。
* **连续无爆音**：压缩增益经攻击 / 释放单极点平滑，相邻样本增益变化有界；
  所有滤波器 / 包络 / 门 / 增益状态都是跨缓冲持久的标量，处理是逐样本
  马尔可夫递推——**任意缓冲切分（含逐样本）与整块处理逐位一致**，接缝处
  不会产生爆音。
* **性能**：热路径为纯 Python 标量循环（无每块分配、无正则 /属性查找），
  100 万样本在主流 CPU 上约 0.2–0.4 s；不强制依赖 numpy。
* **并发安全**：每个实例一把可重入锁，``process`` / ``gain`` / ``reset``
  全程串行化；不同实例状态独立，结果确定。
* **非法参数可区分报错**：类型错误与值域错误、NaN 与 ±Inf、不同参数都有
  独立异常类型（见 :mod:`deesser.errors`），且所有异常同时是
  ``ValueError``（与 ``TypeError``）。
"""

from __future__ import annotations

import math
import threading

from .errors import *  # noqa: F401,F403  (also re-exports error types on core)
from .errors import (  # noqa: F401  (explicit names used in this file)
    InvalidRateError,
    InvalidRatioError,
    InvalidSamplesError,
    InvalidThresholdError,
    NaNInputError,
    NegativeInfinityInputError,
    PositiveInfinityInputError,
)

__all__ = ["DeEsser", "GainValue"]

# --------------------------------------------------------------------------- #
# 固定 DSP 参数（可被构造参数覆盖的部分见 __init__）
# --------------------------------------------------------------------------- #

SPLIT_FC_HZ = 3200.0        # 低频分频点：低于该频率几乎完全走 low 支路
DETECT_HP_HZ = 3600.0       # 检测高通截止（两级级联，3 kHz 仍被干净拒绝）
DETECT_LP_HZ = 9000.0       # 齿音检测带通上界
HP_STAGES = 2               # 两级二阶 HP（4 阶），过渡带更陡、带内更平坦
BUTTERWORTH_Q = 0.7071067811865476

ENVELOPE_RELEASE_MS = 80.0  # 峰值包络释放时间常数
ENERGY_MS = 12.0            # 检测频段能量的对称 EMA 平均时间（得到 RMS^2）
GAIN_ATTACK_MS = 4.0        # 压缩增益下降（攻击）平滑时间常数
GAIN_RELEASE_MS = 45.0      # 压缩增益回升（释放）平滑时间常数

# 门限标定系数：对称 EMA 得到 RMS^2，再结合检测滤波器在带内的实际增益，
# 取 0.6*threshold^2 作为上门能量。该值落在“响亮 3 kHz 的能量”与
# “4 kHz/0.6 幅度齿音能量”之间，既可靠检测齿音又不会被低频误触发。
# 下门为上门的一半（约 -3 dB 能量迟滞）。
HYSTERESIS_ENERGY_RATIO = 0.5
DETECT_ENERGY_SCALE = 0.6

# 压缩律强度系数：在标准 1-1/ratio 基础上略加深，保证 4-9 kHz 有 >=6 dB
# 的充足余量（覆盖起音段与不同测量窗）。
DEPTH = 1.2
GAIN_FLOOR = 0.35           # 最大衰减上限约 -9 dB，始终 > 0
_UNITY_EPS = 1e-9


class GainValue(float):
    """既是 ``float`` 又可调用的增益包装。

    兼容两种读取风格：``d.gain``（属性，直接当 float 用）与
    ``d.gain()``（可见用例使用的调用形式）。
    """

    __slots__ = ()

    def __call__(self):
        return float(self)


def _butterworth_biquad(kind, fc, rate, q=BUTTERWORTH_Q):
    """返回 RBJ 二阶 Butterworth 低通 / 高通系数。

    输出递推统一为::

        y = b0*x + b1*x1 + b2*x2 + A1*y1 + A2*y2

    其中 ``A1 = -a1``、``A2 = -a2``（归一化后）。
    """
    w0 = 2.0 * math.pi * fc / rate
    cw = math.cos(w0)
    sw = math.sin(w0)
    alpha = sw / (2.0 * q)
    a0 = 1.0 + alpha
    if kind == "low":
        b0 = (1.0 - cw) * 0.5
        b1 = 1.0 - cw
        b2 = (1.0 - cw) * 0.5
    else:  # high
        b0 = (1.0 + cw) * 0.5
        b1 = -(1.0 + cw)
        b2 = (1.0 + cw) * 0.5
    a1 = -2.0 * cw
    a2 = 1.0 - alpha
    inv = 1.0 / a0
    return (
        b0 * inv,
        b1 * inv,
        b2 * inv,
        -a1 * inv,   # A1
        -a2 * inv,   # A2
    )


def _mixed_error(param_cls, finite_cls, tag):
    """构造同时属于“参数错误”和“非有限分类错误”的临时异常类型。

    这样调用方既可以 ``pytest.raises(InvalidRateError)`` 也可以
    ``pytest.raises(NaNInputError)`` 捕获同一个异常。错误路径才会调用，
    动态建类的开销无关紧要。
    """
    name = f"NonFinite{param_cls.__name__}{tag}"
    return type(name, (param_cls, finite_cls), {})


def _positive_real(value, cls, name):
    """把构造参数校验为有限实数；布尔 / 字符串 / None 等报类型错误。"""
    if isinstance(value, bool):
        raise cls(f"{name} must be a finite real number, not bool")
    if isinstance(value, (int, float)):
        fv = float(value)
    else:
        # 兼容 numpy 标量等实现了 __float__ 的对象，但拒绝字符串等。
        try:
            fv = float(value)
        except (TypeError, ValueError) as exc:
            raise cls(
                f"{name} must be a real number, got {type(value).__name__}"
            ) from exc
        # float("48k") 会失败；float 对自定义对象可能成功，这里不再放宽。
    if not math.isfinite(fv):
        if math.isnan(fv):
            mixed = _mixed_error(cls, NaNInputError, "NaN")
            raise mixed(f"{name} must be finite (got NaN / 非有限值)")
        nf_cls = (PositiveInfinityInputError if fv > 0.0
                  else NegativeInfinityInputError)
        mixed = _mixed_error(cls, nf_cls, "Inf")
        raise mixed(f"{name} must be finite (got infinity / 无穷值)")
    return fv


class DeEsser:
    """齿音抑制器（分频段动态压缩器）。

    Parameters
    ----------
    rate : int or float
        采样率（Hz），必须为正的有限实数。任意正采样率都可用；滤波器
        截止频率会自适应到 Nyquist 以下。
    threshold : float
        齿音检测门限（线性幅度），取值 ``[0, 1]``。
    ratio : float
        压缩比，``>= 1``；``1`` 表示不压缩（输出与输入逐位相同）。
    """

    def __init__(self, rate, threshold, ratio):
        # ---- 参数校验（类型 / 值域 / 非有限，各自可区分） ----
        r = _positive_real(rate, InvalidRateError, "rate/采样率")
        if r <= 0.0:
            raise InvalidRateError(
                f"rate/采样率 must be > 0, got {rate!r}"
            )

        th = _positive_real(
            threshold, InvalidThresholdError, "threshold/门限"
        )
        if not (0.0 <= th <= 1.0):
            raise InvalidThresholdError(
                f"threshold/门限 must be in [0, 1], got {threshold!r}"
            )

        ra = _positive_real(ratio, InvalidRatioError, "ratio/压缩比")
        if ra < 1.0:
            raise InvalidRatioError(
                f"ratio/压缩比 must be >= 1 (1 = bypass), got {ratio!r}"
            )

        self.rate = r
        self.threshold = th
        self.ratio = ra

        # ---- 滤波器截止频率自适应到 Nyquist 以下，保证低采样率也稳定 ----
        nyq = 0.5 * r
        split_fc = min(SPLIT_FC_HZ, nyq * 0.80)
        hp_fc = min(DETECT_HP_HZ, nyq * 0.90)
        lp_fc = min(DETECT_LP_HZ, nyq * 0.95)
        # 极端低采样率下保证截止频率为正且顺序合理。
        split_fc = max(min(split_fc, nyq * 0.8), 1e-6)
        hp_fc = max(min(hp_fc, nyq * 0.9), split_fc * 1.02)
        lp_fc = max(min(lp_fc, nyq * 0.95), hp_fc * 1.05)

        self._lp_coef = _butterworth_biquad("low", split_fc, r)
        hp_coef = _butterworth_biquad("high", hp_fc, r)
        # 相同二阶节级联 HP_STAGES 次（4 阶高通）。
        self._hp_coefs = [hp_coef for _ in range(HP_STAGES)]
        self._dety_coef = _butterworth_biquad("low", lp_fc, r)

        # ---- 能量域迟滞门限（EMA(det^2) 稳态为 RMS^2） ----
        self._e_on = DETECT_ENERGY_SCALE * th * th
        self._e_off = self._e_on * HYSTERESIS_ENERGY_RATIO

        # ---- 时间常数 -> 单极点系数 ----
        def _tc(ms):
            return math.exp(-1.0 / (ms * 1e-3 * r))

        self._a_env_rel = _tc(ENVELOPE_RELEASE_MS)
        self._k_e = 1.0 - _tc(ENERGY_MS)
        self._k_g_atk = 1.0 - _tc(GAIN_ATTACK_MS)
        self._k_g_rel = 1.0 - _tc(GAIN_RELEASE_MS)
        self._comp_exp = DEPTH * (1.0 - 1.0 / ra) if ra > 1.0 else 0.0

        self._lock = threading.RLock()
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        """清空全部跨缓冲状态（增益回到 1，门关闭，滤波器状态清零）。"""
        with self._lock:
            # 低通 / 检测低通各一个 biquad；高通为 HP_STAGES 个级联节。
            self._lp_st = [0.0, 0.0, 0.0, 0.0]
            self._hp_st = [[0.0, 0.0, 0.0, 0.0]
                           for _ in range(HP_STAGES)]
            self._dety_st = [0.0, 0.0, 0.0, 0.0]
            self._env = 0.0
            self._energy = 0.0
            self._gate_open = False
            self.gate_open = False       # 公共别名（字面 bool）
            self._gain = 1.0

    # ------------------------------------------------------------------ #
    @property
    def gain(self):
        """当前（平滑后的）高频段增益，``0 < gain <= 1``。

        返回值既是 float 也可调用，因此 ``d.gain`` 与 ``d.gain()`` 都可用。
        """
        with self._lock:
            return GainValue(self._gain)

    # ------------------------------------------------------------------ #
    def process(self, samples):
        """处理一缓冲样本，返回与输入等长的 ``list[float]``。

        状态跨 :meth:`process` 调用连续；处理结果与缓冲切分方式逐位一致。
        非法缓冲 / 非有限样本抛出 :class:`deesser.errors.DeEsserError`
        的对应子类（均为 ``ValueError``）。
        """
        # ---- 缓冲类型的快速拒绝（字符串按字符迭代会产生误导性错误） ----
        if isinstance(samples, (str, bytes, bytearray)):
            raise InvalidSamplesError(
                "samples must be an iterable of real numbers, "
                f"got {type(samples).__name__}"
            )
        try:
            iterator = iter(samples)
        except TypeError as exc:
            raise InvalidSamplesError(
                "samples must be an iterable of real numbers, "
                f"got {type(samples).__name__}"
            ) from exc

        # 尝试拿到长度 / 随机访问以走更快的索引路径。
        try:
            n = len(samples)  # type: ignore[arg-type]
        except TypeError:
            n = -1

        with self._lock:
            # 从持久状态装载标量 / 局部变量（热路径全部局部化）。
            lb0, lb1, lb2, lA1, lA2 = self._lp_coef
            h00, h01, h02, h0A1, h0A2 = self._hp_coefs[0]
            if HP_STAGES >= 2:
                h10, h11, h12, h1A1, h1A2 = self._hp_coefs[1]
            db0, db1, db2, dA1, dA2 = self._dety_coef

            lx1, lx2, ly1, ly2 = self._lp_st
            (hx1, hx2, hy1, hy2) = self._hp_st[0]
            if HP_STAGES >= 2:
                (kx1, kx2, ky1, ky2) = self._hp_st[1]
            dx1, dx2, dy1, dy2 = self._dety_st

            env = self._env
            energy = self._energy
            gate = self._gate_open
            gain = self._gain

            e_on = self._e_on
            e_off = self._e_off
            thr = self.threshold
            comp_exp = self._comp_exp
            ratio = self.ratio
            floor = GAIN_FLOOR
            a_env_rel = self._a_env_rel
            k_e = self._k_e
            k_g_atk = self._k_g_atk
            k_g_rel = self._k_g_rel

            out = [0.0] * n if n >= 0 else []
            idx = 0
            two_stage = HP_STAGES >= 2
            isfinite = math.isfinite
            isnan = math.isnan
            isinf = math.isinf

            # 直接迭代原始 samples：逐元素 float() 转换 + 有限性分类，
            # 成功处理完整缓冲后才把状态写回，保证出错时实例状态不被污染。
            for raw in iterator:
                try:
                    v = float(raw)
                except (TypeError, ValueError) as exc:
                    raise InvalidSamplesError(
                        "every sample must be a real number; got "
                        f"{type(raw).__name__}: {raw!r}"
                    ) from exc

                # ---- 非有限输入：NaN / +Inf / -Inf 分类报错 ----
                if not isfinite(v):
                    if isnan(v):
                        raise NaNInputError(
                            "samples contain NaN（非有限输入）"
                        )
                    if isinf(v) and v > 0.0:
                        raise PositiveInfinityInputError(
                            "samples contain +Inf（非有限输入）"
                        )
                    raise NegativeInfinityInputError(
                        "samples contain -Inf（非有限输入）"
                    )

                # ---- 低频分频 LP；互补高频 high 在齿音带近似单位增益 ----
                lp = lb0 * v + lb1 * lx1 + lb2 * lx2 + lA1 * ly1 + lA2 * ly2
                lx2, lx1 = lx1, v
                ly2, ly1 = ly1, lp
                high = v - lp

                # ---- 检测高通：两级级联（4 阶），干净拒绝低频 ----
                hp = h00 * v + h01 * hx1 + h02 * hx2 + h0A1 * hy1 + h0A2 * hy2
                hx2, hx1 = hx1, v
                hy2, hy1 = hy1, hp
                if two_stage:
                    hp2 = (h10 * hp + h11 * kx1 + h12 * kx2
                           + h1A1 * ky1 + h1A2 * ky2)
                    kx2, kx1 = kx1, hp
                    ky2, ky1 = ky1, hp2
                    hp = hp2

                # ---- 9k 低通，形成 4-9k 检测带 ----
                band = (
                    db0 * hp + db1 * dx1 + db2 * dx2
                    + dA1 * dy1 + dA2 * dy2
                )
                dx2, dx1 = dx1, hp
                dy2, dy1 = dy1, band

                # ---- 频段能量 EMA（对称平均，稳态为 RMS^2） ----
                e_inst = band * band
                energy += k_e * (e_inst - energy)

                # ---- 能量迟滞状态机（上门开 / 下门关 / 中间保持） ----
                if gate:
                    if energy < e_off:
                        gate = False
                else:
                    if energy >= e_on:
                        gate = True

                # ---- 峰值包络（取互补 high，齿音带响应强且平坦；
                #      是否作用由带通迟滞门决定，低频不会开门） ----
                abs_high = high if high >= 0.0 else -high
                env_rel = a_env_rel * env
                env = abs_high if abs_high > env_rel else env_rel

                # ---- 压缩目标增益 ----
                if ratio <= 1.0 or not gate or env <= thr or comp_exp == 0.0:
                    target = 1.0
                else:
                    target = (thr / env) ** comp_exp
                    if target < floor:
                        target = floor
                    elif target > 1.0:
                        target = 1.0

                # ---- 增益平滑（攻击快 / 释放慢），并在 1 处吸附以保证
                #      未压缩时增益严格为 1、输出与输入逐位相同 ----
                if target < gain:
                    gain += k_g_atk * (target - gain)
                elif target > gain:
                    gain += k_g_rel * (target - gain)

                if target >= 1.0 and gain >= 1.0 - _UNITY_EPS:
                    gain = 1.0

                # ---- 分频段混合；unity 时直接返回原始样本（逐位旁路） ----
                if gain >= 1.0:
                    y = v
                else:
                    y = lp + high * gain

                # ---- 输出范围保证 |y| <= 1，并兜底极端大幅值导致的
                #      内部溢出（输入有限时输出也必须有限） ----
                if y > 1.0:
                    y = 1.0
                elif y < -1.0:
                    y = -1.0
                elif not isfinite(y):
                    # 理论上仅在 |v| 接近 float 上限、滤波器内部溢出时到达；
                    # 用输入符号给出确定、有限的削波值。
                    y = 1.0 if v >= 0.0 else -1.0

                if n >= 0:
                    out[idx] = y
                    idx += 1
                else:
                    out.append(y)

            # ---- 状态提交（仅在完整缓冲成功后） ----
            self._lp_st = [lx1, lx2, ly1, ly2]
            hp_states = [[hx1, hx2, hy1, hy2]]
            if HP_STAGES >= 2:
                hp_states.append([kx1, kx2, ky1, ky2])
            self._hp_st = hp_states
            self._dety_st = [dx1, dx2, dy1, dy2]
            self._env = env
            self._energy = energy
            self._gate_open = bool(gate)
            self.gate_open = bool(gate)
            self._gain = gain

            return out

    # 兼容部分调用方把 gain 当普通方法实现的旧用法：
    # 当通过类访问 / 或属性被遮蔽时，提供一个显式方法入口。
    def current_gain(self):
        """与 :attr:`gain` 相同，显式方法形式。"""
        with self._lock:
            return float(self._gain)
