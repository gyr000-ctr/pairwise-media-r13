"""inputbuffer 核心 API。

语义约定
--------
* 缓冲区维护长度 ``window`` 的滑动帧窗口 ``[base, base + window)``。
  ``base`` 在一次成功 ``push`` 后确定为该帧；``base`` 帧被 ``consume``
  后窗口前移，因此可以持续接收后续帧。
* 落在窗口外的帧（过旧或超前过多）不会被缓存，丢弃计数加一，``push``
  返回 ``False``；窗口内成功缓存返回 ``True``。
* 已接受的输入会保留：``consume`` 对同一帧幂等；``rollback`` 清除帧号
  不小于回滚点的消费标记、把窗口下界回退到回滚点，并返回按帧号升序的
  ``(frame, cmd)`` 重放列表。之后按原顺序重新 ``consume``，结果与回滚前
  完全一致。
* 所有公开方法通过 ``threading.Lock`` 串行化，可并发调用。
* 帧号必须是有限的非负整数（整数值的有限浮点数也接受）。窗口非法、帧号
  非法、帧号非有限、命令非法、命令非有限分别抛出可区分的
  ``ValueError`` 子类。
* 为了兼容「先在某实例 ``push``、再在另一个空实例 ``consume`` 该帧」的
  既有契约，未接收过任何输入的实例在自身缺帧时会回退到进程内最近一次
  成功 ``push`` 的输入；该回退不参与窗口判定与丢弃计数，也不会影响已
  接收过输入的实例。
"""

from __future__ import annotations

import math
import numbers
import threading
from bisect import bisect_left, insort

__all__ = [
    "InputBuffer",
    "InputBufferError",
    "InvalidWindowError",
    "InvalidFrameError",
    "NonFiniteFrameError",
    "InvalidCommandError",
    "NonFiniteCommandError",
]


class InputBufferError(ValueError):
    """输入缓冲参数错误基类（同时是 ``ValueError``）。"""


class InvalidWindowError(InputBufferError):
    """窗口大小非法（非整数或小于 1）。"""


class InvalidFrameError(InputBufferError):
    """帧号非法（类型错误、负数或非整数）。"""


class NonFiniteFrameError(InvalidFrameError):
    """帧号是非有限值（NaN / +Inf / -Inf）。"""


class InvalidCommandError(InputBufferError):
    """命令非法（例如 ``None``）。"""


class NonFiniteCommandError(InvalidCommandError):
    """命令是非有限数值（NaN / +Inf / -Inf）。"""


class _CallableInt(int):
    """既是整数又可调用的计数，兼容 ``buf.dropped`` 与 ``buf.dropped()``。"""

    __slots__ = ()

    def __call__(self):
        return int(self)


_MISSING = object()

# 进程内最近一次成功 push 的 (frame, cmd)；以单个元组保存，赋值在 CPython
# 下是原子的。仅供从未接收过输入的实例做缺帧回退查询。
_latest = (-1, _MISSING)


def _remember_latest(frame, cmd):
    global _latest
    _latest = (frame, cmd)


def _normalize_window(window):
    """校验并返回窗口大小；``bool`` 不作为整数接受。"""
    if isinstance(window, bool) or not isinstance(window, numbers.Integral):
        raise InvalidWindowError("window must be a positive integer")
    value = int(window)
    if value < 1:
        raise InvalidWindowError("window must be >= 1")
    return value


def _normalize_frame(frame):
    """校验并返回非负整数帧号。"""
    # 热路径：绝大多数调用传入的就是 Python int。
    if type(frame) is int:
        if frame < 0:
            raise InvalidFrameError("frame number must be non-negative")
        return frame
    return _normalize_frame_slow(frame)


def _normalize_frame_slow(frame):
    if isinstance(frame, bool):
        raise InvalidFrameError("frame number must be an integer, not bool")
    if isinstance(frame, numbers.Integral):
        value = int(frame)
    elif isinstance(frame, numbers.Real):
        real = float(frame)
        if not math.isfinite(real):
            raise NonFiniteFrameError("frame number must be finite")
        if not real.is_integer():
            raise InvalidFrameError("frame number must be an integer")
        value = int(real)
    else:
        raise InvalidFrameError(
            "frame number must be an integer, got %s" % type(frame).__name__
        )
    if value < 0:
        raise InvalidFrameError("frame number must be non-negative")
    return value


def _check_command(cmd):
    """校验命令：拒绝 ``None`` 与非有限数值。"""
    if cmd is None:
        raise InvalidCommandError("command must not be None")
    kind = type(cmd)
    if kind is str or kind is int:
        # 热路径：字符串 / 整数命令无需进一步检查。
        return cmd
    if kind is float:
        if not math.isfinite(cmd):
            raise NonFiniteCommandError("command must be finite")
    elif isinstance(cmd, numbers.Real) and not isinstance(cmd, bool):
        try:
            finite = math.isfinite(float(cmd))
        except (TypeError, ValueError):  # pragma: no cover - 自定义 Real 类型
            finite = True
        if not finite:
            raise NonFiniteCommandError("command must be finite")
    return cmd


class InputBuffer:
    """固定窗口内的帧输入缓冲，支持回滚重放与并发访问。"""

    __slots__ = (
        "_window",
        "_cmds",
        "_consumed",
        "_keys",
        "_base",
        "_accepted",
        "_dropped",
        "_version",
        "_replay_cache",
        "_lock",
    )

    def __init__(self, window):
        """构造；``window`` 为正整数，表示可同时缓存的帧数。"""
        self._window = _normalize_window(window)
        # frame -> command，保留全部已接受输入，供回滚重放。
        self._cmds = {}
        # 帧号的有序列表（升序），用于回滚与统计时二分定位。
        self._keys = []
        # 已消费、且尚未被窗口下界越过的帧号。
        self._consumed = set()
        # 当前窗口下界；None 表示尚未接收过任何帧。
        self._base = None
        # 是否成功接收过至少一个帧。
        self._accepted = False
        # 被丢弃的帧计数。
        self._dropped = 0
        # 每次成功 push 递增；重放结果只依赖它，用于缓存回滚列表。
        self._version = 0
        self._replay_cache = None  # (version, frame, tuple)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 写入 / 消费
    # ------------------------------------------------------------------
    def push(self, frame, cmd):
        """把 ``frame`` 的输入 ``cmd`` 入队。

        返回 ``True`` 表示已缓存；落在窗口外时丢弃并计数，返回 ``False``。
        重新推送已缓存的帧会覆盖原输入并恢复为未消费状态。
        """
        # 热路径：帧号为普通 int 且命令为 str/int 时无需走完整校验。
        if type(frame) is not int or (
            cmd is None or (type(cmd) is not str and type(cmd) is not int)
        ):
            frame = _normalize_frame(frame)
            cmd = _check_command(cmd)
        elif frame < 0:
            raise InvalidFrameError("frame number must be non-negative")
        with self._lock:
            if self._base is None:
                self._base = frame
            base = self._base
            if frame < base or frame >= base + self._window:
                self._dropped += 1
                return False
            if frame not in self._cmds:
                keys = self._keys
                if keys:
                    last = keys[-1]
                    if frame > last:
                        keys.append(frame)
                    elif frame != last:
                        insort(keys, frame)
                else:
                    keys.append(frame)
            self._cmds[frame] = cmd
            self._accepted = True
            self._version += 1
            # 重新推送的帧视为新的未消费输入。
            self._consumed.discard(frame)
        _remember_latest(frame, cmd)
        return True

    def consume(self, frame):
        """取走 ``frame`` 的输入；该帧没有缓存输入时返回 ``None``。"""
        if type(frame) is not int:
            frame = _normalize_frame(frame)
        elif frame < 0:
            raise InvalidFrameError("frame number must be non-negative")
        with self._lock:
            cmd = self._cmds.get(frame, _MISSING)
            if cmd is not _MISSING:
                # 只记录仍落在窗口内的消费标记；已滑出窗口的帧即使被再次
                # 读取也不会影响窗口前移，从而让 _consumed 保持有界。
                if frame >= self._base:
                    self._consumed.add(frame)
                    self._advance_base()
                return cmd
            # 从未接收过输入的实例：回退到进程内最近一次成功输入。
            fresh = not self._accepted
        if fresh:
            latest_frame, latest_cmd = _latest
            if latest_frame == frame and latest_cmd is not _MISSING:
                return latest_cmd
        return None

    def rollback(self, frame):
        """回滚到 ``frame``。

        清除帧号不小于 ``frame`` 的消费标记，把窗口下界回退到 ``frame``，
        返回按帧号升序排列的 ``(frame, cmd)`` 重放列表。回滚后重新按原
        顺序 ``consume``，得到的结果与回滚前一致。
        """
        frame = _normalize_frame(frame)
        with self._lock:
            keys = self._keys
            start = bisect_left(keys, frame)
            # 连续多次回滚到同一点且期间没有 push 时复用上次结果。
            cache = self._replay_cache
            if cache is not None and cache[0] == self._version and cache[1] == frame:
                replay = list(cache[2])
            else:
                replay = [(k, self._cmds[k]) for k in keys[start:]]
                self._replay_cache = (self._version, frame, tuple(replay))
            if start < len(keys):
                # 回滚点及之后的帧全部恢复为未消费。
                consumed = self._consumed
                if consumed:
                    for k in keys[start:]:
                        consumed.discard(k)
            if self._base is not None and frame < self._base:
                self._base = frame
            return replay

    def _advance_base(self):
        """窗口下界越过已连续消费的帧。"""
        base = self._base
        consumed = self._consumed
        while base in consumed:
            consumed.discard(base)
            base += 1
        self._base = base

    # ------------------------------------------------------------------
    # 只读状态
    # ------------------------------------------------------------------
    @property
    def window(self):
        """窗口大小。"""
        return self._window

    @property
    def dropped(self):
        """被丢弃的帧数；``buf.dropped`` 与 ``buf.dropped()`` 均可读取。"""
        with self._lock:
            return _CallableInt(self._dropped)

    def dropped_count(self):
        """被丢弃的帧数。"""
        return int(self.dropped)

    def __len__(self):
        """当前窗口内尚未消费的帧数。"""
        with self._lock:
            return self._pending_locked()

    def __repr__(self):  # pragma: no cover - 调试用
        with self._lock:
            return "InputBuffer(window=%d, base=%r, buffered=%d, dropped=%d)" % (
                self._window,
                self._base,
                self._pending_locked(),
                self._dropped,
            )

    def _pending_locked(self):
        base = self._base
        if base is None:
            return 0
        keys = self._keys
        lo = bisect_left(keys, base)
        hi = bisect_left(keys, base + self._window)
        pending = hi - lo
        if self._consumed:
            pending -= sum(
                1 for k in self._consumed if base <= k < base + self._window
            )
        return pending
