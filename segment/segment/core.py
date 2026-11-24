"""segment 核心 API：容差洪水填充分割。

语义约定
--------
- 图像是二维标量（灰度 / 单通道）数组，行优先，矩形（不允许锯齿）。
- 容差按颜色距离判定：像素值 ``v`` 与种子值 ``s`` 的距离为 ``abs(v - s)``。
- 满足 ``abs(v - s) < tol`` 的像素才属于可填充区域（严格小于）；种子像素
  无论容差如何都视为已填充。因此 ``tol == 0`` 时结果只有种子像素本身。
- 填充结果是布尔掩码，``True`` 表示与种子 4-邻接且颜色距离在容差内的连通
  区域；与独立的 BFS 参考实现逐像素一致。
- ``fill`` / ``count`` 不修改入参，实例无可变状态，可跨线程并发调用。
- 性能：100 万像素（1000x1000）的单次 ``fill`` 在纯 Python 下远低于 500 ms，
  使用逐行扫描线（span）洪水填充，避免逐像素递归/队列带来的常数开销。
"""

import math

__all__ = [
    "FloodFill",
    "SegmentError",
    "InvalidArgument",
    "InvalidTolerance",
    "NonFiniteInput",
    "SeedOutOfRange",
]


class SegmentError(ValueError):
    """本模块所有业务错误的基类（兼容 ValueError）。"""


class InvalidArgument(TypeError, SegmentError):
    """参数类型错误（同时是 TypeError 与 ValueError）。"""


class NonFiniteInput(SegmentError):
    """图像中含有 NaN / ±inf 等非有限像素值。"""


class InvalidTolerance(SegmentError):
    """容差非法（负数、NaN、±inf）。"""


class SeedOutOfRange(SegmentError):
    """种子坐标越界或不是合法的二维整数坐标。"""


def _as_float_rows(image):
    """把二维输入规整为矩形浮点行列表；类型/结构错误抛异常。"""
    if isinstance(image, (str, bytes, bytearray, dict, set, frozenset)):
        raise InvalidArgument("image must be a 2-D sequence of numbers")
    try:
        rows = list(image)
    except TypeError as exc:
        raise InvalidArgument("image must be a 2-D sequence of numbers") from exc
    if not rows:
        raise SegmentError("image is empty")
    try:
        width = len(rows[0])
    except TypeError as exc:
        raise InvalidArgument("image must be 2-D (rows must be sequences)") from exc
    if width == 0:
        raise SegmentError("image rows are empty")
    height = len(rows)
    for row in rows:
        if isinstance(row, (str, bytes, bytearray, dict, set, frozenset)):
            raise InvalidArgument("image rows must be numeric sequences")
        try:
            n = len(row)
        except TypeError as exc:
            raise InvalidArgument("image rows must be sequences") from exc
        if n != width:
            raise SegmentError("image is ragged: all rows must have the same length")
    try:
        flat = [float(v) for row in rows for v in row]
    except (TypeError, ValueError) as exc:
        raise InvalidArgument("image pixels must be real numbers") from exc
    if not all(map(math.isfinite, flat)):
        raise NonFiniteInput("image contains non-finite pixel values")
    return flat, height, width


def _check_tol(tol):
    """校验并归一化容差。"""
    if isinstance(tol, (bool, str, bytes, bytearray)):
        raise InvalidArgument("tolerance must be a real number")
    try:
        tol = float(tol)
    except (TypeError, ValueError) as exc:
        raise InvalidArgument("tolerance must be a real number") from exc
    if not math.isfinite(tol):
        raise InvalidTolerance("tolerance must be finite")
    if tol < 0.0:
        raise InvalidTolerance("tolerance must be non-negative")
    return tol


def _check_seed(seed, height, width):
    """返回规范化的 (row, col)；坐标越界才是 SeedOutOfRange。

    形状/类型错误按更一般的参数错误分类，与越界错误区分得开。
    """
    if isinstance(seed, (str, bytes, bytearray)):
        raise InvalidArgument("seed must be a (row, col) pair of integers")
    try:
        pair = tuple(seed)
    except TypeError as exc:
        raise InvalidArgument("seed must be a (row, col) pair of integers") from exc
    if len(pair) != 2:
        raise InvalidArgument("seed must be a (row, col) pair of integers")
    row, col = pair
    for value in (row, col):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidArgument("seed coordinates must be integers")
    if not (0 <= row < height and 0 <= col < width):
        raise SeedOutOfRange(
            "seed (%d, %d) is outside image of shape (%d, %d)"
            % (row, col, height, width)
        )
    return row, col


def _span_fill(mask, start, width, total):
    """扫描线洪水填充。

    ``mask`` 取值：0 = 候选（可填充），2 = 障碍。填充到的格子写 1。
    函数原地修改 mask，返回后 1 即填充结果。

    每个待处理行段只扫描一次：行段在栈上以单个像素为种子记录，弹出后向
    左右扩展到完整的连续行段，再只在上下两行中寻找新的行段。总体复杂度
    O(像素数)，常数开销主要来自逐行切片，远小于逐像素 Python 循环。
    """
    stack = [start]
    pop = stack.pop
    push = stack.append
    while stack:
        i = pop()
        if mask[i] != 0:
            # 入栈后已被同一行段的其它种子覆盖。
            continue
        row_start = i - i % width
        row_end = row_start + width
        left = i
        while left > row_start and mask[left - 1] == 0:
            left -= 1
        right = i
        while right + 1 < row_end and mask[right + 1] == 0:
            right += 1
        mask[left:right + 1] = b"\x01" * (right - left + 1)
        lo = left - row_start
        hi = right - row_start + 1
        if row_start:
            base = row_start - width
            j = base + lo
            limit = base + hi
            while j < limit:
                if mask[j] == 0:
                    push(j)
                    j += 1
                    while j < limit and mask[j] == 0:
                        j += 1
                else:
                    j += 1
        if row_end < total:
            j = row_end + lo
            limit = row_end + hi
            while j < limit:
                if mask[j] == 0:
                    push(j)
                    j += 1
                    while j < limit and mask[j] == 0:
                        j += 1
                else:
                    j += 1


class FloodFill:
    """容差洪水填充。无实例状态，可在多线程间共享。"""

    __slots__ = ()

    def fill(self, image, seed, tol):
        """返回与种子连通的容差区域掩码（list[list[bool]]）。

        容差为颜色距离上界：``abs(pixel - seed_pixel) < tol``，种子像素始终
        属于结果。``image`` 为矩形二维数值数组，不要求 float。
        """
        flat, height, width = _as_float_rows(image)
        row, col = _check_seed(seed, height, width)
        tol = _check_tol(tol)

        start = row * width + col
        base_value = flat[start]
        mask = bytearray([0 if abs(v - base_value) < tol else 2 for v in flat])
        # 种子像素始终属于结果（tol == 0 时严格比较不成立，需要显式加入）。
        mask[start] = 0
        _span_fill(mask, start, width, len(flat))

        return [[v == 1 for v in mask[offset:offset + width]]
                for offset in range(0, len(mask), width)]

    def count(self, mask):
        """返回掩码中 True 的像素个数（面积）。"""
        if isinstance(mask, (str, bytes, bytearray, dict, set, frozenset)):
            raise InvalidArgument("mask must be a 2-D sequence of booleans")
        try:
            rows = list(mask)
        except TypeError as exc:
            raise InvalidArgument("mask must be a 2-D sequence of booleans") from exc
        total = 0
        width = None
        for row in rows:
            if isinstance(row, (str, bytes, bytearray, dict, set, frozenset)):
                raise InvalidArgument("mask rows must be boolean sequences")
            try:
                values = row if isinstance(row, list) else list(row)
            except TypeError as exc:
                raise InvalidArgument("mask rows must be boolean sequences") from exc
            if width is None:
                width = len(values)
            elif len(values) != width:
                raise SegmentError("mask is ragged: all rows must have the same length")
            for value in values:
                if value is True:
                    total += 1
                elif value is False:
                    continue
                elif value == 0:
                    continue
                elif value == 1:
                    total += 1
                else:
                    raise InvalidArgument("mask values must be boolean (0/1)")
        return total
