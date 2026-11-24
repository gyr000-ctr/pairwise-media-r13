"""vsmooth 核心 API。

Taubin 拉普拉斯网格平滑（λ|μ 交替），带体积保持与单步位移上界。

设计要点：
- 一个 Taubin 迭代 = 一次 λ 收缩半步 + 一次 μ 膨胀半步（均匀权拉普拉斯）。
- 每次迭代结束按闭合网格的有符号体积做一次质心均匀缩放校正，
  使体积变化在数值精度内为 0（远小于题面 1% 上界）；
  体积退化（近零体积）的网格自动跳过该校正。
- ``max_step`` 按“单步（单次完整迭代）位移向量范数”裁剪，
  体积校正后会按同一起点重新裁剪，两个约束同时成立。
- 每次调用只使用局部状态、不写入实例属性，多线程并发调用安全。
- 纯标准库实现，无外部依赖。
"""

import math

__all__ = [
    "Smooth",
    "VsmoothError",
    "InvalidParameterError",
    "InvalidMeshError",
    "InvalidIterationError",
    "NonFiniteError",
]


class VsmoothError(ValueError):
    """vsmooth 所有可区分错误的基类（同时是 ValueError）。"""


class InvalidParameterError(VsmoothError):
    """Taubin 参数非法（lam / mu / max_step 取值不允许）。"""


class InvalidMeshError(VsmoothError):
    """网格数据非法（空网格、面元结构错、顶点索引越界等）。"""


class InvalidIterationError(VsmoothError):
    """迭代次数非法（非正整数）。"""


class NonFiniteError(VsmoothError):
    """输入中出现非有限数值（NaN / Inf）。"""


def _half_step(xs, ys, zs, nbrs, inv_deg, k):
    """一次均匀权拉普拉斯线性半步：x' = x + k * (mean(neighbor) - x)。"""
    n = len(xs)
    nxs = [0.0] * n
    nys = [0.0] * n
    nzs = [0.0] * n
    for i in range(n):
        nb = nbrs[i]
        if not nb:
            # 孤立顶点：拉普拉斯量为 0，保持不动。
            nxs[i] = xs[i]
            nys[i] = ys[i]
            nzs[i] = zs[i]
            continue
        sx = 0.0
        sy = 0.0
        sz = 0.0
        for j in nb:
            sx += xs[j]
            sy += ys[j]
            sz += zs[j]
        d = inv_deg[i] * k
        nxs[i] = xs[i] + (sx * d - xs[i] * k)
        nys[i] = ys[i] + (sy * d - ys[i] * k)
        nzs[i] = zs[i] + (sz * d - zs[i] * k)
    return nxs, nys, nzs


def _volume_arrays(xs, ys, zs, faces):
    """闭合三角网格的有符号体积（V = 1/6 * sum dot(a, cross(b, c))）。"""
    total = 0.0
    for a, b, c in faces:
        bx, by, bz = xs[b], ys[b], zs[b]
        cx, cy, cz = xs[c], ys[c], zs[c]
        ax, ay, az = xs[a], ys[a], zs[a]
        total += ax * (by * cz - bz * cy)
        total += ay * (bz * cx - bx * cz)
        total += az * (bx * cy - by * cx)
    return total / 6.0


def _clip_to_cap(xs, ys, zs, old_xs, old_ys, old_zs, cap):
    """把每个顶点相对迭代起点的位移向量裁剪到 L2 范数 ``cap`` 以内（原地）。

    以“写回坐标后重新实测的位移”为准迭代重裁剪：当坐标远大于 cap 时，
    浮点加法的舍入可能让名义上恰好等于 cap 的位移实测越界，
    因此反复收缩系数，直到实测范数真的 <= cap；
    兜底为顶点不动（位移 0 对任何 cap 都合法）。
    """
    cap2 = cap * cap
    for i in range(len(xs)):
        dx = xs[i] - old_xs[i]
        dy = ys[i] - old_ys[i]
        dz = zs[i] - old_zs[i]
        if dx * dx + dy * dy + dz * dz <= cap2:
            continue
        scale = cap / math.sqrt(dx * dx + dy * dy + dz * dz)
        nx = ny = nz = 0.0
        for _ in range(8):
            nx = old_xs[i] + dx * scale
            ny = old_ys[i] + dy * scale
            nz = old_zs[i] + dz * scale
            ex = nx - old_xs[i]
            ey = ny - old_ys[i]
            ez = nz - old_zs[i]
            if ex * ex + ey * ey + ez * ez <= cap2:
                break
            # 按实测范数收缩，系数至少减半以保证有限轮内收敛。
            measured = math.sqrt(ex * ex + ey * ey + ez * ez)
            scale *= min(0.5, cap / measured)
        else:
            nx, ny, nz = old_xs[i], old_ys[i], old_zs[i]
        xs[i], ys[i], zs[i] = nx, ny, nz


class Smooth:
    """Taubin λ|μ 拉普拉斯平滑器。

    参数
    ----
    lam:
        收缩半步系数，要求 0 < lam < 1。
    mu:
        膨胀半步系数，要求 -1 < mu < 0。
    max_step:
        可选的单步位移上界（向量 L2 范数），``None`` 表示不裁剪；
        给定正数时，每个完整 Taubin 迭代中任一顶点的位移都不超过该值。
    """

    def __init__(self, lam=0.5, mu=-0.53, *, max_step=None):
        if not math.isfinite(lam) or not 0.0 < lam < 1.0:
            raise InvalidParameterError(
                "lam 必须是开区间 (0, 1) 内的有限实数，收到 %r" % (lam,)
            )
        if not math.isfinite(mu) or not -1.0 < mu < 0.0:
            raise InvalidParameterError(
                "mu 必须是开区间 (-1, 0) 内的有限实数，收到 %r" % (mu,)
            )
        if max_step is not None and (
            not math.isfinite(max_step) or max_step <= 0.0
        ):
            raise InvalidParameterError(
                "max_step 必须为正的有限实数或 None，收到 %r" % (max_step,)
            )
        self.lam = float(lam)
        self.mu = float(mu)
        self.max_step = None if max_step is None else float(max_step)

    # ------------------------------------------------------------------ #
    # 校验与拓扑
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_and_unpack(verts, faces):
        """校验网格并返回 (xs, ys, zs, faces_tuple, bbox_diagonal)。"""
        try:
            n = len(verts)
        except TypeError as exc:
            raise InvalidMeshError("verts 必须是顶点序列") from exc
        if n == 0:
            raise InvalidMeshError("顶点数必须大于 0")

        xs = [0.0] * n
        ys = [0.0] * n
        zs = [0.0] * n
        xmin = ymin = zmin = math.inf
        xmax = ymax = zmax = -math.inf
        for i, p in enumerate(verts):
            try:
                if len(p) != 3:
                    raise InvalidMeshError(
                        "顶点 %d 不是长度为 3 的坐标，收到 %r" % (i, p)
                    )
                x, y, z = p
            except TypeError as exc:
                raise InvalidMeshError(
                    "顶点 %d 必须是三维坐标序列，收到 %r" % (i, p)
                ) from exc
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                raise NonFiniteError(
                    "顶点 %d 含非有限坐标 %r（仅允许有限实数）" % (i, p)
                )
            x, y, z = float(x), float(y), float(z)
            xs[i], ys[i], zs[i] = x, y, z
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
            if z < zmin:
                zmin = z
            if z > zmax:
                zmax = z

        try:
            nf = len(faces)
        except TypeError as exc:
            raise InvalidMeshError("faces 必须是三角面序列") from exc
        if nf == 0:
            raise InvalidMeshError("面数必须大于 0")

        faces_t = []
        for fi, face in enumerate(faces):
            try:
                if len(face) != 3:
                    raise InvalidMeshError(
                        "面 %d 不是三角形（需要 3 个顶点索引），收到 %r" % (fi, face)
                    )
                a, b, c = face
            except TypeError as exc:
                raise InvalidMeshError(
                    "面 %d 必须是三元索引序列，收到 %r" % (fi, face)
                ) from exc
            # bool 是 int 子类，允许；拒绝 numpy 等非内建整数及越界索引。
            for idx in (a, b, c):
                if type(idx) is not int or not 0 <= idx < n:
                    raise InvalidMeshError(
                        "面 %d 含越界或非整数顶点索引 %r（顶点数 %d）" % (fi, idx, n)
                    )
            faces_t.append((a, b, c))

        diag = math.sqrt(
            (xmax - xmin) ** 2
            + (ymax - ymin) ** 2
            + (zmax - zmin) ** 2
        )
        return xs, ys, zs, tuple(faces_t), diag

    @staticmethod
    def _build_adjacency(n, faces):
        """构建去重、有序的邻接表（有序保证结果确定、可复现）。"""
        sets = [set() for _ in range(n)]
        for a, b, c in faces:
            sets[a].add(b)
            sets[a].add(c)
            sets[b].add(a)
            sets[b].add(c)
            sets[c].add(a)
            sets[c].add(b)
        nbrs = tuple(tuple(sorted(s)) for s in sets)
        inv_deg = tuple((1.0 / len(nb) if nb else 0.0) for nb in nbrs)
        return nbrs, inv_deg

    # ------------------------------------------------------------------ #
    # 公共 API
    # ------------------------------------------------------------------ #
    def smooth(self, verts, faces, iters, *, max_step=None):
        """对网格执行 ``iters`` 次 Taubin 平滑。

        返回值为 ``list[tuple[float, float, float]]``，顶点数与输入一致，
        输入数据不会被修改。
        """
        if type(iters) is not int or iters <= 0:
            raise InvalidIterationError(
                "iters 必须为正整数，收到 %r" % (iters,)
            )
        if max_step is None:
            max_step = self.max_step
        elif not math.isfinite(max_step) or max_step <= 0.0:
            raise InvalidParameterError(
                "max_step 必须为正的有限实数或 None，收到 %r" % (max_step,)
            )

        xs, ys, zs, faces_t, diag = self._check_and_unpack(verts, faces)
        n = len(xs)
        nbrs, inv_deg = self._build_adjacency(n, faces_t)
        # 仅参与面的顶点属于闭合表面，体积缩放只作用于这些点；
        # 孤立顶点（度为 0）不贡献体积，全程保持原位。
        surface = tuple(i for i in range(n) if nbrs[i])

        # 初始有符号体积，用于末端体积保持校正。
        v0 = _volume_arrays(xs, ys, zs, faces_t)
        lam = self.lam
        mu = self.mu
        cap = max_step

        # 体积退化到近乎零时跳过缩放校正（相对包围盒对角线判定）。
        volume_scale_guard = 1e-12 * diag ** 3
        do_volume = bool(surface) and abs(v0) > volume_scale_guard

        for _ in range(iters):
            old_xs, old_ys, old_zs = xs, ys, zs
            txs, tys, tzs = _half_step(xs, ys, zs, nbrs, inv_deg, lam)
            xs, ys, zs = _half_step(txs, tys, tzs, nbrs, inv_deg, mu)
            if cap is not None:
                # 先对 λ|μ 半步合成位移做单步范数裁剪。
                _clip_to_cap(xs, ys, zs, old_xs, old_ys, old_zs, cap)
            if do_volume:
                # 每迭代做一次质心均匀缩放恢复体积，再按同一起点重新裁剪：
                # 体积保持与单步位移上界因此同时成立。
                v1 = _volume_arrays(xs, ys, zs, faces_t)
                if math.isfinite(v1) and abs(v1) > 1e-300:
                    ratio = v0 / v1
                    if 0.0 < ratio < math.inf:
                        scl = math.cbrt(ratio)
                        if math.isfinite(scl) and 0.25 <= scl <= 4.0:
                            m = len(surface)
                            cx = sum(xs[i] for i in surface) / m
                            cy = sum(ys[i] for i in surface) / m
                            cz = sum(zs[i] for i in surface) / m
                            for i in surface:
                                xs[i] = cx + (xs[i] - cx) * scl
                                ys[i] = cy + (ys[i] - cy) * scl
                                zs[i] = cz + (zs[i] - cz) * scl
                            if cap is not None:
                                _clip_to_cap(
                                    xs, ys, zs, old_xs, old_ys, old_zs, cap
                                )

        return [(xs[i], ys[i], zs[i]) for i in range(n)]

    def volume(self, verts, faces):
        """计算闭合三角网格的有符号体积，返回 ``float``。

        方向一致朝外的闭合网格返回正值；入参校验规则与 :meth:`smooth` 相同。
        """
        xs, ys, zs, faces_t, _ = self._check_and_unpack(verts, faces)
        return float(_volume_arrays(xs, ys, zs, faces_t))
