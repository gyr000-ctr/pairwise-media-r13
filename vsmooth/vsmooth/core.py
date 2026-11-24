"""vsmooth 核心 API（初始未实现）。"""


class Smooth:
    def __init__(self, lam=0.5, mu=-0.53):
        """构造"""
        raise NotImplementedError

    def smooth(self, verts, faces, iters):
        """平滑"""
        raise NotImplementedError

    def volume(self, verts, faces):
        """体积"""
        raise NotImplementedError
