"""brdf 核心 API（初始未实现）。"""


class Ggx:
    def eval(self, n_dot_v, n_dot_l, roughness, f0):
        """BRDF 值"""
        raise NotImplementedError

    def pdf(self, n_dot_h, roughness):
        """概率密度"""
        raise NotImplementedError
