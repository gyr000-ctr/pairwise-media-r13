"""deesser 核心 API（初始未实现）。"""


class DeEsser:
    def __init__(self, rate, threshold, ratio):
        """构造"""
        raise NotImplementedError

    def process(self, samples):
        """处理"""
        raise NotImplementedError

    def gain(self):
        """当前增益"""
        raise NotImplementedError
