"""inputbuffer 核心 API（初始未实现）。"""


class InputBuffer:
    def __init__(self, window):
        """构造"""
        raise NotImplementedError

    def push(self, frame, cmd):
        """入队"""
        raise NotImplementedError

    def consume(self, frame):
        """取该帧输入"""
        raise NotImplementedError

    def rollback(self, frame):
        """回滚"""
        raise NotImplementedError
