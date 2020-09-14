import wrapt


class DataBuffer(object):
    """
    Limited capacity synchonized data buffer
    """
    def __init__(self, capacity=1, dataclass=None):
        assert capacity > 0
        self.capacity = capacity
        self.dataclass = dataclass
        self.buffer = []

    @wrapt.synchronized
    def add(self, data):
        if self.dataclass is not None:
            assert isinstance(data, self.dataclass)
        self.buffer.append(data)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    @wrapt.synchronized
    def get_latest(self):
        return self.buffer[-1]

    @wrapt.synchronized
    def is_empty(self):
        return len(self.buffer) == 0



