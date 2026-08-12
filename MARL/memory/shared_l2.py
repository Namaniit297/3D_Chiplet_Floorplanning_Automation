import threading
from collections import OrderedDict


class SharedL2Cache:
    """
    Shared L2 cache simulation for multi-agent experience sharing.
    Thread-safe LRU cache of recent transitions.
    """

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def get_all(self):
        with self.lock:
            return list(self.cache.values())

    def __len__(self):
        return len(self.cache)
