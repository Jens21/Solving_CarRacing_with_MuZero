import threading

class SharedMemory():
    _net_dict = None
    _lock = threading.Lock()

    def set_net_dict(self, net_dict : dict):
        self._lock.acquire()
        self._net_dict = net_dict
        self._lock.release()

    def get_net_dict(self):
        self._lock.acquire()
        net = self._net_dict
        self._lock.release()

        return net