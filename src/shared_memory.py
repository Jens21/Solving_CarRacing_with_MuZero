import threading

class SharedMemory():
    _policy_net_dict = None
    _target_net_dict = None

    def __init__(self):
        _lock = threading.Lock()

    def set_nets_dict(self, policy_net_dict, target_net_dict):
        self._lock.acquire()
        self._policy_net_dict = policy_net_dict
        self._target_net_dict = target_net_dict
        self._lock.release()

    def get_nets_dict(self):
        self._lock.acquire()
        pol, tar = self._policy_net_dict, self._target_net_dict
        self._lock.release()

        return pol, tar