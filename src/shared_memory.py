import threading
import torch.multiprocessing

class SharedMemory():
    _net_dict = torch.multiprocessing.Manager().dict()
    _lock = torch.multiprocessing.Lock()

    def set_net_dict(self, net_dict : dict):
        self._lock.acquire()
        self._net_dict['model_dict'] = net_dict
        self._lock.release()

    def get_net_dict(self) -> dict():
        self._lock.acquire()
        net_dict = self._net_dict['model_dict']
        self._lock.release()

        return net_dict