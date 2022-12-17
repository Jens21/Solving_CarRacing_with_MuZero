import threading

class SharedMemory():
    policy_net_dict = None
    target_net_dict = None

    def __init__(self):
        lock = threading.Lock()

    def set_nets_dict(self, policy_net_dict, target_net_dict):
        self.lock.acquire()
        self.policy_net_dict = policy_net_dict
        self.target_net_dict = target_net_dict
        self.lock.release()

    def get_nets_dict(self):
        self.lock.acquire()
        pol, tar = self.policy_net_dict, self.target_net_dict
        self.lock.release()

        return pol, tar