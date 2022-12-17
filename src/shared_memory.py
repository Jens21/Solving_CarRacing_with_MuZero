import threading

class SharedMemory():
    policy_net = None
    target_net = None

    def __init__(self):
        lock = threading.Lock()

    def set_nets(self, policy_net, target_net):
        self.lock.acquire()
        self.policy_net = policy_net
        self.target_net = target_net
        self.lock.release()

    def get_nets(self):
        self.lock.acquire()
        pol, tar = self.policy_net, self.target_net
        self.lock.release()

        return pol, tar