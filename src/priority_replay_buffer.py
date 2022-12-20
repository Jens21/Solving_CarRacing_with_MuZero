import numpy as np
import torch.multiprocessing
import torch as th

class PriorityReplayBuffer():
    _mem_idx = th.LongTensor([0]).share_memory_()
    _current_mem_size = th.LongTensor([0]).share_memory_()
    _lock = th.multiprocessing.Lock()

    def __init__(self, mem_size, n_history, K, n_actions):
        self._mem_size = mem_size

        self.obs = th.empty((mem_size, n_history, 96, 96)).float().share_memory_()
        self.rewards = th.empty((mem_size, K)).float().share_memory_()
        self.values = th.empty((mem_size, K+1)).float().share_memory_()
        self.policies = th.empty((mem_size, K+1, n_actions)).float().share_memory_()
        self.prev_action_indices = th.empty((mem_size, K, n_actions)).float().share_memory_()
        self.action_indices = th.empty((mem_size, K, n_actions)).float().share_memory_()

    def len(self):
        return len(self._mem)

    def add_samples(self, obs, rewards, values, policies, prev_action_indices, action_indices):
        # this method adds a new item to the replay memory with the given priority
        # item: the item to store

        self._lock.acquire()
        raise NotImplementedError()
        self._lock.release()

    def get_random_samples(self, n_samples):
        # this method returns n_samples random samples
        # n_samples: the number of random samples to return
        # return: samples (C, item)

        self._lock.acquire()
        indices = np.random.randint(0, len(self._mem), n_samples)
        data = [self._mem[i] for i in indices]
        self._lock.release()

        return data