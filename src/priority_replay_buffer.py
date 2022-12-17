import numpy as np

class PriorityReplayBuffer():
    _mem = []
    _mem_idx = 0

    def __init__(self, mem_size):
        self._mem_size = mem_size

    def len(self):
        return len(self._mem)

    def add_sample(self, item):
        # this method adds a new item to the replay memory with the given priority
        # item: the item to store

        if len(self._mem) < self._mem_size:
            self._mem.append(item)
        else:
            self._mem[self._mem_idx] = item
            self._mem_idx += 1
            self._mem_idx %= self._mem_size

    def get_random_samples(self, n_samples):
        # this method returns n_samples random samples
        # n_samples: the number of random samples to return
        # return: samples (C, item)

        samples = np.random.choice(self._mem, n_samples)
        return samples