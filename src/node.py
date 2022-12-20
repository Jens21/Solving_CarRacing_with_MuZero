import torch as th

class Node():
    def __init__(self, hidden_state, policy, value):
        self.hidden_state = hidden_state
        self.P = policy
        self.N = th.zeros(self.P.shape[0])
        self.Q = th.zeros(self.P.shape[0])

        self.R = th.inf * th.ones(self.P.shape[0])
        self.S = -1 * th.ones(self.P.shape[0], dtype=th.int) # the children indices in the list

        self.value = value
        self.is_leaf = True

    def __str__(self):
        # return 'Node(N: {}, Q: {}, P: {}, R: {}, S: {})'.format(self.N.int().tolist(), self.Q.float().tolist(), self.P.float().tolist(), self.R.float().tolist(), self.S.int().tolist())
        return 'Node(N: {}, Q: {}, R: {}, S: {})'.format(self.N.int().tolist(), self.Q.float().tolist(), self.R.float().tolist(), self.S.int().tolist())

