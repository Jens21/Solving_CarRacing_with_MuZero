import torch as th
from prediction_network import PredictionNetwork as PredictionNetwork
from representation_network import RepresentationNetwork as RepresentationNetwork
from dynamic_network import DynamicNetwork as DynamicNetwork
from node import Node as Node
import numpy as np
import time

class Network(th.nn.Module):
    possible_actions = th.FloatTensor([[1,0,0], [-1,0,0], [0,1,0], [0,0,0.8], [0,0,0]])

    n_maps = 32
    n_res_blocks_pred = 8
    n_res_blocks_dyn = 8
    n_res_blocks_repr = 8
    n_actions = len(possible_actions)
    n_history = 5
    n_simulations = 50
    c_1 = 1.25
    c_2 = 19652
    gamma = 0.99
    K = 5
    n = 5 # TODO, was 10 in the paper

    def __init__(self, device):
        super(Network, self).__init__()
        self.prediction_network = PredictionNetwork(n_maps=self.n_maps, n_res_blocks=self.n_res_blocks_pred, n_actions=self.n_actions).to(device)
        self.dynamic_network = DynamicNetwork(n_maps=self.n_maps, n_res_blocks=self.n_res_blocks_dyn, n_actions=self.n_actions).to(device)
        self.representation_network = RepresentationNetwork(n_maps=self.n_maps, n_res_blocks=self.n_res_blocks_repr, n_history=self.n_history).to(device)
        self.device = device

        # self.optimizer = th.optim.AdamW(self.parameters())
        self.optimizer = th.optim.Adam(self.parameters())
        self.loss_fn = th.nn.MSELoss()
        self.times = []

    def select_leaf(self, tree):
        def compute_action_idx(tree, node_idx):
            P = tree[node_idx].P
            Q = tree[node_idx].Q
            N = tree[node_idx].N
            action_idx = th.argmax(Q + P * N.sum().sqrt()/(1 + N) * (self.c_1 + ((N.sum() + self.c_2 + 1)/self.c_2).log()))

            return action_idx

        node_idx = 0
        node_trace = [node_idx]
        action_idx = compute_action_idx(tree, node_idx)
        action_trace = [action_idx]

        while not tree[node_idx].is_leaf:
            node_idx = tree[node_idx].S[action_idx]
            node_trace.append(node_idx)
            action_idx = compute_action_idx(tree, node_idx)
            action_trace.append(action_idx)

        return node_trace, action_trace, node_idx, action_idx

    def expand_the_tree(self, tree, trace, node_idx, action_idx):
        parent_node = tree[node_idx]
        state = parent_node.hidden_state

        tiled_action_idx = th.tile(action_idx/(self.n_actions-1), (1, 1, 6, 6))
        inp = th.concat([state, tiled_action_idx], dim=1)

        state, reward = self.dynamic_network(inp)
        reward = reward[0]
        p, v = self.prediction_network(state)

        p = p[0]
        v = v[0]
        idx = len(tree)

        parent_node.S[action_idx] = idx
        parent_node.R[action_idx] = reward
        parent_node.is_leaf = False
        tree.append(Node(hidden_state=state, policy=p, value=v))

        return tree, trace

    def backup_the_tree(self, tree, node_trace, action_trace):
        v = tree[tree[node_trace[-1]].S[action_trace[-1]]].value
        G = self.gamma * v

        for (node_idx, action_idx) in zip(node_trace, action_trace):
            node = tree[node_idx]
            G += node.R[action_idx]
            N = node.N[action_idx]
            Q = node.Q[action_idx]
            node.Q[action_idx] = (N * Q + G) / (N + 1)
            node.N[action_idx] += 1

            G *= self.gamma

        return tree

    def create_the_tree(self, inp):
        s_0 = self.representation_network(inp)
        p_0, v_0 = self.prediction_network(s_0)
        p_0 = p_0[0]
        v_0 = v_0[0]

        # the actual tree is represented as a list with the root node at the beginning
        tree = [Node(hidden_state=s_0, policy=p_0, value=v_0)]

        for _ in range(self.n_simulations):
            node_trace, action_trace, node_idx, action_idx = self.select_leaf(tree)
            tree, trace = self.expand_the_tree(tree, node_trace, node_idx, action_idx)
            tree = self.backup_the_tree(tree, node_trace, action_trace)

        return tree

    def extract_data_from_tree(self, tree, T):
        N = tree[0].N
        probs = N ** (1/T)
        probs /= probs.sum() ** (1/T)
        indices = np.arange(self.possible_actions.shape[0])
        action_idx = np.random.choice(indices, p = probs.numpy())
        action = self.possible_actions[action_idx].numpy().tolist()
        value = tree[0].value
        policy = tree[0].P

        return action, action_idx, value, policy

    def get_network_prediction(self, observations, action_indices, T):
        with th.no_grad():
            # convert the observations and action_indices to the networks input
            observations = (self.n_history)*[np.zeros((96, 96))] + observations
            observations = observations[-self.n_history:]
            observations = th.from_numpy(np.array(observations))
            action_indices = (self.n_history)*[0] + action_indices
            action_indices = action_indices[-self.n_history:]
            action_indices = th.from_numpy(np.array(action_indices)) / (self.n_actions-1)
            action_indices = th.tile(action_indices[:, None, None], (1, 96, 96))

            inp = th.concat([observations, action_indices], dim = 0)[None].to(self.device).float()
            tree = self.create_the_tree(inp)
            action, action_idx, value, policy = self.extract_data_from_tree(tree, T)

            return action, action_idx, value, policy

    def convert_simulation_data_to_buffer_items(self, observations, action_indices, rewards, values, policies):
        observations = observations[:-1]
        action_indices = th.from_numpy(np.array(self.n_history*[0] + action_indices[:-1]))
        rewards = th.from_numpy(np.array(rewards[:-1]))
        values = th.concat(values, dim=0)
        policies = th.stack(policies)
        observations = observations + (self.n_history-1) * [np.zeros((96, 96))]
        observations = th.from_numpy(np.array(observations))

        items = []
        for i in range(len(rewards)-self.K):
            o = observations[i:i+self.n_history].float()
            r = rewards[i : i+self.K].float()
            v = values[i : i+self.K+1].float()
            p = policies[i : i+self.K+1].float()
            prev_act_ind = action_indices[i:i+self.n_history].float()
            act_ind = action_indices[self.n_history+i : self.n_history+i+self.K].float()

            items.append((o, r, v, p, prev_act_ind, act_ind))

        return items

    def forward(self, x):
        raise NotImplementedError()

    def do_train_step(self, replay_buffer, batch_size):
        self.zero_grad()

        items = replay_buffer.get_random_samples(batch_size)

        obs = th.stack([item[0] for item in items], dim = 0).to(self.device)
        rewards = th.stack([item[1] for item in items], dim = 0).to(self.device)
        values = th.stack([item[2] for item in items], dim = 0).to(self.device)
        policies = th.stack([item[3] for item in items]).to(self.device)
        prev_action_indices = th.stack([item[4] for item in items], dim = 0).to(self.device)
        action_indices = th.stack([item[5] for item in items], dim = 0).to(self.device)

        prev_action_indices = prev_action_indices[:, :, None, None].tile((1, 1, 96, 96))
        inp = th.concat([obs, prev_action_indices], dim=1).to(self.device).float()
        s = self.representation_network(inp)
        p, v = self.prediction_network(s)
        v = v.flatten()
        loss = self.loss_fn(v, values[:, 0]) + self.loss_fn(p, policies[:, 0])

        for i in range(self.K):
            tiled_action_idx = th.tile(action_indices[:, i][:, None, None, None] / (self.n_actions - 1), (1, 1, 6, 6))
            inp = th.concat([s, tiled_action_idx], dim=1)

            s, r = self.dynamic_network(inp)
            r = r.flatten()
            p, v = self.prediction_network(s)
            v = v.flatten()
            loss += self.loss_fn(r, rewards[:, i]) + self.loss_fn(v, values[:, 1+i]) + self.loss_fn(p, policies[:, 1+i])

        # TODO, scale the gradients of the dynamic model with 1/2
        loss *= 1./self.K
        loss.backward()

        self.optimizer.step()