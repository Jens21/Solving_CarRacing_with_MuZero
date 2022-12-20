import torch as th
from residual_block import ResidualBlock as ResidualBlock

class DynamicNetwork(th.nn.Module):
    def __init__(self, n_maps, n_res_blocks, n_actions):
        super(DynamicNetwork, self).__init__()

        self.n_actions = n_actions

        self.res_blocks = th.nn.Sequential(*[th.nn.Conv2d(in_channels=n_maps+1, out_channels=n_maps, kernel_size=1, stride=1, padding=0)]
                                            +
                                            [ResidualBlock(n_maps) for _ in range(n_res_blocks)])

        self.reward_network = th.nn.Sequential(
            th.nn.Conv2d(in_channels=n_maps, out_channels=n_maps//2, kernel_size=1, stride=1, padding=0),
            th.nn.Flatten(1),
            th.nn.Linear(6*6*(n_maps//2), 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 1)
        )

    def normalize_hidden_state(self, state):
        min_values, _ = state.reshape((state.shape[0], -1)).min(1)
        min_values = min_values[:, None, None, None]
        max_values, _ = state.reshape((state.shape[0], -1)).max(1)
        max_values = max_values[:, None, None, None]
        state = (state-min_values) / (max_values-min_values)

        return state

    def forward(self, x):
        state = self.res_blocks(x)
        reward = self.reward_network(state)

        return self.normalize_hidden_state(state), reward

if __name__ == '__main__':
    n_maps = 32
    n_res_blocks = 8
    n_actions = 5
    net = DynamicNetwork(n_maps, n_res_blocks, n_actions)

    inp = th.rand((128, n_maps+1, 6, 6))
    hidden_states, rewards = net(inp)

    print(rewards.shape)
    print(hidden_states.shape)