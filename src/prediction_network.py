import torch as th
from residual_block import ResidualBlock as ResidualBlock
import time
import numpy as np

class PredictionNetwork(th.nn.Module):
    def __init__(self, n_maps, n_res_blocks, n_actions):
        super(PredictionNetwork, self).__init__()

        self.res_blocks = th.nn.Sequential(*[ResidualBlock(n_maps) for _ in range(n_res_blocks)])
        self.policy_head = th.nn.Sequential(
            th.nn.Conv2d(in_channels=n_maps, out_channels=n_maps//2, kernel_size=1, stride=1, padding=0),
            th.nn.Flatten(1),
            th.nn.Linear(6*6*(n_maps//2), 64),
            th.nn.ReLU(),
            th.nn.Linear(64, n_actions),
            th.nn.Softmax(1)
        )
        self.value_head = th.nn.Sequential(
            th.nn.Conv2d(in_channels=n_maps, out_channels=n_maps//2, kernel_size=1, stride=1, padding=0),
            th.nn.Flatten(1),
            th.nn.Linear(6*6*(n_maps//2), 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 1),
            th.nn.Tanh()
        )

    def forward(self, x):
        # x shape: (batch_size, channels, 6, 6)

        out = self.res_blocks(x)

        policy_out = self.policy_head(out)
        value_out = self.value_head(out)


        return policy_out, value_out

if __name__ == '__main__':
    n_maps = 32
    n_res_blocks = 8
    n_actions = 5
    net = PredictionNetwork(n_maps, n_res_blocks, n_actions)

    inp = th.rand((128, 32, 6, 6))
    policy_out, value_out = net(inp)

    print(policy_out.shape)
    print(value_out.shape)
