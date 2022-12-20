import torch as th
from residual_block import ResidualBlock as ResidualBlock

class RepresentationNetwork(th.nn.Module):
    def __init__(self, n_maps, n_res_blocks, n_history):
        super(RepresentationNetwork, self).__init__()

        # according to MuZero
        self.model = th.nn.Sequential(
            *[th.nn.Conv2d(in_channels=2*n_history, out_channels=n_maps//2, kernel_size=3, stride=2, padding=1, bias=False),
            ResidualBlock(n_maps//2),
            ResidualBlock(n_maps//2),
            th.nn.Conv2d(in_channels=n_maps//2, out_channels=n_maps, kernel_size=3, stride=2, padding=1, bias=False),
            ResidualBlock(n_maps),
            ResidualBlock(n_maps),
            ResidualBlock(n_maps),
            th.nn.AvgPool2d(kernel_size=2, stride=2),
            ResidualBlock(n_maps),
            ResidualBlock(n_maps),
            ResidualBlock(n_maps),
            th.nn.AvgPool2d(kernel_size=2, stride=2)]
                +
            [ResidualBlock(n_maps) for _ in range(n_res_blocks)]
        )

    def normalize_hidden_state(self, state):
        min_values, _ = state.reshape((state.shape[0], -1)).min(1)
        min_values = min_values[:, None, None, None]
        max_values, _ = state.reshape((state.shape[0], -1)).max(1)
        max_values = max_values[:, None, None, None]
        state = (state-min_values) / (max_values-min_values)

        return state

    def forward(self, x):
        # x shape: (batch_size, channels, 96, 96)
        out = self.model(x)

        return self.normalize_hidden_state(out)

if __name__ == '__main__':
    n_maps = 32
    n_res_blocks = 8
    n_history = 5
    net = RepresentationNetwork(n_maps, n_res_blocks, n_history)

    inp = th.rand((128,2*n_history,96,96))
    out = net(inp)
    print(out.shape)
