import torch as th

class ResidualBlock(th.nn.Module):
    def __init__(self, n_maps):
        super(ResidualBlock, self).__init__()

        self.model = th.nn.Sequential(
            th.nn.Conv2d(in_channels=n_maps, out_channels=n_maps, kernel_size=3, stride=1, padding=1, bias=False),
            th.nn.BatchNorm2d(n_maps),
            th.nn.ReLU(),
            th.nn.Conv2d(in_channels=n_maps, out_channels=n_maps, kernel_size=3, stride=1, padding=1, bias=False),
            th.nn.BatchNorm2d(n_maps),
        )

    def forward(self, x):
        return th.relu(x + self.model(x))