import torch
from torch import nn
from torch.nn.init import xavier_uniform_, zeros_
from args import args


class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh, sigmoid=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nout),
        )
        self.sigmoid = sigmoid
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight.data, gain=args.gain)
                zeros_(layer.bias.data)

    def forward(self, x):
        if not self.sigmoid:
            return self.net(x)
        else:
            return torch.sigmoid(self.net(x))
