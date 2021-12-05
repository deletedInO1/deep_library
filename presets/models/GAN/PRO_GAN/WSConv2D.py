import torch
import torch.nn as nn


class WSConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, gain=2):
        super(WSConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.scale = (gain/(kernel_size**2 * in_channels))**0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        c = self.conv(x * self.scale)
        bias = self.bias.view(1, self.bias.shape[0], 1, 1)
        return c + bias
