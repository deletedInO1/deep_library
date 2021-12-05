import torch
import torch.nn as nn
from presets.models.GAN.PRO_GAN.WSConv2D import WSConv2D
from presets.models.GAN.PRO_GAN.PixelNorm import PixelNorm

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_pn=True):
        super(Block, self).__init__()
        self.conv1 = WSConv2D(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = WSConv2D(out_channels, out_channels, kernel_size, stride, padding)

        self.activation = nn.LeakyReLU(0.2)

        self.use_pn = use_pn
        self.pn = PixelNorm()
        self.in_c, self.out_c = in_channels, out_channels

    def forward(self, x):
        x = self.activation(self.conv1(x))
        if self.use_pn:
            x = self.pn(x)
        x = self.activation(self.conv2(x))
        if self.use_pn:
            x = self.pn(x)
        return x

    def __str__(self):
        return "Block("+str(self.in_c) + ", " +str(self.out_c)+")"

if __name__ == "__main__":
    b = Block(512, 64)
    x = torch.randn(128, 512, 32, 32)
    y = b(x)

    print(y.shape)