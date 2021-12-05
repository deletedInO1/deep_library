import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from presets.models.GAN.PRO_GAN.Block import Block
from presets.models.GAN.PRO_GAN.WSConv2D import WSConv2D
from presets.models.GAN.PRO_GAN.ToFromImage import ToImage
from presets.models.GAN.PRO_GAN.PixelNorm import PixelNorm

class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels, factors):
        super(Generator, self).__init__()

        self.blocks = nn.ModuleList()
        self.finals = nn.ModuleList() # rgbs

        initialBlock = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv2D(in_channels=in_channels, out_channels=in_channels*factors[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        firstFinal = ToImage(in_channels, img_channels)

        #initial
        self.blocks.append(initialBlock)
        self.finals.append(firstFinal)

        for i in range(1, len(factors)):
            in_c = int(in_channels * factors[i-1])
            out_c = int(in_channels * factors[i])
            self.blocks.append(Block(in_c, out_c))
            self.finals.append(ToImage(out_c, img_channels))


    def fade_in(self, upscaled, generated, alpha):
        return torch.tanh((upscaled * (1 - alpha)) + (generated * alpha))

    def forward(self, x, alpha, steps):#0: 4x4, 1#8x8
        x = self.blocks[0](x) #initial block
        if steps == 0:
            return self.finals[0](x) #no fade in, because there is nothing to upscale

        for step in range(1, steps+1):# runs from 1 [inclusive] to steps [NOT inclusive]
            upscaled = F.interpolate(x, scale_factor=2, mode="nearest") #upsample
            x = self.blocks[step](upscaled) #progressive block

        old = self.finals[steps - 1](upscaled)
        new = self.finals[steps](x)
        x = self.fade_in(upscaled=old, generated=new, alpha=alpha)
        return x


if __name__ == "__main__":
    factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]
    in_channels = 64
    z_dim = 100
    batch_size = 1
    img_channels = 1
    gen = Generator(z_dim, in_channels, img_channels, factors)

    for idx, x in enumerate(gen.blocks):
        print(idx, x)
    for idx, x in enumerate(gen.finals):
        print(idx, x)

    #exit()
    for i in range(len(factors)):
        x = torch.randn(batch_size, z_dim, 1, 1)
        y = gen(x, 0.8, i)
        print(y.shape)