import torch
import torch.nn as nn
import torch.nn.functional as F
from presets.models.GAN.PRO_GAN.Block import Block
from presets.models.GAN.PRO_GAN.WSConv2D import WSConv2D
from presets.models.GAN.PRO_GAN.ToFromImage import FromImage


class MinibatchStandardDeviation(nn.Module):
    def __init__(self):
        super(MinibatchStandardDeviation, self).__init__()

    def forward(self, x):
        # standard deviation = Standardabweichung
        standard_deviation = torch.std(x, dim=0).mean()
        repeated = standard_deviation.repeat(x.shape[0], 1, x.shape[2], x.shape[3]) #batch statistics
        return torch.cat([x, repeated], dim=1)

class Discriminator(nn.Module):
    def __init__(self, gen_in_channels, img_channels, factors):
        super(Discriminator, self).__init__()
        self.factors = factors
        self.blocks = nn.ModuleList()
        self.initials = nn.ModuleList() #from rgbs

        assert int(gen_in_channels * factors[len(factors)-1]) != 0, "gen_in_channels to small"

        for factor_idx in range(len(factors)-1, 0, -1):
            in_c = int(factors[factor_idx] * gen_in_channels)
            out_c = int(factors[factor_idx-1] * gen_in_channels)

            self.initials.append(FromImage(img_channels, in_c))
            self.blocks.append(Block(in_c, out_c))


        self.initials.append(FromImage(img_channels, gen_in_channels)) # for 4x4
        last_kernel_size = 4
        self.blocks.append(nn.Sequential(
            MinibatchStandardDeviation(),
            WSConv2D(gen_in_channels+1, gen_in_channels, kernel_size=3, stride=1, padding=1), # gen_in_channels+1: 'cause of the minibatch std
            nn.LeakyReLU(0.2),
            nn.Conv2d(gen_in_channels, gen_in_channels, kernel_size=last_kernel_size, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv2D(gen_in_channels, 1, kernel_size=1, padding=0, stride=1),
        ))

        self.down_scale = nn.AvgPool2d(2)
        self.activation = nn.LeakyReLU(0.2)


    def fade_in(self, downscaled, new, alpha):
        return (downscaled * (1 - alpha)) + (new * alpha)

    def forward(self, x, alpha, steps):
        begin = len(self.factors)-steps-1
        out = self.activation(self.initials[begin](x))

        if steps == 0:
            out = self.blocks[begin](out) # including minibatch std
            return out.view(x.shape[0], -1)

        # fade
        new = self.blocks[begin](out)
        new = self.down_scale(new)
        downscaled = self.down_scale(x)
        downscaled = self.initials[begin + 1](downscaled)
        downscaled = self.activation(downscaled)

        x = self.fade_in(downscaled, new, alpha)

        #blocks
        for i in range(begin+1, len(self.blocks)-1):
            x = self.blocks[i](x)
            x = self.down_scale(x)

        x = self.blocks[-1](x) # final block
        x = x.view(x.shape[0], -1)
        return x







if __name__ == "__main__":
    factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]
    #factors = [1, 1, 1, 1/2, 1/4]
    gen_c = 128
    disc = Discriminator(gen_c, 3, factors)
    for idx, x in enumerate(disc.blocks):
        print(idx, x)

    print()

    for idx, x in enumerate(disc.initials):
        print(idx, x)


    print()
    print()
    for i in range(len(factors)):
        img_size = 2 ** i * 4
        print("image size:", img_size)
        x = torch.randn(1, 3, img_size, img_size)
        y = disc(x, 0.8, i)
        assert sum(y.shape) == len(y.shape) # each member equals 0
        #print(y.shape)