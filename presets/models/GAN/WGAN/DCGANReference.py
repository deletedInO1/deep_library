import torch
import torch.nn as nn

import presets.models.GAN.GAN as gan
from presets.models.GAN.gan_var import GanVar
from presets.training.train import TrainingParams


class DCGANReference(gan.GAN):
    def __init__(self, img_channels=3, z_dim=100, features : GanVar=GanVar(64,64), num_blocks=3, img_size=64):
        #img_channels: the amount of channels in the final img; RGB -> 3
        #z_dim:  the length of the 1D array of random numbers as the generator input
        #num_blocks:  count of blocks except the initial and final block


        super(DCGANReference, self).__init__(self.get_generator(z_dim, img_channels, features.g, num_blocks),
                                    self.get_discriminator(img_channels, features.d, num_blocks), 100, img_channels, img_size)
        initialize_weights(self.discriminator)
        initialize_weights(self.generator)
    def get_discriminator(self, img_channels, features_d, num_blocks):
        return Discriminator(img_channels, features_d, num_blocks)
    def get_generator(self, z_dim, img_channels, features_g, num_blocks):
        return Generator(z_dim, img_channels, features_g, num_blocks)




class Discriminator(gan.Discriminator):
    def __init__(self, img_channels, features_d, num_blocks=3):#paper: img_channels=3, features_d=64, num_blocks:3
        super(Discriminator, self).__init__()

        #(before) size:  img_channels # img_size x  img_size = 3#64x64
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        #(before blocks after initial) size:  features_d#(img_size/2)x(img_size/2) = 64#32x32
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(self.block(features_d*(2**i), features_d*2*(2**i), 4, 2, 1))
        # size: (features_d*(2**num_blocks))#4x4 = 512#4x4
        self.final = nn.Conv2d(features_d*(2**num_blocks), 1, kernel_size=4, stride=2, padding=0)
        #size: 1#1x1
        self.activation = nn.Sigmoid()
        # size: 1#1x1

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        outp = x
        outp = self.initial(outp)
        for b in self.blocks:
            outp = b(outp)
        outp = self.final(outp)
        return self.activation(outp)

class Generator(gan.Generator):
    def __init__(self, z_dim, img_channels, features_g, num_blocks=3): #paper: z_dim=100, img_channels=3, features_g=64, num_blocks:3
        super(Generator, self).__init__()
        # size: z_dim#1x1
        self.initial = self.block(z_dim, features_g*(2**(num_blocks+1)), 4, 1, 0)
        # size: (features_g*(2**(num_blocks+1))#4x4=1024#4x4

        self.blocks = nn.ModuleList()
        for i in range(num_blocks+1, 1, -1):
            self.blocks.append(self.block(features_g*(2**i), int(features_g*(2**i)/2), 4, 2, 1))
        # size: (features_g*2)#4x4=128#32x32
        self.final =nn.ConvTranspose2d(features_g*2, img_channels, kernel_size=4, stride=2, padding=1)
        # size: z_dim#4x4=3#64x54
        self.activation = nn.Tanh()
    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(),
                             )
    def forward(self, x):
        outp = x
        outp = self.initial(outp)
        for m in self.blocks:
            outp = m(outp)
        return self.activation(self.final(outp))

def initialize_weights(model):

    #mean of weights is 0
    #the standarddeviation (Standardabweichung) is 0.02

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == "__main__":
    def test():
        x = torch.randn((N, img_channels, H, W))
        d = dcgan.disc(x)
        assert d.shape == (N, 1, 1, 1)
        z = torch.randn((N, z_dim, 1, 1))
        g = dcgan.gen(z)
        assert g.shape == (N, img_channels, H, W)
        print("success")

    def print_disc():
        print('discriminator:')
        print('initial:', dcgan.discriminator.initial)
        print('blocks:')
        for b in dcgan.discriminator.blocks:
            print('\t', b)
        print(dcgan.discriminator.final)


    def print_gen():
        print('generator:')
        for b in dcgan.generator.net:
            print(b)

    N, img_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    dcgan = DCGANReference(img_channels=img_channels, z_dim=z_dim, features=GanVar(64, 64), num_blocks=3, img_size=64)
    #print_gen()
    #print_disc()
    test()


