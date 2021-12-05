from math import log2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as tvutils
from tqdm import tqdm
import numpy as np

import presets.models.GAN.GAN as gan
from presets.exceptions.ImageSizeException import ImageSizeException
from presets.models.GAN.WGAN.WGAN import gradient_penalty
from presets.models.GAN.gan_var import GanVar
from presets.models.GAN.PRO_GAN.Generator import Generator
from presets.models.GAN.PRO_GAN.Discriminator import Discriminator
from presets.training.GAN.ProGANTrainingParams import ProGANTrainingParams
from presets.training.TrainingParams import TrainingParams

fixed_noise = torch.randn(1, 100, 1, 1)

class NetAlphaStep(nn.Module):
    def __init__(self, n, a, s):
        super(NetAlphaStep, self).__init__()
        self.n = n
        self.a = a
        self.s = s

    def forward(self, x):
        return self.n(x, self.a, self.s)

class ProGAN(gan.GAN):
    def __init__(self, z_dim=100, in_channels=256, img_channels=3, factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]):
        super(ProGAN, self).__init__(generator=self.get_generator(z_dim, in_channels, img_channels, factors),
                                     discriminator=self.get_discriminator(in_channels, img_channels, factors),
                                     z_dim=z_dim,
                                     img_channels=img_channels,
                                     img_size=4
                                     )

    def get_generator(self, z_dim, in_channels, img_channels, factors):
        return Generator(z_dim, in_channels, img_channels, factors)

    def get_discriminator(self, in_channels, img_channels, factors):
        return Discriminator(in_channels, img_channels, factors)

    def run_train(self, training:ProGANTrainingParams, loss_count=1, save=True, load=True):
        self.try_load(self, training.optim, load)
        self.generator.train()
        self.discriminator.train()


        #loss_count: the last ...  losses will be used for average
        step = int(log2(training.start_img_size / 4)) #0: 4x4, 1: 8x8, 2:16x16, ...

        training.step = 0
        losses = []

        get_loader_fn = training.data_loader #function to get loader
        training.data_loader = None #against bugs

        for _ in range(step, training.resolution_count):
            print("Training for image size " + str(2**step*4) + ".")
            num_epochs = training.progressive_epochs[step]
            training.data_loader = get_loader_fn(step)
            training.total_prog_batch_count = num_epochs * len(training.data_loader.dataset)
            training.current_batch = 0

            for epoch in range(num_epochs):
                l = self.train_epoch(training, epoch, step=step)
                losses.append(l)
                if len(losses) > loss_count:
                    losses = list(losses[-loss_count:])

                self.try_save(self, training.optim, save)
            print("losses: " + str(losses))
            step += 1

        return sum(losses) / len(losses)

    def train_epoch(self, training:ProGANTrainingParams, epoch, step):

        loop = tqdm(training.data_loader)
        losses = []
        for idx, data in enumerate(loop):
            l = self.train_batch(data, training, epoch, cycle=idx, step=step)
            losses.append(l)
            training.step += 1
            training.current_batch += 1
        return sum(losses) / len(losses)

    def train_batch(self, data, train:ProGANTrainingParams, epoch, cycle, step):
        do_gp = train.gradient_penalty

        alpha = train.current_batch / train.total_prog_batch_count
        alpha = max(alpha, 1e-5)
        if alpha > 1:
            print("alpha was bigger than 1!")
            alpha = 1
        elif alpha == 1:
            print("alpha is 1!")

        real, _ = data  # contains dataset image and label


        if not real.shape[0] == train.batch_sizes[step]:
            smaller_batch_size = True
        else:
            smaller_batch_size = False

        for _ in range(train.critic_iterations):
            noise = torch.randn(train.batch_sizes[step], self.z_dim, 1, 1)
            fake = self.generator(noise, alpha, step)  # generate fake with generator

            c_real = self.discriminator(real, alpha, step).reshape(-1)
            c_fake = self.discriminator(fake, alpha, step).reshape(-1)
            if do_gp and not smaller_batch_size:
                gp = gradient_penalty(NetAlphaStep(self.discriminator, a=alpha, s=step), real, fake)
                c_loss = (
                        train.criterion.d(c_real=c_real, c_fake=c_fake)  # minimize -(c_real - c_fake)
                        + train.gp_coefficient * gp
                )
            else:
                c_loss = (
                    train.criterion.d(c_real=c_real, c_fake=c_fake)  # minimize -(c_real - c_fake)
                )
            train.optim.d.zero_grad()
            c_loss.backward(retain_graph=True)
            train.optim.d.step()

            # weight clipping
            if not do_gp:
                for p in self.discriminator.parameters():
                    p.data.clamp_(-train.weight_clip, train.weight_clip)

        # train generator
        c_fake = self.discriminator(fake, alpha, step)
        g_loss = train.criterion.g(c_fake=c_fake)

        train.optim.g.zero_grad()
        g_loss.backward()
        g_loss_number: float = g_loss.item()
        train.optim.g.step()

        if cycle % 10 == 0:
            fake_grid = tvutils.make_grid(self.generator(torch.randn(train.batch_sizes[step], self.z_dim, 1, 1), alpha, step), normalize=True)
            real_grid = tvutils.make_grid(real, normalize=True)
            fixed_fake_grid = tvutils.make_grid(self.generator(fixed_noise, alpha, step), normalize=True)
            fixed_noise_grid = tvutils.make_grid(torch.reshape(fixed_noise, (1, 1, 10, -1)))
            train.writer.add_image("fake_images", fake_grid, global_step=train.step)
            train.writer.add_image("real_images", real_grid, global_step=train.step,)
            train.writer.add_image("fixed_images", fixed_fake_grid, global_step = train.step)
            train.writer.add_image("fixed_noise", fixed_noise_grid, global_step = train.step)
        train.writer.add_scalar("Critic loss", c_loss, train.step)
        train.writer.add_scalar("Generator loss", g_loss, train.step)

        return g_loss_number

if __name__ == "__main__":
    batch_size = 64
    in_channels = 1
    net = ProGAN(in_channels)
    img = torch.randn(batch_size, in_channels)
