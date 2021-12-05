import presets.models.GAN.WGAN.DCGANReference


import torch

from presets.exceptions.ImageSizeException import ImageSizeException

import torchvision.utils as tvutils
import torch.nn as nn

from presets.models.GAN.WGAN import DCGANReference


class Discriminator(DCGANReference.Discriminator):
    def __init__(self, gp, img_channels, features_d, num_blocks=3):
        self.gp = gp
        super(Discriminator, self).__init__(img_channels, features_d, num_blocks=3)


    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True)if self.gp else nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

class WGan(DCGANReference.DCGANReference):

    def __init__(self, gp=True, **kwargs):
        self.gp = gp
        super(WGan, self).__init__(**kwargs)
        self.discriminator.activation = nn.Identity()

    def get_discriminator(self, img_channels, features_d, num_blocks):
        return Discriminator(self.gp,  img_channels, features_d, num_blocks)


    def train_batch(self, data, train, epoch, cycle):
        do_gp = train.gradient_penalty

        real, _ = data  # contains dataset image and label
        expected_shape = torch.Size([train.batch_size, self.image_channels, self.image_size, self.image_size])
        if not real.shape[1:] == expected_shape[1:]:
            raise ImageSizeException(real, expected_shape)
        smaller_batch_size = False
        if not real.shape[0] == expected_shape[0]:
            smaller_batch_size = True


        for _ in range(train.critic_iterations):
            noise = torch.randn(train.batch_size, self.z_dim, 1, 1)
            fake = self.gen(noise)  # generate fake with generator

            c_real = self.disc(real).reshape(-1)
            c_fake = self.disc(fake).reshape(-1)
            if do_gp and not smaller_batch_size:
                gp = gradient_penalty(self.discriminator, real, fake)
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

            #weight clipping
            if not do_gp:
                for p in self.discriminator.parameters():
                    p.data.clamp_(-train.weight_clip, train.weight_clip)

        # train generator
        c_fake = self.disc(fake)
        g_loss = train.criterion.g(c_fake=c_fake)

        train.optim.g.zero_grad()
        g_loss.backward()
        g_loss_number: float = g_loss.item()
        train.optim.g.step()

        if cycle % 100 == 0:
            fake_grid = tvutils.make_grid(self.gen(torch.randn(train.batch_size, self.z_dim, 1, 1)))
            real_grid = tvutils.make_grid(real)
            train.writer.add_image("fake_images", fake_grid, global_step=train.step)
            train.writer.add_image("real_images", real_grid, global_step=train.step)
        train.writer.add_scalar("Critic loss", c_loss, train.step)
        train.writer.add_scalar("Generator loss", g_loss, train.step)

        return g_loss_number


def gradient_penalty(critic, real, fake):
    bs, c, h, w = real.shape
    epsilon = torch.rand((bs, 1, 1, 1)).repeat(1, c, h, w)# for batch_size: image wich is filled with ONE individual random number for this image
    interpolated_image = real * epsilon + fake * (1-epsilon)
    #calculate critic score
    c_interpolated = critic(interpolated_image)
    gradient = torch.autograd.grad(
        inputs = interpolated_image,
        outputs=c_interpolated,
        grad_outputs=torch.ones_like(c_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp


if __name__ == "__main__":
    m = WGan()
    print(m.discriminator)
