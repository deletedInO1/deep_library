import torch

from presets.models.model import Model
from presets.training.train import TrainingParams, Trainable
from presets.exceptions.ImageSizeException import ImageSizeException
import presets.save_load as sl

import torchvision.utils as tvutils


class GAN(Trainable):
    def __init__(self, generator, discriminator, z_dim, img_channels, img_size):
        self.generator : Generator = generator
        self.discriminator : Discriminator = discriminator
        self.z_dim = z_dim
        self.image_size = img_size
        self.image_channels = img_channels

    def get_discriminator(self):
        raise NotImplementedError()

    def get_generator(self):
        raise NotImplementedError()

    def disc(self, x):
        return self.discriminator(x)

    def gen(self, x):
        return self.generator(x)

    def try_load(self, model, optimizer, do_load):
        if do_load:
            try:
                sl.load_checkpoint(model, optimizer, is_gan=True)
            except FileNotFoundError:
                pass

    def try_save(self, model, optimizer, do_save):
        if do_save:
            sl.save_checkpoint(model, optimizer, is_gan=True)

    def run_train(self, training:TrainingParams, loss_count=1, save=True, load=True):
        self.generator.train()
        self.discriminator.train()
        super(GAN, self).run_train(training, loss_count=loss_count, save=save, load=load)

    def train_epoch(self, training:TrainingParams, epoch):
        return super(GAN, self).train_epoch(training, epoch)
        #save_image(self.gen(torch.randn(1, self.z_dim, 1, 1)), "test2"+str(epoch)+".jpg")

    def train_batch(self, data, train :TrainingParams, epoch, cycle):
        # for each output from dataloader
        noise = torch.randn(train.batch_size, self.z_dim, 1, 1)
        real, _ = data # contains dataset image and label
        expected_shape = torch.Size([train.batch_size, self.image_channels, self.image_size, self.image_size])
        if not real.shape[1:] == expected_shape[1:]:
            raise ImageSizeException(real, expected_shape)
        fake = self.gen(noise) # generate fake with generator

        d_real = self.disc(real)
        d_fake = self.disc(fake.detach())
        d_real_loss = train.criterion.d(d_real, torch.ones_like(d_real) * 0.95)
        d_fake_loss = train.criterion.d(d_fake, torch.ones_like(d_fake) * 0.05)
        d_loss = (d_real_loss + d_fake_loss) / 2



        train.optim.d.zero_grad()
        d_loss.backward()
        train.optim.d.step()

        d_fake = self.disc(fake)
        g_loss = train.criterion.g(d_fake, torch.ones_like(d_fake) * 0.95)


        train.optim.g.zero_grad()
        g_loss.backward()
        g_loss_number : float = g_loss.item()
        train.optim.g.step()

        if cycle % 100 == 0:
            fake_grid = tvutils.make_grid(self.gen(torch.randn(train.batch_size, self.z_dim, 1, 1)))
            real_grid = tvutils.make_grid(real)
            train.writer.add_image("fake_images", fake_grid, global_step=train.step)
            train.writer.add_image("real_images", real_grid, global_step=train.step)
        train.writer.add_scalar("Discriminator loss", d_loss, train.step)
        train.writer.add_scalar("Generator loss", g_loss, train.step)

        return g_loss_number

    def test_batch(self, *args, **kwargs):
        pass #do nothing
    def test(self, training):
        pass


class Discriminator(Model):
    pass

class Generator(Model):
    pass