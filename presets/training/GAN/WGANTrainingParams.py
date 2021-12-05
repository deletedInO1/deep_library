from presets.training.GAN.GANTrainingParams import GANTrainingParams
from presets.models.GAN.gan_var import GanVar
from presets.models.GAN import GAN

import torch
import torch.optim as toptim
import torch.nn as nn

class WGANTrainingParams(GANTrainingParams):
    def __init__(self,  model : GAN, data_loader,
                 gradient_penalty=True,  gp_coefficient=10, critic_iterations=5, weight_clip=0.01,
                 batch_size=64, num_epochs=5, do_gen_writer=True,):

        super(WGANTrainingParams, self).__init__(
            optim=get_optim(model,gradient_penalty),
            criterion=GanVar(loss_fn_d, loss_fn_g),
            batch_size=batch_size,
            num_epochs=num_epochs,

            data_loader=data_loader,
            do_gen_writer=do_gen_writer,
            )

        self.critic_iterations = critic_iterations
        self.weight_clip = weight_clip
        self.gradient_penalty = gradient_penalty
        self.gp_coefficient = gp_coefficient


def get_optim(model, gp):
    if gp:
        return get_optim_gp(model)
    else:
        return get_optim_nogp(model)

def get_optim_nogp(model):
    optim = GanVar(
        toptim.RMSprop(model.discriminator.parameters(), lr=5e-5),
        toptim.RMSprop(model.generator.parameters(), lr=5e-5)) if model else None
    return optim

def get_optim_gp(model):
    optim = GanVar(
        toptim.Adam(model.discriminator.parameters(), lr=1e-4, betas=(0, 0.9)),
        toptim.Adam(model.generator.parameters(), lr=1e-4, betas=(0, 0.9))) if model else None
    return optim

def loss_fn_d(c_real, c_fake):
    return -(torch.mean(c_real) - torch.mean(c_fake))

def loss_fn_g(c_fake):
    return -torch.mean(c_fake)

