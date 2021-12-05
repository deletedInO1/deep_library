from presets.training.GAN.GANTrainingParams import GANTrainingParams
from presets.models.GAN.gan_var import GanVar
from presets.models.GAN import GAN

import torch.optim as toptim
import torch.nn as nn

class DCGANTrainingParamas(GANTrainingParams):
    def __init__(self,  model : GAN, data_loader,  do_gen_writer=True, optim=None):
        super(DCGANTrainingParamas, self).__init__(
            optim=GanVar(toptim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
                         , toptim.Adam(model.generator.parameters(),  lr=2e-4, betas=(0.5, 0.999))) if model else None,
            criterion=GanVar(nn.BCELoss(), nn.BCELoss()),
            batch_size=128,
            num_epochs=5,

            data_loader=data_loader,
            do_gen_writer=do_gen_writer,
        )
