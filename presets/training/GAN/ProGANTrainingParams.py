from presets.training.GAN.WGANTrainingParams import WGANTrainingParams
from presets.models.GAN.gan_var import GanVar
from presets.models.GAN import GAN

import torch.optim as toptim
import torch.nn as nn

class ProGANTrainingParams(WGANTrainingParams):
    def __init__(self,  model : GAN, get_data_loader, start_img_size=4, resolution_count=9, gp=True, gp_coefficient=10,  critic_iterations=1, lr=1e-3, do_gen_writer=True):
        super(ProGANTrainingParams, self).__init__(
            model=model,
            batch_size=None,
            num_epochs=None,

            data_loader=get_data_loader, #function to create loader for step
            do_gen_writer=do_gen_writer,
            gradient_penalty=gp,
            gp_coefficient=gp_coefficient,
            critic_iterations=critic_iterations,
        )
        self.optim=GanVar(toptim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.0, 0.99)), toptim.Adam(model.generator.parameters(), lr=lr, betas=(0.0, 0.99)))#override

        self.start_img_size = start_img_size
        self.resolution_count = resolution_count #4, 8, 16, 32, 64, 128, 256, 512, 1024
        self.batch_sizes = [64, 64, 32, 32, 16, 16, 8, 4, 4]
        self.progressive_epochs = [30] * len(self.batch_sizes)

        self.total_prog_batch_count = None
        self.current_batch = None


