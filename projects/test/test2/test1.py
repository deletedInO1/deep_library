from presets.models.GAN.DCGAN.DCGAN import DCGAN
from presets.models.GAN.gan_var import GanVar
from presets.training.train import TrainingParamas

import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as trf
import presets.optimize as optimize
import optuna as op

def gen_params(trial : op.trial.Trial, model):
    params={
        "batch_size": trial.suggest_int("batch_size", 1, 128, 16),
        "optim": GanVar(
            d=optim.Adam(model.discriminator.parameters(), trial.suggest_loguniform("lr_d", 1e-5, 1e-1)),
            g=optim.Adam(model.generator.parameters(), trial.suggest_loguniform("lr_g", 1e-5, 1e-1))
        )
    }
    return params

def gen_model():
    return DCGAN(1, 100, GanVar(8, 8), num_blocks=3)

if __name__ == "__main__":
    batch_size = 64
    transforms = trf.Compose((trf.Resize((64, 64)), trf.ToTensor()))
    dataset = datasets.MNIST("datasets/", True, transforms, download=True)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    training = TrainingParamas(
        optim=None,
        criterion=GanVar(
            d = nn.BCELoss(),
            g = nn.BCELoss()
        ),
        batch_size=batch_size,
        data_loader=data_loader,
        num_epochs=2,
        do_gen_writer=False #for optimizing
    )
    try:
        #gan.run_train(training, 2)
        optimize.optimize(gen_model, training, gen_params)
    except KeyboardInterrupt:
        print("keyboard interrupt")
