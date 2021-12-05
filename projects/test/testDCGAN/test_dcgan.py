from tensorboard import data

from presets.models.GAN.DCGAN.DCGAN import DCGAN
from presets.training.GAN.DCGANTrainingParams import DCGANTrainingParamas
from presets.models.GAN.gan_var import GanVar

import torch.optim as optim
import torchvision.transforms as trfs
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import projects.test.TEST_CONFIG as conf

if __name__ == "__main__":
    dcgan = DCGAN(img_channels=1, features=GanVar(8,8))
    img_size = 64
    transforms = trfs.Compose([
        trfs.Resize(dcgan.image_size),
        trfs.ToTensor(),
        trfs.Normalize(
            [0.5 for _ in range(dcgan.image_channels)], #mean
            [0.5 for _ in range(dcgan.image_channels)] #std
        ),
    ])
    dataset = datasets.MNIST(root=conf.TO_ROOT+conf.MNIST_LOCATION, train=True, download=True, transform=transforms)
    #dataset = datasets.ImageFolder(conf.TO_ROOT + conf.HORSE_LOCATION, transform=transforms)

    training_params = DCGANTrainingParamas(model=dcgan, data_loader=None)
    loader = DataLoader(dataset, batch_size= training_params.batch_size, shuffle=True,)
    training_params.data_loader=loader

    end_loss = dcgan.run_train(training_params, save=False, load=False)

    training_params.num_epochs = 100 #TODOs

    print('End Loss:', str(end_loss))
