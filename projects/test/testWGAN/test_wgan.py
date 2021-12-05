from presets.models.GAN.WGAN.WGAN import WGan
from presets.training.GAN.WGANTrainingParams import WGANTrainingParams
from presets.models.GAN.gan_var import GanVar

import torch.optim as optim
import torchvision.transforms as trfs
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import projects.test.TEST_CONFIG as conf


if __name__ == "__main__":
    wgan = WGan(gp=True,  img_channels=1, features=GanVar(8,8))
    img_size = 64
    transforms = trfs.Compose([
        trfs.Resize(wgan.image_size),
        trfs.ToTensor(),
        trfs.Normalize(
            [0.5 for _ in range(wgan.image_channels)], #mean
            [0.5 for _ in range(wgan.image_channels)] #std
        ),
    ])
    #dataset = datasets.MNIST(root=conf.TO_ROOT+conf.MNIST_LOCATION, train=True, download=True, transform=transforms)

    training_params = WGANTrainingParams(gradient_penalty=wgan.gp, model=wgan, data_loader=None)
    loader = DataLoader(dataset, batch_size= training_params.batch_size, shuffle=True,)
    training_params.data_loader=loader

    end_loss = wgan.run_train(training_params)

    training_params.num_epochs = 100 #TODOs

    print('End Loss:', str(end_loss))
