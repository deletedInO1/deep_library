import math

from presets.debug import DebugDatasets
from presets.models.GAN.PRO_GAN.ProGAN import ProGAN
from presets.training.GAN.ProGANTrainingParams import ProGANTrainingParams
from projects.test.TEST_CONFIG import *

import torch
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as trfs
import torchvision.datasets as datasets

import os.path as P



def get_data_loader(step):
    global net
    transforms = trfs.Compose([
        trfs.Resize((2**step * 4, 2**step * 4)),
        trfs.ToTensor(),
        trfs.RandomHorizontalFlip(p=0.5),
        trfs.Normalize(
            [0.5 for _ in range(net.image_channels)],
            [0.5 for _ in range(net.image_channels)],
        ),
    ])
    dataset = datasets.ImageFolder(location, transform=transforms)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=training_params.batch_sizes[step])
    return dataloader

def get_debug_data_loader(step):
    global net
    transforms = trfs.Compose([
        #trfs.ToPILImage(),
        trfs.Resize(2 ** step * 4),
        trfs.ToTensor(),
        trfs.RandomHorizontalFlip(0.5),
        trfs.Normalize(
            [0.5 for _ in range(net.image_channels)],
            [0.5 for _ in range(net.image_channels)],
        ),
    ])
    #dataset = DebugDatasets.ConstNumbers(1, 1000, transforms)
    dataset = datasets.ImageFolder(TO_ROOT + HORSE_LOCATION, transform=transforms)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=training_params.batch_sizes[step])
    return dataloader


#DEBUG:
def train_generator(gen):
    import torch.nn as nn
    import torch.optim as optim
    epochs = 30
    gen.train()

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(gen.parameters())
    for ep in range(epochs):
        for idx in range(1000):
            target = torch.ones((32, 3, 4, 4))
            noise = torch.randn((32, 100, 1, 1))
            y = gen(noise, alpha=1, steps=0)
            loss = loss_fn(y, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)

def train_generator_with_critic(gen, critic):
    import torch.nn as nn
    import torch.optim as optim
    epochs = 30
    gen.train()

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(gen.parameters())
    for ep in range(epochs):
        for idx in range(1000):
            noise = torch.randn((32, 100, 1, 1))
            y = gen(noise, alpha=1, steps=0)
            c = critic(y, alpha=1, steps=0)
            loss = loss_fn(c, torch.ones_like(c))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)


def train_critic(critic):
    import torch.nn as nn
    import torch.optim as optim
    epochs = 30

    critic.train()

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(critic.parameters(), 1e-3)


    for ep in range(epochs):
        #accuracy
        total_count = 0
        right = 0
        #loss
        total_loss = 0
        for idx in range(100):
            real = torch.ones((32, 3, 4, 4))
            fake = torch.rand((32, 3, 4, 4))
            c_real = critic(real, alpha=1, steps=0)
            c_fake = critic(fake, alpha=1, steps=0)

            assert c_real.shape == c_fake.shape
            assert c_real.shape == torch.Size([32, 1])
            #accuracy
            for i in range(c_real.shape[0]):
                c_r = c_real[i]
                c_f = c_fake[i]

                total_count  += 1
                if c_r.item() > c_f.item():
                    right += 1

            loss_real = loss_fn(c_real, torch.ones_like(c_real))
            loss_fake = loss_fn(c_fake, torch.zeros_like(c_fake))
            loss = loss_real + loss_fake

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("---------------------------------------------------------------------")
        print("average loss:", str(total_loss / total_count))
        print("accuracy:", str(right/total_count *100)+"%")
        if right/total_count > 0.75:
            break

if __name__ == "__main__":
    net = ProGAN()

    training_params = ProGANTrainingParams(net, get_data_loader=get_data_loader, resolution_count=7, critic_iterations=4)
    location = TO_ROOT + HORSE_LOCATION
    location = P.abspath(location)

    net.run_train(training_params, load=True, save=False)