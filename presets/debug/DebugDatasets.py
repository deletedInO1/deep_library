import torch
from torch.utils.data import Dataset

class ConstNumbers(Dataset):
    def __init__(self, num, len, transform):
        self.num = num
        self.length = len
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = torch.ones((3, 256, 256)) * 1
        if self.transform:
            img = self.transform(img)
        return (img, 1)