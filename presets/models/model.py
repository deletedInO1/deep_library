import torch
import torch.nn as nn
from tqdm import tqdm

import abc as abstract #abstract

class Model(abstract.ABC, nn.Module):
    def __init__(self):
        super(Model, self).__init__()



