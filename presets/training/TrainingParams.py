from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import os
import torch.utils.tensorboard as tb

class TrainingParams:
    def __init__(self, optim=None, criterion=None, batch_size :int=None, data_loader :DataLoader=None, num_epochs :int=None, do_gen_writer=True):
        self.optim = optim
        self.criterion = criterion
        self.batch_size = batch_size
        self.i_disc = 0
        self.i_gen = 1
        self.data_loader = data_loader
        self.test_loader = None
        self.num_epochs = num_epochs
        self.writer = tb.SummaryWriter(f'runs/tensorboard'+str(self.writer_idx(f'runs/'))+'_train') if do_gen_writer else None
        self.test_writer = tb.SummaryWriter(f'runs/tensorboard'+str(self.writer_idx(f'runs/'))+'_test') if do_gen_writer else None
        self.step = 0

    def writer_idx(self, path):
        if os.path.exists(path):
            return len(os.listdir(path)) // 2
        return 0
