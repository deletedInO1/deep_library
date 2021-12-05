import torch
import optuna.exceptions

from presets.training.TrainingParams import TrainingParams
from presets import save_load as sl

import abc as abstract #abstract
from tqdm import tqdm

from presets.training.train_results import TrainResults


class Trainable(abstract.ABC):

    def run_train(self, training:TrainingParams, loss_count=1, save=True, load=True, trial=None):
        #loss_count: the last ...  losses will be used for average
        self.try_load(self, training.optim, load)
        training.step = 0
        losses = []
        self.test(training)
        train_results = None
        for epoch in range(training.num_epochs):
            l = self.train_epoch(training, epoch + 1)
            losses.append(l)
            if len(losses) > loss_count:
                losses = list(losses[-loss_count:])

            train_results = self.test(training)
            if trial is not None:
                self.report_optuna(trial, train_results, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            self.try_save(self, training.optim, save)

        print("losses: " + str(losses))
        #return sum(losses) / len(losses)
        return train_results

    def report_optuna(self, trial, train_results, epoch):
        pass

    def test(self, training):

        s = TrainResults(0, 0, 0)
        c = 0
        for data in training.test_loader:
            s = s + self.test_batch(data, training)
            c += 1
        if training.test_writer is not None:
            training.test_writer.add_scalar("loss", s.get_loss(), training.step)
            training.test_writer.add_scalar("accuracy", s.get_accuracy(), training.step)
        return s


    def try_load(self, model, optimizer, do_load, release=False):
        if do_load:
            try:
                sl.load_checkpoint(model, optimizer, file="latest.pt"if not release else "release.pt")
            except FileNotFoundError:
                pass

    def try_save(self, model, optimizer, do_save):
        if do_save:
            sl.save_checkpoint(model, optimizer)

    @abstract.abstractmethod
    def test_batch(self, batch, training_params) -> TrainResults:
        pass

    def train_epoch(self, training:TrainingParams, epoch):
        loop = tqdm(training.data_loader)
        losses = []
        for idx, data in enumerate(loop):
            l = self.train_batch(data, training, epoch, cycle=idx)
            losses.append(l)
            training.step += training.batch_size
        return sum(losses) / len(losses)

    @abstract.abstractmethod
    def train_batch(self, data, train: TrainingParams, epoch, cycle) -> float:
        raise NotImplementedError("abstract")