import optuna as op
from presets.training.train import TrainingParamas
from functools import partial
import torch.utils.tensorboard as tb

class OptimizedTrainingParamas(TrainingParamas):
    def __init__(self, training, params):
        super(OptimizedTrainingParamas, self).__init__(do_gen_writer=False)
        attr = self.training_attributes(training)
        for name in attr:
            value = None
            if name in params:
                value = params[name]
            else:
                value = getattr(training, name)
            setattr(self, name, value)
    def gen_writer(self):
        self.writer = tb.SummaryWriter(f'runs/tensorboard' + str(self.writer_idx(f'runs/')))

    def training_attributes(self, training):
        return [a for a in dir(training) if not a.startswith('__') and not callable(getattr(training, a))]


def optimize(gen_model, training, get_params, minimize=True, n_trials=10):
    study = op.create_study(direction="minimize" if minimize else "maximize")
    partial_objective = partial(objective, gen_model, training, get_params)
    study.optimize(partial_objective, n_trials=n_trials)

    print("best trial: ")
    trial = study.best_trial
    print(trial.values)
    print("best parameters: ")
    print(trial.params)



def objective(gen_model, orig_training:TrainingParamas, get_params, trial : op.trial.Trial):
    model = gen_model()
    all_losses = []
    training = OptimizedTrainingParamas(orig_training, get_params(trial, model))
    for i in range(5):
        training.gen_writer()
        l = model.run_train(training)
        all_losses.append(l)

    return sum(all_losses) / len(all_losses)

