from presets.training.TrainingParams import TrainingParamas
from presets.optimize import OptimizedTrainingParamas

from presets.models.GAN.gan_var import GanVar

if __name__ == "__main__":

    tr = TrainingParamas(GanVar("a", "b"), "crit", "batch", "dal", "nep", "ctor_opt")
    opt = OptimizedTrainingParamas(tr, {"batch_size": 64})
    opt.optim.g = "edit"
    print(opt)