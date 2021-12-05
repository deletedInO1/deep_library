import torch
import torch.nn

import os
import os.path as P


folder_id = None

def save_checkpoint(model : torch.nn.Module, optim, path='checkpoints/', is_gan=False):
    if is_gan:
        d = {
            "model": {
                "d": model.discriminator.state_dict(),
                "g": model.generator.state_dict(),
            },
            "optimizer": {
                "d": optim.d.state_dict(),
                "g": optim.g.state_dict(),
            },
        }
    else:
        d = {
            "model": model.state_dict(),
            "optimizer": optim.state_dict()
        }

    global folder_id
    if folder_id is None:
        folder_id = next_id_str(path, False)
    name = 'c' + next_id_str(P.join(path, folder_id), True) + '.pt'
    save_obj(d, P.join(P.join(path, folder_id), name))

    save_obj(d, P.join(path, "latest.pt"))


def load_checkpoint(model, optim, file="latest.pt", dir_path="checkpoints/", is_gan=False):
    checkpoint = load_obj(P.join(dir_path, file))
    cm = checkpoint["model"]
    co = checkpoint["optimizer"]
    if is_gan:
        model.discriminator.load_state_dict(cm["d"])
        model.generator.load_state_dict(cm["g"])
        optim.d.load_state_dict(co["d"])
        optim.g.load_state_dict(co["g"])
    else:
        model.load_state_dict(cm)
        optim.load_state_dict(co)

def save_obj(obj, path):
    torch.save(obj, path)

def load_obj(path):
    return torch.load(path)

def next_id_str(path, cf):
    id = next_id(path, cf)
    return '0'*(2-len(str(id))) + str(id)

def next_id(path, count_files):
    if not P.exists(path):
        os.mkdir(path)
        return 0
    if count_files:
        return len(os.listdir(path))
    else:
        return len([name for name in os.listdir(path) if P.isdir(P.join(path, name))])