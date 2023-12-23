import torch
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
import os
from utils.logger import *


def build_opti_sche(base_model, optimizer_type, weight_decay, lr):
    if optimizer_type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]

        param_groups = add_weight_decay(base_model, weight_decay=weight_decay)
        optimizer = optim.AdamW(param_groups, lr=lr)
        #optimizer = optim.AdamW(base_model.parameters(), lr=lr)

    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=True)
    else:
        raise NotImplementedError()

    scheduler = CosineLRScheduler(optimizer,
                t_initial=50,
                lr_min=1e-6,
                cycle_decay=0.1,
                warmup_lr_init=1e-6,
                warmup_t=0,
                cycle_limit=1,
                t_in_epochs=True)

    return optimizer, scheduler

def save_checkpoint(base_model, optimizer, epoch, prefix, experiment_path, logger = None):
    torch.save({
                'base_model' : base_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch' : epoch,
                }, os.path.join(experiment_path, prefix + '.pth'))
    print_log(f"Save checkpoint at {os.path.join(experiment_path, prefix + '.pth')}", logger = logger)