import torch
import torch.nn.functional as F
import numpy as np
import os
import random


def loss(pred, target):
    p = pred[target['mask'][:, :pred.shape[1]]]
    y = target['react'][target['mask']].clip(0, 1)
    loss = F.l1_loss(p, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()

    return loss


def flatten(o):
    for item in o:
        if isinstance(o, dict):
            yield o[item]
            continue
        elif isinstance(item, str):
            yield item
            continue
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def dict_to(x, device='cuda'):
    return {k: x[k].to(device) for k in x}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
