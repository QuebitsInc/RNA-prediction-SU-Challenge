import fastai
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import os
import gc
import numpy as np
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from fastai.vision.all import *


@delegates(GradScaler)
class MixedPrecision(Callback):
    order = 10
    def __init__(self, **kwargs): self.kwargs = kwargs

    def before_fit(self):
        self.autocast, self.learn.scaler, self.scales = autocast(), GradScaler(**
                                                                               self.kwargs), L()

    def before_batch(self): self.autocast.__enter__()

    def after_pred(self):
        if next(flatten(self.pred)).dtype == torch.float16:
            self.learn.pred = to_float(self.pred)

    def after_loss(self): self.autocast.__exit__(None, None, None)

    def before_backward(
        self): self.learn.loss_grad = self.scaler.scale(self.loss_grad)

    def before_step(self):
        self.skipped = True
        self.scaler.step(self)
        if self.skipped:
            raise CancelStepException()
        self.scales.append(self.scaler.get_scale())

    def after_step(self): self.learn.scaler.update()

    @property
    def param_groups(self):
        return self.opt.param_groups

    def step(self, *args, **kwargs):
        self.skipped = False

    def after_fit(
        self): self.autocast, self.learn.scaler, self.scales = None, None, None


fastai.callback.fp16.MixedPrecision = MixedPrecision
