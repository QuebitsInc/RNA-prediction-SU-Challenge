import torch
import torch.nn as nn


class AliBiPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, qlen, klen, device):
        alibi = torch.arange(klen, device=device).unsqueeze(
            0) - torch.arange(qlen, device=device).unsqueeze(1)
        alibi = alibi.abs().float()
        alibi = alibi / (self.dim ** 0.5)
        return alibi
