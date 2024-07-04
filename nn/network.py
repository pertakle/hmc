import torch
from utils import wrappers
import numpy as np

def res_block():
    return torch.nn.Sequential(
        torch.nn.Linear(1000, 1000, bias=False),
        torch.nn.BatchNorm1d(1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 1000),
        torch.nn.BatchNorm1d(1000)
    )
# https://github.com/forestagostinelli/DeepCubeA/blob/master/utils/pytorch_models.py

class Network(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()

        self.relu = torch.nn.ReLU()

        self.l1 = torch.nn.Linear(in_size, 5000, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(5000)

        self.l2 = torch.nn.Linear(5000, 1000, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(1000)

        self.res_blocks = torch.nn.ParameterList([res_block() for _ in range(4)])

        self.out = torch.nn.Linear(1000, out_size)
        self.apply(wrappers.torch_init_with_xavier_and_zeros)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = torch.nn.functional.one_hot(x.reshape([B, -1]).type(torch.long), 6).reshape([B, -1]).type(torch.float32)
        hidden = self.l1(x)
        hidden = self.bn1(hidden)
        hidden = self.relu(hidden)

        hidden = self.l2(hidden)
        hidden = self.bn2(hidden)
        hidden = self.relu(hidden)

        for res_block in self.res_blocks:
            skip = hidden
            res = res_block(hidden)
            hidden = res + skip
            hidden = self.relu(hidden)

        out = self.out(hidden)
        return out

