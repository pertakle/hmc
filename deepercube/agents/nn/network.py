import torch
from deepercube.utils import wrappers
from deepercube.agents.nn.noisy_linear import NoisyLinear


class OneHot(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        oh = torch.nn.functional.one_hot(x, self.num_classes)
        return oh.reshape(x.shape[0], -1)


class DeepCubeACore(torch.nn.Module):
    """
    DeepCubeA network architecture without the output layer.

    Agostinelli, F., McAleer, S., Shmakov, A., & Baldi, P. (2019). Solving the Rubik’s cube with deep reinforcement learning and search. Nature Machine Intelligence, 1(8), 356–363.

    Params:
        `in_features`: number of input features
        `noisy`: if True, NoisyLinear layers will be used instad of Linear

    Returns:
        DeepCubeA with `in_features` inputs and 1000 outputs.
        User should add custom last layer on top of this core model.

    Examples:
        ```python
        model = torch.nn.Sequential(
            OneHot(6),
            DeepCubeACore(6*54, noisy=False),
            torch.nn.Linear(1000, 6)
        )
        ```
    """

    def __init__(self, in_features: int, noisy: bool = False) -> None:

        def res_block():
            nonlocal Linear
            return torch.nn.Sequential(
                Linear(1000, 1000, bias=False),
                torch.nn.BatchNorm1d(1000),
                torch.nn.ReLU(),
                Linear(1000, 1000),
                torch.nn.BatchNorm1d(1000),
            )

        super().__init__()
        Linear = NoisyLinear if noisy else torch.nn.Linear

        self.relu = torch.nn.ReLU()

        self.l1 = Linear(in_features, 5000, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(5000)

        self.l2 = Linear(5000, 1000, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(1000)

        self.res_blocks = torch.nn.ParameterList([res_block() for _ in range(4)])

        if not noisy:
            self.apply(wrappers.torch_init_with_xavier_and_zeros)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        return hidden
