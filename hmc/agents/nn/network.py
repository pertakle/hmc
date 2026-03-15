import torch
from hmc.utils import wrappers
from hmc.agents.nn.noisy_linear import NoisyLinear


class OneHot(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        oh = torch.nn.functional.one_hot(x, self.num_classes)
        out = oh.reshape(x.shape[0], -1).type(torch.float32)
        return out


class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Sum(torch.nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(self._dim)

class ResBlock(torch.nn.Module):
    def __init__(self, size: int, Linear: type, Norm: type, Activation: type, Norm2: type) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            Linear(size, size),
            Norm(size),
            Activation(),
            Linear(size, size),
            Norm2(size)
        )
        self.post_act = Activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x)
        h = x + h
        h = self.post_act(h)
        return h


class ResMLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        in_classes: int,
        n1: int,
        n2: int,
        nr: int,
        out_features: int,
        noisy: bool,
        norm: str | None = None,
        norm_last_only: bool = True,
        activation: str = "relu",
    ) -> None:
        """
        norm:
            - "layer"
            - "batch"
            - None (default)

        activation:
            - "relu" (default)
            - "silu" aka swish
        """
        super().__init__()

        Linear = NoisyLinear if noisy else torch.nn.Linear

        NormLast = Identity
        if norm is not None:
            if norm == "layer":
                NormLast = torch.nn.LayerNorm
            elif norm == "batch":
                NormLast = torch.nn.BatchNorm1d
            else:
                raise ValueError("Unknown normalization layer!")
        Norm = Identity if norm_last_only else NormLast

        if activation == "relu":
            Activation = torch.nn.ReLU
        elif activation == "silu":
            Activation = torch.nn.SiLU
        else:
            raise ValueError("Unknown activation function!")

        oh_out = in_features * in_classes
        has_l1 = n1 > 0
        l1_in = oh_out
        l1_out = n1

        has_l2 = n2 > 0
        l2_in = n1 if has_l1 else oh_out
        l2_out = n2

        out_in = l2_out if has_l2 else (l1_out if has_l1 else oh_out)

        assert has_l2 or nr == 0, "Res blocks must have l2."

        self.one_hot = OneHot(in_classes)

        self.l1 = torch.nn.Sequential(
            Linear(l1_in, l1_out),
            Norm(l1_out),
            Activation(),
        )
        self.l2 = torch.nn.Sequential(
            Linear(l2_in, l2_out),
            Norm(l2_out),
            Activation()
        ) if has_l2 else Identity()

        self.res_blocks = torch.nn.Sequential(
            *(
                ResBlock(n2, Linear, Norm, Activation, NormLast if i == nr - 1 else Norm)
                for i in range(nr)
            )
        )

        self.out = Linear(out_in, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.one_hot(x)
        hidden = self.l1(hidden)
        hidden = self.l2(hidden)
        hidden = self.res_blocks(hidden)
        out = self.out(hidden)
        return out


class RResMLP(torch.nn.Module):
    def __init__(self, network: ResMLP) -> None:
        out = network.out
        self.lstm = torch.nn.LSTMCell(out.in_features, out.out_features)
        network.out = Identity()  # type: ignore
        self.backbone = network

    def forward(
        self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        return self.lstm(hidden, (hx, cx))


class ResMLPPlus(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        in_classes: int,
        n1: int,
        n2: int,
        nr: int,
        actions: int,
        atoms: int,
        norm: str | None = None,
        norm_last_only: bool = True,
    ) -> None:
        super().__init__()
        self.actions = actions
        self.atoms = atoms
        self.value_head = ResMLP(
            in_features, in_classes, n1, n2, nr, atoms, True, norm, norm_last_only
        )
        self.adv_head = ResMLP(
            in_features, in_classes, n1, n2, nr, actions * atoms, True, norm, norm_last_only
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value_head(x).reshape(-1, 1, self.atoms)
        adv = self.adv_head(x).reshape(-1, self.actions, self.atoms)
        return value + adv - adv.mean(1, keepdim=True)


class DeepCubeACore(torch.nn.Module):
    """
    DeepCubeA network architecture without the output layer.

    Agostinelli, F., McAleer, S., Shmakov, A., & Baldi, P. (2019). Solving the Rubik’s cube with deep reinforcement learning and search. Nature Machine Intelligence, 1(8), 356–363.

    Params:
        `in_features`: number of input features
        `noisy`: if True, NoisyLinear layers will be used instad of Linear
        `norm`:
            - None: no normalization
            - "layer": LayerNorm
            - "batch": BatchNorm
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

    def __init__(
        self, in_features: int, noisy: bool = False, norm: str | None = None
    ) -> None:

        def res_block():
            nonlocal Linear
            nonlocal Norm
            return torch.nn.Sequential(
                Linear(1000, 1000, bias=False),
                Norm(1000),
                torch.nn.ReLU(),
                Linear(1000, 1000),
                Norm(1000),
            )

        super().__init__()
        Linear = NoisyLinear if noisy else torch.nn.Linear

        Norm = Identity
        if norm is not None:
            if norm == "layer":
                Norm = torch.nn.LayerNorm
            elif norm == "barch":
                torch.nn.BatchNorm1d
            else:
                assert False, "Unknown type of normalization!"

        self.relu = torch.nn.ReLU()

        self.l1 = Linear(in_features, 5000, bias=False)
        self.norm1 = Norm(5000)

        self.l2 = Linear(5000, 1000, bias=False)
        self.norm2 = Norm(1000)

        self.res_blocks = torch.nn.ParameterList([res_block() for _ in range(4)])

        if not noisy:
            self.apply(wrappers.torch_init_with_xavier_and_zeros)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.l1(x)
        hidden = self.norm1(hidden)
        hidden = self.relu(hidden)

        hidden = self.l2(hidden)
        hidden = self.norm2(hidden)
        hidden = self.relu(hidden)

        for res_block in self.res_blocks:
            skip = hidden
            res = res_block(hidden)
            hidden = res + skip
            hidden = self.relu(hidden)

        return hidden
