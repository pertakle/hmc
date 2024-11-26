from ..nn.network import Network
from deepercube.utils import wrappers
import numpy as np
import copy
import torch


class Rainbow:
    """
    Rainbow agent class.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, obs_bound: int, obs_size: int, actions: int) -> None:
        super().__init__()
        self.model = Network(obs_bound, obs_size, actions).to(self.device)
        self.model.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.target_network = copy.deepcopy(self.model).to(self.device)
        self.rng = torch.Generator()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss = torch.nn.MSELoss()

        self.copy_each = 100

    @wrappers.typed_torch_function(device, torch.int32, float)
    def predict_moves(self, states: torch.Tensor) -> torch.Tensor:  # type: ignore
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(states)
        return torch.argmax(q_values, -1)

    @wrappers.typed_torch_function(device, torch.float32)
    def train(
        self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def copy_weights(self) -> None:
        self.model.load_state_dict(self.target_network.state_dict())
