from ..nn.network import DeepCubeACore, OneHot
from ..nn.noisy_linear import NoisyLinear
from deepercube.utils import wrappers
import numpy as np
import copy
import torch


class RainbowNetwork(torch.nn.Module):
    def __init__(self, obs_bound: int, obs_size: int, actions: int) -> None:
        super().__init__()
        self.one_hot = OneHot(obs_bound)
        self.dca = DeepCubeACore(obs_size * obs_bound, True)
        self.output = NoisyLinear(1000, actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.one_hot(x).type(torch.float32)
        hidden = self.dca(hidden)
        output = self.output(hidden)
        return output


class Rainbow:
    """
    Rainbow agent class.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, obs_bound: int, obs_size: int, actions: int) -> None:
        super().__init__()
        self.netowrk = RainbowNetwork(obs_bound, obs_size, actions).to(self.device)
        self.target_network = copy.deepcopy(self.netowrk).to(self.device)

        self.opt = torch.optim.Adam(self.netowrk.parameters(), lr=0.000_0625)
        self.loss = torch.nn.MSELoss()

        self.copy_each = 100
        self.gamma = 1

    @wrappers.typed_torch_function(device, torch.int64, bool)
    def predict_moves(self, states: torch.Tensor, greedy: bool) -> torch.Tensor:
        if greedy:
            self.netowrk.eval()
        else:
            self.netowrk.train()
        with torch.no_grad():
            q_values = self.netowrk(states)
        return torch.argmax(q_values, -1)

    @wrappers.typed_torch_function(
        device,
        torch.int64,
        torch.int32,
        torch.float32,
        torch.bool,
        torch.bool,
        torch.int64,
    )
    def train(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_states: torch.Tensor,
    ) -> None:
        self.netowrk.train()
        self.target_network.train()

        with torch.no_grad():
            q_values_target: torch.Tensor = self.target_network(next_states)
            q_values = self.netowrk(next_states)
        next_actions = torch.argmax(q_values, dim=1)

        returns = (
            rewards
            + (~terminated)
            * self.gamma
            * q_values_target[torch.arange(len(q_values_target)), next_actions]
        )

        self.opt.zero_grad()
        predicted_q_values = self.netowrk(states)
        predicted_returns = predicted_q_values[torch.arange(len(states)), actions]
        loss = self.loss(predicted_returns, returns)

        loss.backward()
        self.opt.step()

    def copy_weights(self) -> None:
        self.netowrk.load_state_dict(self.target_network.state_dict())
