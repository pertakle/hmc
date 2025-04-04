from ..nn.network import DeepCubeACore, OneHot
from ..nn.noisy_linear import NoisyLinear
from hmc.utils import wrappers
import numpy as np
import torch
import argparse


class Reshape(torch.nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.shape)

class DuelingDQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, atoms):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.atoms = atoms

        # Advantage stream
        self.advantage = torch.nn.Sequential(
            NoisyLinear(input_dim, hidden_size),
            torch.nn.ReLU(),
            NoisyLinear(hidden_size, self.output_dim * self.atoms),
        )

        # Value stream
        self.value = torch.nn.Sequential(
            NoisyLinear(input_dim, hidden_size),
            torch.nn.ReLU(),
            NoisyLinear(hidden_size, self.atoms),
        )

    def forward(self, x):
        #x = self.features(x)
        #x = x.view(x.size(0), -1)

        advantage = self.advantage(x)
        value = self.value(x)

        # Combine value and advantage streams
        value = value[:, None]  # [B, None, atom]
        advantage = advantage.view(
            -1, self.output_dim, self.atoms
        )  # [B, actions, atom]

        logits = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return logits


class Rainbow:
    """
    Rainbow agent class.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self, args: argparse.Namespace, obs_bound: int, obs_size: int, actions: int
    ) -> None:
        super().__init__()

        self.hidden_size = 512
        self.num_actions = actions
        self.num_atoms = args.atoms
        self.atoms_np, self.atom_delta = np.linspace(args.v_min, args.v_max, args.atoms, retstep=True)
        self.atoms_torch = torch.tensor(self.atoms_np, device=Rainbow.device)

        self.network = torch.nn.Sequential(
            OneHot(obs_bound),
            DeepCubeACore(obs_size * obs_bound, True),
            DuelingDQN(1000, actions, self.hidden_size, self.num_atoms),

            #NoisyLinear(obs_size * obs_bound, 1000),
            #NoisyLinear(1000, actions * self.num_atoms),
            Reshape((-1, actions, self.num_atoms))
        ).to(Rainbow.device)

        self.opt = torch.optim.Adam(self.network.parameters(), lr=args.learning_rate)
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def probs_to_expected(self, probs: np.ndarray) -> np.ndarray:
        return (probs * self.atoms_np[None, None]).sum(2)

    @wrappers.typed_torch_function(device, torch.int64, bool)
    def predict_probs(self, states: torch.Tensor, greedy: bool) -> torch.Tensor:
        if greedy:
            self.network.eval()
        else:
            self.network.train()
        with torch.no_grad():
            return self.network(states).softmax(2)

    def predict_q_values(self, states: np.ndarray, greedy: bool) -> np.ndarray:
        probs: np.ndarray = self.predict_probs(states, greedy)  # type: ignore
        return self.probs_to_expected(probs)

    @wrappers.typed_torch_function(
        device,
        torch.int64,
        torch.int32,
        torch.float32,
        torch.float32,
    )
    def train(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        target: torch.Tensor,
        isw: torch.Tensor,
    ) -> torch.Tensor:
        self.network.train()
        predictions = self.network(states)
        #predictions = predictions.view(-1, self.num_actions, self.num_atoms)
        predictions = predictions[torch.arange(len(predictions), device=states.device), actions]
        # predictions = predictions.log_softmax(-1)

        loss = self.loss(predictions, target)
        self.opt.zero_grad()
        # loss = loss.sum(1)
        loss_reduced = loss @ isw
        loss_reduced.backward()
        with torch.no_grad():
            self.opt.step()
        return loss

    def copy_weights_from(self, other: "Rainbow", tau: float) -> None:
        # self._model.load_state_dict(other._model.state_dict())
        for target_param, param in zip(
            self.network.parameters(), other.network.parameters()
        ):
            target_param.data.mul_(1 - tau)
            target_param.data.add_(tau * param.data)
