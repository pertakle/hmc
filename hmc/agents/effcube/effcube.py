import argparse
import torch
from hmc.agents.nn.network import ResMLP
from hmc.utils.wrappers import torch_init_with_orthogonal_and_zeros
from hmc.problems.baseproblem import Problem


class EffCube:

    def __init__(
        self,
        args: argparse.Namespace,
        problem: Problem,
        obs_bound: int,
        obs_size: int,
        num_actions: int,
        device: torch.device,
    ) -> None:

        self.problem = problem

        self.network = ResMLP(
            obs_size,
            obs_bound,
            args.n1,
            args.n2,
            args.nr,
            num_actions,
            noisy=False,
            norm="layer",
            norm_last_only=False,
        ).apply(torch_init_with_orthogonal_and_zeros).to(device)
        self.opt = torch.optim.Adam(self.network.parameters(), args.learning_rate)
        self.loss = torch.nn.CrossEntropyLoss()

    def predict_action_probs(self, states: torch.Tensor) -> torch.Tensor:
        """Runs a batched diffusion process and returns the final states."""
        self.network.eval()
        with torch.no_grad():
            logits = self.network(states)
            return torch.softmax(logits, -1)

    def predict_action(self, states: torch.Tensor, greedy: bool) -> torch.Tensor:
        probs = self.predict_action_probs(states)
        if greedy:
            actions = probs.argmax(-1)
        else:
            distributions = torch.distributions.Categorical(probs=probs)
            actions = distributions.sample()
        return actions


    def train(self, states: torch.Tensor, targets: torch.Tensor) -> None:
        """Performs a single training step with the parameters.

        The time dim (1st) is sorted from T down to 1.

        Params:
            - states: (T, B, S)
            - targets: (T, B)
        """
        states = states.flatten(0, 1)
        targets = targets.flatten()

        self.network.train()
        self.opt.zero_grad()
        logits = self.network(states)
        loss = self.loss(logits, targets)
        loss.backward()
        self.opt.step()
