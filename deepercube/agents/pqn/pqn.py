import torch
from deepercube.agents.nn.network import OneHot, DeepCubeACore
import argparse


class PQN:
    def __init__(
        self,
        args: argparse.Namespace,
        obs_bound: int,
        obs_size: int,
        actions: int,
        device: torch.device,
    ) -> None:
        self.device = device

        self.actions = actions
        self.clip_grad_norm = args.clip_grad_norm
        self.model = torch.nn.Sequential(
            OneHot(obs_bound),
            DeepCubeACore(obs_size * obs_bound, False, "layer"),
            torch.nn.Linear(1000, actions),
        ).to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), args.learning_rate)
        self.loss = torch.nn.MSELoss()

    def predict_egreedy_actions(
        self, states: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        batch_size = len(states)
        noise = torch.rand(batch_size, device=self.device) < epsilon
        random_actions = torch.randint(
            0, self.actions, (batch_size,), device=self.device
        )
        return noise * random_actions + ~noise * self.predict_q(states).argmax(-1)

    def predict_q(self, states: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(states)

    def train(
        self, states: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        self.model.train()

        predictions = self.model(states)[range(len(states)), actions]
        loss = self.loss(predictions, targets)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.opt.step()
