import torch
from hmc.agents.nn.network import ResMLP
from hmc.utils.wrappers import torch_init_with_orthogonal_and_zeros
import argparse


class CayleyPy:

    def __init__(
        self,
        args: argparse.Namespace,
        obs_bound: int,
        obs_size: int,
        num_actions: int,
        device: torch.device,
    ) -> None:
        self.num_actions = num_actions
        self.model = ResMLP(
            obs_size, obs_bound, args.n1, args.n2, args.nr, 1,
            noisy=False, norm="batch", norm_last_only=False
        ).apply(torch_init_with_orthogonal_and_zeros).to(device)

        self.opt = torch.optim.Adam(self.model.parameters(), args.learning_rate)
        # self.opt = torch.optim.AdamW(self.model.parameters(), args.learning_rate, weight_decay=0.001)
        self.loss = torch.nn.MSELoss()

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def train(self, x, target):
        self.model.train()

        predictions = self.model(x).squeeze()
        loss = self.loss(predictions, target)

        self.opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            self.opt.step()
            
