import torch
from hmc.agents.nn.network import ResMLP
import argparse
import hmc.kostka.torch_cube_vec as tcv
from hmc.utils.wrappers import torch_init_with_orthogonal_and_zeros

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
        self.model = ResMLP(
            obs_size, obs_bound, 1024, 1024, 1, actions,
            noisy=False, norm="layer", norm_last_only=True
        ).apply(torch_init_with_orthogonal_and_zeros).to(self.device)

        self.opt = torch.optim.AdamW(self.model.parameters(), args.learning_rate, weight_decay=0.0001)
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

    def _successors(self, stategoals: torch.Tensor) -> torch.Tensor:
        B = stategoals.shape[0]
        F = stategoals.shape[1] // 2
        states = stategoals[:, :F]
        goals = stategoals[:, F:]

        successors = tcv.make_all_moves_vec(states.reshape(B, 6, 3, 3)).reshape(B, -1, 6, 3, 3)
        ssg = torch.concat((successors, goals[:, None]), dim=1)
        return ssg

    def predict_q(self, states: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            # B = states.shape[0]
            # FF = states.shape[1]
            # succ = self._successors(states).reshape(-1, FF)
            # return self.model(succ).reshape(B, -1, FF)
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
