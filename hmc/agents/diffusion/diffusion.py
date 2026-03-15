import argparse
import torch
from hmc.agents.nn.network import ResMLP
from hmc.utils.wrappers import torch_init_with_orthogonal_and_zeros
from hmc.agents.diffusion.noop_problem_wrapper import NoOpProblemWrapper


class Diffusion:

    def __init__(
        self,
        args: argparse.Namespace,
        problem: NoOpProblemWrapper,
        obs_bound: int,
        obs_size: int,
        num_actions: int,
        steps: int,
        device: torch.device,
    ) -> None:

        self.problem = problem
        # self.rng = torch.random.get_rng_state()  # seed is set in the main.py

        self.networks = (
            torch.nn.ParameterList(
                [
                    # new_diffusion_step(obs_bound, obs_size, num_actions, args.n1) for _ in range(steps)
                    ResMLP(
                        obs_size,
                        obs_bound,
                        args.n1,
                        args.n2,
                        args.nr,
                        num_actions,
                        noisy=False,
                        norm="layer",
                        norm_last_only=False,
                    )
                    for _ in range(steps)
                ]
            )
            .apply(torch_init_with_orthogonal_and_zeros)
            .to(device)
        )
        self.opt = torch.optim.Adam(self.networks.parameters(), args.learning_rate)
        self.loss = torch.nn.CrossEntropyLoss()

    def predict(self, states: torch.Tensor, greedy: bool) -> torch.Tensor:
        """Runs a batched diffusion process and returns the final states."""
        self.networks.eval()
        with torch.no_grad():
            for diff_network in self.networks:
                logits = diff_network(states)
                if greedy:
                    actions = logits.argmax(-1)
                else:
                    distributions = torch.distributions.Categorical(logits=logits)
                    actions = distributions.sample()
                self.problem.perform_action_vec(states, actions)
            return states

    def train(self, states: torch.Tensor, targets: torch.Tensor) -> None:
        """Performs a single training step with the parameters.

        The time dim (1st) is sorted from T down to 1.

        Params:
            - states: (T, B, S)
            - targets: (T, B)
        """
        self.networks.train()
        self.opt.zero_grad()
        loss = torch.tensor(0)
        for states_i, targets_i, network_i in zip(states, targets, self.networks):
            logits_i = network_i(states_i)
            loss_i = self.loss(logits_i, targets_i)
            loss = loss + loss_i
        loss.backward()
        self.opt.step()
