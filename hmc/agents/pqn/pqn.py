import torch
import hmc.agents.rl_utils.torch_buffers as tbuf
from hmc.agents.nn.network import ResMLP
from hmc.problems.baseproblem import Problem
import argparse
import hmc.kostka.torch_cube_vec as tcv
from hmc.utils.wrappers import torch_init_with_orthogonal_and_zeros
from hmc.agents.rl_utils.torch_buffers import TorchReplayEpData


def unroll_train_data(
        episodes: tbuf.TorchReplayEpData, returns: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns batched states, actions, returns trainable tripplet."""
    # unroll returns
    mask = (
        torch.arange(episodes.states.shape[1], device=returns.device)[
            None, :
        ]
        < episodes.lengths[:, None]
    )
    returns_unrolled = returns[mask]
    episodes_unrolled = episodes.unroll()
    states_unrolled = episodes_unrolled.states
    actions_unrolled = episodes_unrolled.actions

    return states_unrolled, actions_unrolled, returns_unrolled


class PQN:
    def __init__(
        self,
        args: argparse.Namespace,
        obs_bound: int,
        obs_size: int,
        num_actions: int,
        noisy: bool,
        device: torch.device,
    ) -> None:
        self.device = device

        self.num_actions = num_actions
        self.noisy = noisy
        self.clip_grad_norm = args.clip_grad_norm
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.gamma = args.gamma
        self.lambd = args.lambd
        self.model = (
            ResMLP(
                obs_size,
                obs_bound,
                args.n1,
                args.n2,
                args.nr,
                num_actions,
                noisy=noisy,
                norm="layer",
                # norm_last_only=True,
                norm_last_only=False,
            )
            .apply(torch_init_with_orthogonal_and_zeros)
            .to(self.device)
        )

        self.opt = torch.optim.AdamW(
            self.model.parameters(), args.learning_rate, weight_decay=args.l2
        )
        self.loss = torch.nn.MSELoss()

    def _predict_noisy_actions(
        self, states: torch.Tensor, greedy: bool
    ) -> torch.Tensor:
        if greedy:
            self.model.train()
        else:
            self.model.eval()
        return self.predict_q(states).argmax(-1)

    def _predict_egreedy_actions(
        self, states: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        self.model.eval()
        batch_size = len(states)
        noise = torch.rand(batch_size, device=self.device) < epsilon
        random_actions = torch.randint(
            0, self.num_actions, (batch_size,), device=self.device
        )
        return noise * random_actions + ~noise * self.predict_q(states).argmax(-1)

    def predict_actions(self, states: torch.Tensor, epsilon: float) -> torch.Tensor:
        if self.noisy:
            return self._predict_noisy_actions(states, epsilon > 0)
        else:
            return self._predict_egreedy_actions(states, epsilon)

    def _successors(self, stategoals: torch.Tensor) -> torch.Tensor:
        B = stategoals.shape[0]
        F = stategoals.shape[1] // 2
        states = stategoals[:, :F]
        goals = stategoals[:, F:]

        successors = tcv.make_all_moves_vec(states.reshape(B, 6, 3, 3)).reshape(
            B, -1, 6, 3, 3
        )
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

    def make_returns(self, episodes: TorchReplayEpData) -> torch.Tensor:
        B = episodes.batch_size()
        T = episodes.states.shape[1]
        next_q = self.predict_q(episodes.next_states.reshape(B * T, -1)).reshape(B, T, -1)
        next_v = next_q.max(-1).values

        assert torch.all(episodes.lengths > 0)
        last_next_stategoals = episodes.next_states[range(B), episodes.lengths - 1].reshape(B, 2, -1)
        end_terminated = torch.all(last_next_stategoals[:, 0] == last_next_stategoals[:, 1], dim=1)

        returns = torch.zeros_like(episodes.rewards)

        # last possible transition manually
        # [-1] is either the last -> we use ~end_terminated
        #      or it is not valid -> we do not care anyway about this value
        returns[:, -1] = episodes.rewards[:, -1] + ~end_terminated * self.gamma * next_v[:, -1]

        for t in range(T - 2, -1, -1):
            # this transition ended/terminated
            dones = (t + 1) == episodes.lengths
            terminations = dones & end_terminated


            #       terminated: 0                                    -->  handled below
            # truncated ~ done: next_v[t]                            -->  ll = 1
            #            !done: (1 - l) * R[t+1] + l * next_v[:, t]  -->  ll = l
            ll = self.lambd * ~dones + dones  # typechecker is wrong: `dones` is always a Tensor, not a bool
            gl_next = (1 - ll) * returns[:, t + 1] + ll * next_v[:, t]
            returns[:, t] = episodes.rewards[:, t] + ~terminations * self.gamma * gl_next
        return returns

    def _make_train_data(
        self, episodes: tbuf.TorchReplayEpData
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        returns = self.make_returns(episodes)
        return unroll_train_data(episodes, returns)

    def train(self, episodes: tbuf.TorchReplayEpData) -> None:
        if episodes.unrolled_batch_size() < self.batch_size:
            print("Warning: not enough data for a batch, no train step!")

        for _ in range(self.epochs):
            # recomputing targets each episode similarly to PPO
            # although the PQN paper does not do this
            data = self._make_train_data(episodes)
            self._train_epoch(*data)

    def _train_epoch(
        self, states: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        data_size = len(states)

        indices = torch.randperm(data_size, dtype=torch.long, device=self.device)
        for i in range(0, data_size - self.batch_size + 1, self.batch_size):  # includes full batch sizes only
        # for i in range(0, data_size, self.batch_size):  # includes last smaller batch
            batch_indices = indices[i : i + self.batch_size]

            self._train_step(
                states[batch_indices],
                actions[batch_indices],
                targets[batch_indices],
            )

    def _train_step(
        self, states: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        self.model.train()

        predictions = self.model(states)[range(len(states)), actions]
        loss = self.loss(predictions, targets)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.opt.step()
