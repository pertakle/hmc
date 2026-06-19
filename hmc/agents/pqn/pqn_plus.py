import torch
import hmc.agents.rl_utils.torch_buffers as tbuf
from hmc.agents.pqn.pqn import unroll_train_data
from hmc.agents.nn.network import ResMLPPlus
from hmc.problems.baseproblem import Problem
import argparse
import hmc.kostka.torch_cube_vec as tcv
from hmc.utils.wrappers import torch_init_with_orthogonal_and_zeros
from hmc.agents.rl_utils.torch_buffers import TorchReplayEpData


class PQNPlus:
    def __init__(
        self,
        args: argparse.Namespace,
        obs_bound: int,
        obs_size: int,
        num_actions: int,
        device: torch.device,
    ) -> None:
        self.device = device

        self.num_atoms = args.atoms
        assert args.atoms >= 2
        assert args.v_min < args.v_max
        self.atoms = torch.linspace(
            args.v_min, args.v_max, args.atoms, dtype=torch.float32, device=device
        ) 
        self.atom_delta = self.atoms[1] - self.atoms[0]
        self.num_actions = num_actions
        self.clip_grad_norm = args.clip_grad_norm
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.gamma = args.gamma
        self.lambd = args.lambd
        self.model = (
            ResMLPPlus(
                obs_size,
                obs_bound,
                args.n1,
                args.n2,
                args.nr,
                num_actions,
                args.atoms,
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
        self.loss = torch.nn.CrossEntropyLoss()

    def predict_actions(self, states: torch.Tensor, epsilon: float) -> torch.Tensor:
        if epsilon > 0:
            self.model.train()
        else:
            self.model.eval()
        return self.predict_q(states).argmax(-1)

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

    def predict_dist(self, states: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(states)
            return torch.softmax(logits, -1)

    def predict_q(self, states: torch.Tensor) -> torch.Tensor:
        dists = self.predict_dist(states)
        expected_q = self._expected_q(dists)
        return expected_q

    def _expected_q(self, dist: torch.Tensor) -> torch.Tensor:
        return dist @ self.atoms

    def _project_on_support(
        self,
        dist: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        """Projects distribution onto self.atoms support.

        Params:
            - dist: [..., A]
            - support: [..., A]
        """
        assert (
            dist.shape == support.shape
        ), f"Inconsistent distribution and support shapes! ({dist.shape} != {support.shape})"
        assert (
            dist.shape[-1] == self.atoms.shape[-1]
        ), f"Invalid distribution size! (expected {self.atoms.shape[-1]}, got {dist.shape[-1]})"
        assert (
            support.shape[-1] == self.atoms.shape[-1]
        ), f"Invalid support size! (expected {self.atoms.shape[-1]}, got {support.shape[-1]})"

        normed = (support.clip(self.atoms[0], self.atoms[-1]) - self.atoms[0]) / self.atom_delta
        lower_normed_atoms = torch.floor(normed).long().clip(0, self.num_atoms - 1)
        upper_normed_atoms = torch.ceil(normed).long().clip(0, self.num_atoms - 1)

        upper_mass = normed - lower_normed_atoms + (lower_normed_atoms == upper_normed_atoms)
        lower_mass = upper_normed_atoms - normed

        new_dist = torch.zeros_like(dist)
        # B = sum(dist.shape[:-1])
        B = dist.view(-1, dist.shape[-1]).shape[0]
        # print(dist.shape, B)
        # exit()
        offsets = torch.linspace(0, (B - 1) * self.num_atoms, B, device=self.device, dtype=torch.long)[..., None]

        new_dist.view(-1).index_add_(0, (offsets + lower_normed_atoms).view(-1), lower_mass.view(-1) * dist.view(-1))
        new_dist.view(-1).index_add_(0, (offsets + upper_normed_atoms).view(-1), upper_mass.view(-1) * dist.view(-1))

        return new_dist

    def make_returns(self, episodes: TorchReplayEpData) -> torch.Tensor:
        T = episodes.states.shape[1]
        E = episodes.states.shape[0]
        B = episodes.batch_size()

        next_q_dist = self.predict_dist(
            episodes.next_states.reshape(B * T, -1)
        ).reshape(B, T, self.num_actions, self.num_atoms)
        next_q = self._expected_q(next_q_dist)
        next_v, next_a_opt = next_q.max(-1)

        assert torch.all(episodes.lengths > 0)
        last_next_stategoals = episodes.next_states[
            range(B), episodes.lengths - 1
        ].reshape(B, 2, -1)
        end_terminated = torch.all(
            last_next_stategoals[:, 0] == last_next_stategoals[:, 1], dim=1
        )

        returns_dist = torch.zeros(
            [E, T, self.num_atoms], dtype=torch.float32, device=self.device
        )

        # last possible transition manually
        # [-1] is either the last -> we use ~end_terminated
        #      or it is not valid -> we do not care anyway about this value
        # returns[:, -1] = episodes.rewards[:, -1] + ~end_terminated * gamma * next_v[:, -1]
        _rang = torch.arange(next_q_dist.shape[0], device=self.device)
        next_v_dist_t = next_q_dist[_rang, -1, next_a_opt[:, -1]]
        returns_dist[:, -1] = self._project_on_support(
            next_v_dist_t,
            # episodes.rewards[:, -1] + ~end_terminated * gamma * self.atoms,
            # [B]                         [B]            []         [A]
            # chci [B, A]
            # [B, 1]                         [B, 1]            []         [1, A]
            episodes.rewards[:, -1][:, None] + ~end_terminated[:, None] * self.gamma * self.atoms[None],
        )

        for t in range(T - 2, -1, -1):
            # this transition ended/terminated
            dones = (t + 1) == episodes.lengths
            terminations = dones & end_terminated


            #       terminated: 0                                    -->  handled below
            # truncated ~ done: next_v[t]                            -->  ll = 1
            #            !done: (1 - l) * R[t+1] + l * next_v[:, t]  -->  ll = l
            ll = self.lambd * ~dones + dones  # typechecker is wrong: `dones` is always a Tensor, not a bool
            next_v_dist_t = next_q_dist[_rang, t, next_a_opt[:, t]]

            gl_next_dist_t = (1 - ll[:, None]) * returns_dist[:, t + 1] + ll[:, None] * next_v_dist_t  # lambda-return

            returns_dist[:, t] = self._project_on_support(
                gl_next_dist_t,
                episodes.rewards[:, t][:, None] + ~terminations[:, None] * self.gamma * self.atoms[None],
            )
        return returns_dist

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

        logits = self.model(states)[range(len(states)), actions]
        loss = self.loss(logits, targets)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.opt.step()
