import torch
from typing import NamedTuple, Iterable, Tuple
import hmc.utils.torch_utils as tut


class TorchReplayData(NamedTuple):
    """
    Transitions from the environment stored as 6-tuple of batched tensors:
        - states: (B, state_shape), `Any`
        - actions: (B,), `int`
        - rewards: (B,), `float`
        - terminations: (B,), `bool`
        - truncations: (B,), `bool`
        - next_states: (B, state_shape), `Any`
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    next_states: torch.Tensor

    def batch_size(self) -> int:
        return len(self.states)

    def filter(self, mask: torch.Tensor) -> "TorchReplayData":
        return TorchReplayData(*(d[mask] for d in self))

    @staticmethod
    def concatenate(data: Iterable["TorchReplayData"]) -> "TorchReplayData":
        return TorchReplayData(*(torch.concatenate(d, dim=0) for d in zip(*data)))

    @staticmethod
    def empty() -> "TorchReplayData":
        return TorchReplayData(
            *((torch.tensor([], device=tut.get_torch_cube_device()),) * 6)
        )


class TorchReplayEpData(NamedTuple):
    """
    Episodes stored as 4-tuple of tensors:
        - states: (num_envs, ep_limit, state_shape), `Any`
        - actions: (num_envs, ep_limit), `int`
        - rewards: (num_envs, ep_limit), `float`
        - next_states: (num_envs, ep_limit state_shape), `Any`
        - lengths: (num_envs,), `int`
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    lengths: torch.Tensor

    @staticmethod
    def concatenate(data: Iterable["TorchReplayEpData"]) -> "TorchReplayEpData":
        return TorchReplayEpData(*(torch.concatenate(d, dim=0) for d in zip(*data)))

    def batch_size(self) -> int:
        return len(self.states)

    def get_device(self) -> torch.device:
        return self.states.device

    def get_ep_limit(self) -> int:
        return self.rewards.shape[1]

    def unroll(self) -> TorchReplayData:
        """
        TODO: `self.compute_returns(gamma)`
        """
        mask = (
            torch.arange(self.states.shape[1], device=tut.get_torch_cube_device())[
                None, :
            ]
            < self.lengths[:, None]
        )
        states = self.states[mask];
        actions = self.actions[mask]
        rewards = self.rewards[mask]
        next_states = self.next_states[mask]

        num_eps = self.states.shape[0]
        _all_eps = torch.arange(num_eps, device=tut.get_torch_cube_device())
        last_indices = self.lengths - 1

        # split states and goals in a dimension
        stategoal_size = states.shape[-1]
        goal_index = stategoal_size // 2
        last_next_states = self.next_states[_all_eps, last_indices]
        # goal reached
        terminated_last = torch.all(
            last_next_states[:, :goal_index]  # state
            == last_next_states[:, goal_index:],  # goal
            dim=1,
        )

        terminated = torch.full(
            self.rewards.shape,
            False,
            dtype=torch.bool,
            device=tut.get_torch_cube_device(),
        )
        terminated[_all_eps, last_indices] = terminated_last
        terminated = terminated[mask]

        # episode limit hit
        truncated = torch.full(
            self.rewards.shape,
            False,
            dtype=torch.bool,
            device=tut.get_torch_cube_device(),
        )
        truncated[:, -1] = True
        truncated = truncated[mask]

        return TorchReplayData(states, actions, rewards, terminated, truncated, next_states)


class TorchEpisodeBuffer:
    def __init__(
        self,
        num_envs: int,
        state_shape: torch.Size,
        max_ep_len: int,
        device: torch.device,
        state_dtype=None,
    ) -> None:
        padded_max_ep_len = max_ep_len + 1
        self.num_envs = num_envs
        self.device = device
        self.states = torch.zeros(
            [num_envs, padded_max_ep_len, *state_shape],
            dtype=state_dtype,
            device=self.device,
        )
        self.actions = torch.zeros(
            [num_envs, padded_max_ep_len], dtype=torch.int64, device=self.device
        )
        self.rewards = torch.zeros(
            [num_envs, padded_max_ep_len], dtype=torch.float32, device=self.device
        )
        self.next_states = torch.zeros(
            [num_envs, padded_max_ep_len, *state_shape],
            dtype=state_dtype,
            device=self.device,
        )
        self.next_indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.reseting = torch.full(
            [num_envs], False, dtype=torch.bool, device=self.device
        )

    def store_transitions(
        self,
        transitions: TorchReplayData,
    ) -> TorchReplayEpData:
        """
        Stores the newly observed transitions including those between episodes.
        It automatically takes care of auto-reset transitions.

        Params:
            transitions: batched transitions, see `TorchReplayData`

        Returns:
            ReplayEpData to store into replay buffer,
        """
        # store transition for each env
        _all = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.states[_all, self.next_indices] = transitions.states
        self.actions[_all, self.next_indices] = transitions.actions
        self.rewards[_all, self.next_indices] = transitions.rewards
        self.next_states[_all, self.next_indices] = transitions.next_states
        self.next_indices += 1

        # get all terminated or truncated episodes
        finished = (transitions.terminations | transitions.truncations) & ~self.reseting
        finished_ep = TorchReplayEpData(
            self.states[finished],
            self.actions[finished],
            self.rewards[finished],
            self.next_states[finished],
            self.next_indices[finished],
        )
        self.next_indices *= ~(finished | self.reseting)
        self.reseting = finished

        return finished_ep
