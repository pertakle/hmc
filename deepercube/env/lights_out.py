from . import base_env
import gymnasium as gym
import torch

class LightsOut(base_env.BaseTorchEnv):
    def __init__(self, size: int, scramble_len: int, ep_limit: int, device=None) -> None:
        self._size = size
        super().__init__(scramble_len, ep_limit, device)

        self.observation_space = gym.spaces.MultiBinary(self._size**2)
        self.action_space = gym.spaces.Discrete(self._size**2)

    def _new_state(self) -> torch.Tensor:
        return torch.zeros([self._size, self._size], dtype=torch.long, device=self._device)

    def _scramble(self, state: torch.Tensor) -> None:
        for action in torch.randint(0, self._size**2, [self._scramble_len], device=self._device):
            self._perform_action(action, state)

    def _perform_action(self, action: int, state: torch.Tensor) -> None:
        row = action // self._size
        col = action % self._size
        min_row = max(0, row - 1)
        max_row = min(self._size, row + 2)
        min_col = max(0, col - 1)
        max_col = min(self._size, col + 2)
        state[min_row : max_row, min_col : max_col] = 1 - state[min_row : max_row, min_col : max_col]

    def _is_solved(self) -> bool:
        return bool(torch.all(self._state == self._goal).item())

    def print(self) -> None:
        import numpy as np
        letters = np.array(["O", "X"])
        print("State:")
        print(letters[self._state.cpu().numpy()])
        print("Goal:")
        print(letters[self._goal.cpu().numpy()])


class LightsOutVec(base_env.BaseTorchEnvVec):
    def __init__(self, size: int, num_envs: int, scramble_len: int, ep_limit: int, device=None) -> None:
        self._size = size
        super().__init__(num_envs, scramble_len, ep_limit, device)
        self.single_observation_space = gym.spaces.MultiBinary(self._size**2)
        self.single_action_space = gym.spaces.Discrete(self._size**2)
        self.observation_space = gym.spaces.MultiBinary([num_envs, self._size**2])
        self.action_space = gym.spaces.MultiDiscrete([num_envs, self._size**2])

        self._row_shift = torch.as_tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=torch.long, device=device)
        self._col_shift = torch.as_tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=torch.long, device=device)

    def _new_states(self, num_states: int) -> torch.Tensor:
        """Returns a batch of `num_states` new states."""
        return torch.zeros([num_states, self._size, self._size], dtype=torch.long, device=self._device)

    # def _is_solved(self) -> torch.Tensor:
        # """
        # Returns a vector of bools whether each state in `self._states` reached it's `self._goals`.
        # """
        # return torch.all(self._states == self._goals, dim=list(range(1, len(self._states.shape))))

    def _scramble(self, states: torch.Tensor) -> None:
        """Scrambles `states` in-place with `self._scramble_len` random moves."""
        for actions in torch.randint(0, self._size**2, [self._scramble_len, len(states)], device=self._device):
            self._perform_actions(actions, states)
        # scramble = torch.randint_like(states, 2)
        # states[:] = states ^ scramble

    def _perform_actions(self, actions: torch.Tensor, states: torch.Tensor) -> None:
        """Performs corresponding actions on each state in `self._state`."""
        rows = actions // self._size
        cols = actions % self._size

        rows_indices = torch.clip(rows[:, None] + self._row_shift[None], 0, self._size - 1)
        cols_indices = torch.clip(cols[:, None] + self._col_shift[None], 0, self._size - 1)
        b_range = torch.arange(len(actions), device=self._device)

        states[b_range[:, None], rows_indices, cols_indices] = (
            1 - states[b_range[:, None], rows_indices, cols_indices]
        )


    def print(self) -> None:
        import numpy as np
        letters = np.array(["O", "X"])
        print("\n============\n")
        for state, g in zip(self._states, self._goals):
            print("State:")
            print(letters[state.cpu().numpy()])
            print("Goal:")
            print(letters[g.cpu().numpy()])
            print("---------------")

