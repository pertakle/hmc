from . import base_env
import gymnasium as gym
import torch
import numpy as np

def swap_fn(hole: int, action: int, size: int) -> int:
    column = hole % size
    row = hole // size
    if (action == 0 and column < size - 1) or (action == 2 and column == 0):
        return hole + 1
    elif (action == 2 and column > 0) or (action == 0 and column == size - 1):
        return hole - 1
    elif (action == 1 and row < size - 1) or (action == 3 and row == 0):
        return hole + size
    else:
        return hole - size


class Sliding(base_env.BaseTorchEnv):

    def __init__(self, size: int, scramble_len: int, ep_limit: int, device=None) -> None:
        super().__init__(size, scramble_len, ep_limit, device)

        self.observation_space = gym.spaces.MultiDiscrete(np.full([2 * self._size**2], self._size**2))
        self.action_space = gym.spaces.Discrete(4)

        self._swaps = torch.tensor(
            [[swap_fn(hole, action, self._size) for action in range(4)] for hole in range(self._size**2)],
            dtype=torch.long, device=self._device
        )

    def _new_state(self) -> torch.Tensor:
        # flattened(size x size matrix) + hole index
        # 0 represents the hole
        new_state = torch.arange(self._size**2 + 1, dtype=torch.long, device=self._device)
        new_state[-1] = 0
        return new_state

    def _scramble(self, state: torch.Tensor) -> None:
        for action in torch.randint(0, 4, [self._scramble_len], device=self._device):
            self._perform_action(action, state)

    def _perform_action(self, action: int, state: torch.Tensor) -> None:
        hole = state[-1]
        swap = self._swaps[hole, action]
        state[hole] = state[swap]
        state[swap] = 0  # hole
        state[-1] = swap

    def print(self) -> None:
        print("State:")
        print(self._state[:-1].cpu().numpy().reshape(self._size, self._size))
        print("Goal:")
        print(self._goal[:-1].cpu().numpy().reshape(self._size, self._size))

    # override
    def _get_observation(self) -> torch.Tensor:
        return torch.hstack((self._state[:-1], self._goal[:-1]))


class SlidingVec(base_env.BaseTorchEnvVec):
    
    metadata = base_env.BaseTorchEnvVec.metadata
    
    def __init__(self, num_envs: int, size: int, scramble_len: int, ep_limit: int, device=None) -> None:
        super().__init__(num_envs, size, scramble_len, ep_limit, device)

        self.single_observation_space = gym.spaces.MultiDiscrete(np.full([2 * self._size**2], self._size**2))
        self.single_action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.MultiDiscrete(np.full([num_envs, 2 * self._size**2], self._size**2))
        self.action_space = gym.spaces.MultiDiscrete(np.full([num_envs], 4))

        self._swaps = torch.tensor(
            [[swap_fn(hole, action, self._size) for action in range(4)] for hole in range(self._size**2)],
            dtype=torch.long, device=self._device
        )


    def _new_states(self, num_states: int) -> torch.Tensor:
        """Returns a batch of `num_states` new states."""
        new_states = torch.arange(0, self._size**2 + 1, dtype=torch.long, device=self._device)
        new_states = new_states.repeat(num_states, 1)
        new_states[:, -1] = 0
        return new_states

    def _scramble(self, states: torch.Tensor) -> None:
        """Scrambles `states` in-place with `self._scramble_len` random moves."""
        scrambles = torch.randint(0, 4, [self._scramble_len, len(states)], device=self._device)
        # print(scrambles.cpu().numpy())
        for actions in scrambles:
            self._perform_actions(actions, states)

    def _perform_actions(self, actions: torch.Tensor, states: torch.Tensor) -> None:
        """Performs corresponding actions on each state in `self._state`."""
        holes = states[:, -1].clone()  # torch complains without clone
        swaps = self._swaps[holes, actions]
        _brang = torch.arange(len(states), device=self._device)
        states[_brang, holes] = states[_brang, swaps]
        states[_brang, swaps] = 0  # hole
        states[:, -1] = swaps

    # override
    def _get_observations(self) -> torch.Tensor:
        return torch.hstack((self._states[:, :-1], self._goals[:, :-1]))

    def print(self) -> None:
        print("\n============\n")
        for state, g in zip(self._states, self._goals):
            print("State:")
            print(state[:-1].cpu().numpy().reshape(self._size, self._size))
            print("Goal:")
            print(g[:-1].cpu().numpy().reshape(self._size, self._size))
            print("---------------")

