import gymnasium as gym
import torch
import numpy as np
import hmc.kostka.torch_cube as tcu
import hmc.kostka.torch_cube_vec as tcv
from hmc.env import base_env

class RubiksCubeEnv(base_env.BaseTorchEnv):
    def __init__(self, size: int, scramble_len: int, ep_limit: int, device=None) -> None:
        super().__init__(size, scramble_len, ep_limit, device)

        self._cube_features = 6 * 3 * 3
        self._colors = 6
        self._actions = 12

        self.action_space = gym.spaces.Discrete(self._actions)
        self.observation_space = gym.spaces.MultiDiscrete(
            np.full(self._cube_features * 2, self._colors)
        )

    def _new_state(self) -> torch.Tensor:
        return tcu.new_cube()

    def _scramble(self, state: torch.Tensor) -> None:
        tcu.scramble(state, self._scramble_len)

    def _action_to_move(self, action: int) -> int:
        # action - 5 - (action < 6)
        return action - 6 if action < 6 else action - 5

    def _perform_action(self, action: int, state: torch.Tensor) -> None:
        move = self._action_to_move(action)
        tcu.make_move(state, move)

    def _is_solved(self) -> bool:
        return tcu.is_same(self._state, self._goal)

    def print(self) -> None:
        print("State:")
        tcu.print_cube(self._state)
        print("Goal:")
        tcu.print_cube(self._goal)

class RubiksCubeEnvVec(base_env.BaseTorchEnvVec):
    
    metadata = base_env.BaseTorchEnvVec.metadata
    
    def __init__(self, num_envs: int, size: int, scramble_len: int, ep_limit: int, device=None) -> None:
        super().__init__(num_envs, size, scramble_len, ep_limit, device)

        self._cube_features = 6 * 3 * 3
        self._colors = 6
        self._actions = 12

        self.single_action_space = gym.spaces.Discrete(self._actions)
        self.single_observation_space = gym.spaces.MultiDiscrete(
            np.full(self._cube_features * 2, self._colors)
        )
        self.action_space = gym.spaces.MultiDiscrete(
            np.full(self.num_envs, self._colors)
        )
        self.observation_space = gym.spaces.MultiDiscrete(
            np.full([self.num_envs, self._cube_features * 2], self._colors)
        )

    def _new_states(self, num_states: int) -> torch.Tensor:
        return tcv.new_cube_vec(num_states)

    def _scramble(self, states: torch.Tensor) -> None:
        tcv.scramble_vec(states, self._scramble_len)

    def _actions_to_moves(self, actions: torch.Tensor) -> torch.Tensor:
        return actions - 5 - (actions < 6).type(torch.long)

    def _perform_actions(self, actions: torch.Tensor, states: torch.Tensor) -> None:
        moves = self._actions_to_moves(actions)
        tcv.make_move_vec(states, moves)


