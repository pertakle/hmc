import gymnasium as gym
from typing import Optional, Tuple, Dict, Any, SupportsFloat
import torch
import numpy as np
import deepercube.kostka.torch_cube as tcu
import deepercube.kostka.torch_cube_vec as tcv
import deepercube.utils.torch_utils as tut


class TorchRubiksCubeEnv(gym.Env):

    def __init__(self, scramble_len: int, ep_limit: int) -> None:
        super().__init__()
        self._scramble_len = scramble_len
        self._ep_limit = ep_limit
        self._made_steps = 0
        self._cube = tcu.new_cube()
        self._goal = tcu.new_cube()
        self._device = tut.get_torch_cube_device()

        self._cube_features = 6 * 3 * 3
        self._colors = 6
        self._actions = 12

        self.action_space = gym.spaces.Discrete(self._actions)
        self.observation_space = gym.spaces.MultiDiscrete(
            np.full(self._cube_features, self._colors), dtype=np.uint8
        )

    def _get_observation(self) -> torch.Tensor:
        """
        Merges `state` with the `goal`.
        Returns:
            merged stategoal - array (108,) of ints
        """
        CUBE_LEN = 6 * 3 * 3

        state_goal = torch.empty([2 * CUBE_LEN], dtype=tcu.TColorT, device=self._device)
        state_goal[:CUBE_LEN] = self._cube.flatten()  # reshape(CUBE_LEN)
        state_goal[CUBE_LEN:] = self._goal.flatten()  # reshape(CUBE_LEN)
        return state_goal

    def _action_to_move(self, action: int) -> int:
        """
        Converts `action` (0..11) to move (-6..1,1..6).
        """
        minus_move = int(action > 5)
        move = action + 1
        move -= 2 * action * minus_move
        move += 4 * minus_move
        return move

    def _get_info(self) -> Dict:
        return {}

    def _is_solved(self) -> bool:
        """
        Checks the cube to be equal to it's goal.

        Returns:
            - bool: True if the cube is same as the goal
        """
        res = tcu.is_same(self._cube, self._goal)
        return res

    def _move_cube(self, action: int) -> None:
        """
        Perform action on the cube.
        """
        move = self._action_to_move(action)
        tcu.make_move(self._cube, move)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        super().reset(seed=seed)

        self._cube = tcu.new_cube()
        self._goal = tcu.new_cube()
        tcu.scramble(self._goal, self._scramble_len)  # TODO: vyuzit self.random...
        self._made_steps = 0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[torch.Tensor, SupportsFloat, bool, bool, Dict[str, Any]]:
        self._move_cube(action)
        self._made_steps += 1

        reward = -1
        terminated = self._is_solved()
        truncated = self._made_steps >= self._ep_limit

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

class BaseTorchEnvVec(gym.vector.VectorEnv):
    def __init__(self, num_envs: int, scramble_len: int, ep_limit: int, device=None) -> None:
        super().__init__()
        self.num_envs = num_envs
        self._scramble_len = scramble_len
        self._ep_limit = ep_limit

        self._device = device
        self._made_steps = torch.zeros([num_envs], dtype=torch.int64, device=self._device)
        self._states = self._new_states(num_envs)
        self._goals = self._new_states(num_envs)
        self._auto_reseting = torch.full([num_envs], False, dtype=torch.bool, device=self._device)

    def _new_states(self, num_states: int) -> torch.Tensor:
        """Returns a batch of `num_states` new states."""
        raise NotImplementedError

    def _is_solved(self) -> torch.Tensor:
        """
        Returns a vector of bools whether each state in `self._states` reached it's `self._goals`.
        """
        return torch.all(self._states == self._goals, dim=list(range(1, len(self._states.shape))))

    def _scramble(self, states: torch.Tensor) -> None:
        """Scrambles `states` in-place with `self._scramble_len` random moves."""
        raise NotImplementedError

    def _perform_actions(self, actions: torch.Tensor) -> None:
        """Performs corresponding actions on each state in `self._state`."""
        raise NotImplementedError

    def _get_observations(self) -> torch.Tensor:
        return torch.hstack((self._states.flatten(1), self._goals.flatten(1)))

    def _get_info(self) -> Dict:
        return {}

    def _autoreset(self) -> None:
        """
        Resets envs acording to `self._auto_reseting`.
        """
        num_of_reset = torch.count_nonzero(self._auto_reseting)
        if num_of_reset == 0:  # Cube operations would break
            return

        # set cubes to start
        self._states[self._auto_reseting] = self._new_states(num_of_reset)

        # sample new goals
        new_goals = self._new_states(num_of_reset)
        tcv.scramble_vec(new_goals, self._scramble_len)
        self._goals[self._auto_reseting] = new_goals

        # made steps to zeros
        self._made_steps *= ~self._auto_reseting

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        super().reset(seed=seed)

        self._cubes = tcv.new_cube_vec(self.num_envs)
        self._goals = tcv.new_cube_vec(self.num_envs)
        tcv.scramble_vec(self._goals, self._scramble_len)
        self._made_steps[:] = 0
        self._auto_reseting &= False

        obs = self._get_observations()
        info = self._get_info()
        return obs, info

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        assert len(actions.shape) == 1, "Actions should be a 1D vector of actions."
        assert actions.shape[0] == self.num_envs, "Invalid number of actions provided."

        self._perform_actions(actions)
        self._made_steps += 1
        self._autoreset()

        rewards = -torch.ones(self.num_envs, device=self._device)
        terminated = self._is_solved()
        truncated = self._made_steps >= self._ep_limit# - 1

        self._auto_reseting = terminated | truncated

        obs = self._get_observations()
        info = self._get_info()

        return obs, rewards, terminated, truncated, info

class TorchRubiksCubeEnvVec(gym.vector.VectorEnv):

    def __init__(self, num_envs: int, scramble_len: int, ep_limit: int) -> None:
        super().__init__()
        self.num_envs = num_envs
        self._cube_features = 6 * 3 * 3
        self._colors = 6
        self._actions = 12
        self._scramble_len = scramble_len
        self._ep_limit = ep_limit
        self._device = tut.get_torch_cube_device()
        self._made_steps = torch.zeros([num_envs], dtype=torch.int64, device=self._device)
        self._cubes = tcv.new_cube_vec(num_envs)
        self._goals = tcv.new_cube_vec(num_envs)
        self._auto_reseting = torch.full([num_envs], False, dtype=torch.bool, device=self._device)

        self.single_action_space = gym.spaces.Discrete(self._actions)
        self.single_observation_space = gym.spaces.MultiDiscrete(
            np.full(2 * self._cube_features, self._colors)
        )

        self.action_space = gym.spaces.MultiDiscrete(np.full(num_envs, self._actions))
        self.observation_space = gym.spaces.MultiDiscrete(
            np.full([num_envs, 2 * self._cube_features], self._colors)
        )

    def _get_observation(self) -> torch.Tensor:
        """
        Merges `states` with their corresponding `goals`.
        Returns:
            merged stategoals - tensor (batch, 108) of ints
        """
        batch_size = self._cubes.shape[0]
        CUBE_LEN = 6 * 3 * 3

        state_goal = torch.empty([batch_size, 2 * CUBE_LEN], dtype=tcu.TColorT, device=self._device)
        state_goal[:, :CUBE_LEN] = self._cubes.reshape(batch_size, CUBE_LEN)
        state_goal[:, CUBE_LEN:] = self._goals.reshape(batch_size, CUBE_LEN)
        return state_goal

    def _actions_to_moves(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Converts `actions` (0..11) to moves (-6..1,1..6).
        """
        # TODO: cast to int64 if it does not work
        minus_moves = (actions > 5)#.type(torch.int64)
        moves = actions + 1
        moves -= 2 * actions * minus_moves
        moves += 4 * minus_moves
        return moves

    def _get_info(self) -> Dict:
        return {}

    def _is_solved(self) -> torch.Tensor:
        """
        Checks each cube to be equal to it's corresponding goal.

        Returns:
            - Tensor (num_envs,) of type bool
        """
        return tcv.is_same_vec(self._cubes, self._goals)

    def _move_cubes(self, actions: torch.Tensor) -> None:
        """
        Perform actions on the cubes.
        """
        moves = self._actions_to_moves(actions)
        tcv.make_move_vec(self._cubes, moves)

    def _autoreset(self) -> None:
        """
        Resets envs acording to `self._auto_reseting`.
        """
        num_of_reset = torch.count_nonzero(self._auto_reseting)
        if num_of_reset == 0:  # Cube operations would break
            return

        # set cubes to start
        self._cubes[self._auto_reseting] = tcv.new_cube_vec(num_of_reset)

        # sample new goals
        new_goals = tcv.new_cube_vec(num_of_reset)
        tcv.scramble_vec(new_goals, self._scramble_len)
        self._goals[self._auto_reseting] = new_goals

        # made steps to zeros
        self._made_steps *= ~self._auto_reseting

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        super().reset(seed=seed)

        self._cubes = tcv.new_cube_vec(self.num_envs)
        self._goals = tcv.new_cube_vec(self.num_envs)
        tcv.scramble_vec(self._goals, self._scramble_len)
        self._made_steps[:] = 0
        self._auto_reseting &= False

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        assert len(actions.shape) == 1, "Actions should be a 1D vector of actions."
        assert actions.shape[0] == self.num_envs, "Invalid number of actions provided."

        self._move_cubes(actions)
        self._made_steps += 1
        self._autoreset()

        rewards = -torch.ones(self.num_envs, device=self._device)
        terminated = self._is_solved()
        truncated = self._made_steps >= self._ep_limit# - 1

        self._auto_reseting = terminated | truncated

        obs = self._get_observation()
        info = self._get_info()

        return obs, rewards, terminated, truncated, info
