import gymnasium as gym
from typing import Optional, Tuple, Dict, Any, SupportsFloat
import numpy as np
import numpy.typing as npt
import deepercube.kostka.kostka_vek as kv
import deepercube.kostka.kostka as ko


class RubiksCubeEnv(gym.Env):

    def __init__(self, scramble_len: int, ep_limit: int) -> None:
        super().__init__()
        self._scramble_len = scramble_len
        self._ep_limit = ep_limit
        self._made_steps = 0
        self._cube = ko.nova_kostka()
        self._goal = ko.nova_kostka()

        self._cube_features = 6 * 3 * 3
        self._colors = 6
        self._actions = 12

        self.action_space = gym.spaces.Discrete(self._actions)
        self.observation_space = gym.spaces.MultiDiscrete(
            np.full(self._cube_features, self._colors), dtype=np.uint8
        )

    def _get_observation(self) -> npt.NDArray:
        """
        Merges `state` with the `goal`.
        Returns:
            merged stategoal - array (108,) of ints
        """
        CUBE_LEN = 6 * 3 * 3

        state_goal = np.empty([2 * CUBE_LEN], dtype=self._cube.dtype)
        state_goal[:CUBE_LEN] = self._cube.reshape(CUBE_LEN)
        state_goal[CUBE_LEN:] = self._goal.reshape(CUBE_LEN)
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
        res = ko.je_stejna(self._cube, self._goal)
        return res

    def _move_cube(self, action: int) -> None:
        """
        Perform action on the cube.
        """
        move = self._action_to_move(action)
        ko.tahni_tah(self._cube, move)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[npt.NDArray, Dict]:
        super().reset(seed=seed)

        self._cube = ko.nova_kostka()
        self._goal = ko.nova_kostka()
        ko.zamichej(self._goal, self._scramble_len)  # TODO: vyuzit self.random...
        self._made_steps = 0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[npt.NDArray, SupportsFloat, bool, bool, Dict[str, Any]]:
        self._move_cube(action)
        self._made_steps += 1

        reward = -1
        terminated = self._is_solved()
        truncated = self._made_steps >= self._ep_limit

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info


class RubiksCubeEnvVec(gym.vector.VectorEnv):

    def __init__(self, num_envs: int, scramble_len: int, ep_limit: int) -> None:
        super().__init__()
        self.num_envs = num_envs
        self._cube_features = 6 * 3 * 3
        self._colors = 6
        self._actions = 12
        self._scramble_len = scramble_len
        self._ep_limit = ep_limit
        self._made_steps = np.zeros(num_envs, dtype=np.uint64)
        self._cubes = kv.nova_kostka_vek(num_envs)
        self._goals = kv.nova_kostka_vek(num_envs)
        self._auto_reseting = np.full(num_envs, False)

        self.single_action_space = gym.spaces.Discrete(self._actions)
        self.single_observation_space = gym.spaces.MultiDiscrete(
            np.full(2 * self._cube_features, self._colors)
        )

        self.action_space = gym.spaces.MultiDiscrete(np.full(num_envs, self._actions))
        self.observation_space = gym.spaces.MultiDiscrete(
            np.full([num_envs, 2 * self._cube_features], self._colors)
        )

    def _get_observation(self) -> npt.NDArray:
        """
        Merges `states` with their corresponding `goals`.
        Returns:
            merged stategoals - array (batch, 108) of ints
        """
        batch_size = self._cubes.shape[0]
        CUBE_LEN = 6 * 3 * 3

        state_goal = np.empty([batch_size, 2 * CUBE_LEN], dtype=self._cubes.dtype)
        state_goal[:, :CUBE_LEN] = self._cubes.reshape(batch_size, CUBE_LEN)
        state_goal[:, CUBE_LEN:] = self._goals.reshape(-1, CUBE_LEN)
        return state_goal

    def _actions_to_moves(self, actions: npt.NDArray) -> npt.NDArray:
        """
        Converts `actions` (0..11) to moves (-6..1,1..6).
        """
        minus_moves = (actions > 5).astype(np.int64)
        moves = actions + 1
        moves -= 2 * actions * minus_moves
        moves += 4 * minus_moves
        return moves

    def _get_info(self) -> Dict:
        return {}

    def _is_solved(self) -> npt.NDArray:
        """
        Checks each cube to be equal to it's corresponding goal.

        Returns:
            - ndarray (num_envs,) of type bool
        """
        res = kv.je_stejna(self._cubes, self._goals)
        return res

    def _move_cubes(self, actions: npt.NDArray) -> None:
        """
        Perform actions on the cubes.
        """
        moves = self._actions_to_moves(actions)
        kv.tahni_tah_vek(self._cubes, moves)

    def _autoreset(self) -> None:
        """
        Resets envs acording to `self._auto_reseting`.
        """
        num_of_reset = np.count_nonzero(self._auto_reseting)
        if num_of_reset == 0:  # Cube operations would break
            return

        # set cubes to start
        self._cubes[self._auto_reseting] = kv.nova_kostka_vek(num_of_reset)

        # sample new goals
        new_goals = kv.nova_kostka_vek(num_of_reset)
        kv.zamichej_nahodnymi_tahy_vek(
            new_goals, self._scramble_len
        )  # TODO: vyuzit self.random...
        self._goals[self._auto_reseting] = new_goals

        # made steps to zeros
        self._made_steps *= np.logical_not(self._auto_reseting)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[npt.NDArray, Dict]:
        super().reset(seed=seed)

        self._cubes = kv.nova_kostka_vek(self.num_envs)
        self._goals = kv.nova_kostka_vek(self.num_envs)
        kv.zamichej_nahodnymi_tahy_vek(
            self._goals, self._scramble_len
        )  # TODO: vyuzit self.random...
        self._made_steps *= 0
        self._auto_reseting &= False

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, actions: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, Dict[str, Any]]:
        assert len(actions.shape) == 1, "Actions should be a 1D vector of actions."
        assert actions.shape[0] == self.num_envs, "Invalid number of actions provided."

        self._move_cubes(actions)
        self._made_steps += 1
        self._autoreset()

        rewards = -np.ones(self.num_envs)
        terminated = self._is_solved()
        truncated = self._made_steps >= self._ep_limit# - 1

        self._auto_reseting = np.logical_or(terminated, truncated)

        obs = self._get_observation()
        info = self._get_info()

        return obs, rewards, terminated, truncated, info
