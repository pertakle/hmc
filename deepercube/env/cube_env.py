import gymnasium as gym
from typing import Optional, SupportsFloat, Tuple, Dict, Any
import numpy as np
import numpy.typing as npt
import deepercube.kostka.kostka_vek as kv
import deepercube.utils.data_gen as dg

class RubiksCubeEnv(gym.Env):
    
    def __init__(self, num_envs: int, scramble_len: int, ep_limit: int) -> None:
        super().__init__()
        self._num_envs = num_envs
        self._scramble_len = scramble_len
        self._ep_limit = ep_limit
        self._made_steps = 0
        self._cubes = kv.nova_kostka_vek(num_envs)
        self._goals = kv.nova_kostka_vek(num_envs)

        self.action_space = gym.spaces.Discrete(12)
        self.observation_space = gym.spaces.Box(low=0, high=5, shape=(num_envs, 108), dtype=np.uint8)

    def _get_observation(self) -> npt.NDArray:
        """
        Merges `states` with their corresponding `goals`.
        Returns:
            merged stategoals - array (batch, 108) of ints
        """
        batch_size = self._cubes.shape[0]
        CUBE_LEN = 6*3*3

        state_goal = np.empty([batch_size, 2*CUBE_LEN], dtype=self._cubes.dtype)
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
        # TODO: check shape - 1d or (-1, 1) 2d
        res = kv.je_stejna(self._cubes, self._goals)
        assert len(res.shape) == 1
        return res

    def _move_cubes(self, actions: npt.NDArray) -> None:
        moves = self._actions_to_moves(actions)
        print("taken moves:", moves)
        kv.tahni_tah_vek(self._cubes, moves)


    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[npt.NDArray, Dict]:
        super().reset(seed=seed)

        self._cubes = kv.nova_kostka_vek(self._num_envs)
        self._goals = kv.nova_kostka_vek(self._num_envs)
        kv.zamichej_nahodnymi_tahy_vek(self._goals, self._scramble_len) # TODO: vyuzit self.random...
        self._made_steps = 0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    # gym.Env thinks it is a single env and does not like array of rewards etc.
    def step(self, actions: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, Dict]: # type: ignore
        print("taken actions:", actions)
        assert len(actions.shape) == 1
        assert actions.shape[0] == self._num_envs

        self._move_cubes(actions)
        self._made_steps += 1

        rewards = -np.ones(self._num_envs)
        terminated = self._is_solved()
        truncated = np.full(terminated.shape, self._made_steps >= self._ep_limit)

        obs = self._get_observation()
        info = self._get_info()

        return obs, rewards, terminated, truncated, info
