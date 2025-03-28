from typing import Dict, Tuple, Optional, Any, SupportsFloat
import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch

class BaseTorchEnv(gym.Env):
    def __init__(self, scramble_len: int, ep_limit: int, device=None) -> None:
        self._scramble_len = scramble_len
        self._ep_limit = ep_limit
        self._made_steps = 0
        self._device = device

        self._state = self._new_state()
        self._goal = self._new_state()

    def _new_state(self) -> torch.Tensor:
        """Returns new state."""
        raise NotImplementedError

    def _scramble(self, state: torch.Tensor) -> None:
        """Scramble `state` with `self._scramble_len` random moves."""
        raise NotImplementedError

    def _perform_action(self, action: int, state: torch.Tensor) -> None:
        """Perform `action` on the `state` in-place."""
        raise NotImplementedError

    def _is_solved(self) -> bool:
        return bool(torch.all(self._state == self._goal).item())

    def _get_observation(self) -> torch.Tensor:
        return torch.hstack((self._state.flatten(), self._goal.flatten()))

    def _get_info(self) -> dict:
        return {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        super().reset(seed=seed)

        self._state = self._new_state()
        self._goal = self._new_state()
        self._scramble(self._goal)
        self._made_steps = 0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[torch.Tensor, SupportsFloat, bool, bool, Dict[str, Any]]:
        self._perform_action(action, self._state)
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

    def _perform_actions(self, actions: torch.Tensor, states: torch.Tensor) -> None:
        """Performs corresponding actions on each state in `state` in-place."""
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
        self._scramble(new_goals)
        self._goals[self._auto_reseting] = new_goals

        # made steps to zeros
        self._made_steps *= ~self._auto_reseting

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        super().reset(seed=seed)

        self._cubes = self._new_states(self.num_envs)
        self._goals = self._new_states(self.num_envs)
        self._scramble(self._goals)
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

        self._perform_actions(actions, self._states)
        self._made_steps += 1
        self._autoreset()

        rewards = -torch.ones(self.num_envs, device=self._device)
        terminated = self._is_solved()
        truncated = self._made_steps >= self._ep_limit# - 1

        self._auto_reseting = terminated | truncated

        obs = self._get_observations()
        info = self._get_info()

        return obs, rewards, terminated, truncated, info


class BaseNumpyEnv(gym.Env):
    def __init__(self, scramble_len: int, ep_limit: int) -> None:
        self._scramble_len = scramble_len
        self._ep_limit = ep_limit
        self._made_steps = 0

        self._state = self._new_state()
        self._goal = self._new_state()

    def _new_state(self) -> npt.NDArray:
        """Returns new state."""
        raise NotImplementedError

    def _scramble(self, state: npt.NDArray) -> None:
        """Scramble `state` with `self._scramble_len` random moves."""
        raise NotImplementedError

    def _perform_action(self, action: int, state: npt.NDArray) -> None:
        """Perform `action` on the `state` in-place."""
        raise NotImplementedError

    def _is_solved(self) -> bool:
        return np.all(self._state == self._goal).item()

    def _get_observation(self) -> npt.NDArray:
        return np.hstack((self._state.flatten(), self._goal.flatten()))

    def _get_info(self) -> dict:
        return {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[npt.NDArray, Dict]:
        super().reset(seed=seed)

        self._state = self._new_state()
        self._goal = self._new_state()
        self._scramble(self._goal)
        self._made_steps = 0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[npt.NDArray, SupportsFloat, bool, bool, Dict[str, Any]]:
        self._perform_action(action, self._state)
        self._made_steps += 1

        reward = -1
        terminated = self._is_solved()
        truncated = self._made_steps >= self._ep_limit

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

class BaseNumpyEnvVec(gym.vector.VectorEnv):
    def __init__(self, num_envs: int, scramble_len: int, ep_limit: int) -> None:
        super().__init__()
        self.num_envs = num_envs
        self._scramble_len = scramble_len
        self._ep_limit = ep_limit

        self._made_steps = np.zeros([num_envs], dtype=np.int64)
        self._states = self._new_states(num_envs)
        self._goals = self._new_states(num_envs)
        self._auto_reseting = np.full([num_envs], False, dtype=np.bool)

    def _new_states(self, num_states: int) -> npt.NDArray:
        """Returns a batch of `num_states` new states."""
        raise NotImplementedError

    def _is_solved(self) -> npt.NDArray:
        """
        Returns a vector of bools whether each state in `self._states` reached it's `self._goals`.
        """
        return np.all(self._states == self._goals, axis=tuple(range(1, len(self._states.shape))))

    def _scramble(self, states: npt.NDArray) -> None:
        """Scrambles `states` in-place with `self._scramble_len` random moves."""
        raise NotImplementedError

    def _perform_actions(self, actions: npt.NDArray, states: npt.NDArray) -> None:
        """Performs corresponding actions on each state in `self._state`."""
        raise NotImplementedError

    def _get_observations(self) -> npt.NDArray:
        return np.hstack((self._states.reshape(self.num_envs, -1), self._goals.reshape(self.num_envs, -1)))

    def _get_info(self) -> Dict:
        return {}

    def _autoreset(self) -> None:
        """
        Resets envs acording to `self._auto_reseting`.
        """
        num_of_reset = np.count_nonzero(self._auto_reseting)
        if num_of_reset == 0:  # Cube operations would break
            return

        # set cubes to start
        self._states[self._auto_reseting] = self._new_states(num_of_reset)

        # sample new goals
        new_goals = self._new_states(num_of_reset)
        self._scramble(new_goals)
        self._goals[self._auto_reseting] = new_goals

        # made steps to zeros
        self._made_steps *= ~self._auto_reseting

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[npt.NDArray, Dict]:
        super().reset(seed=seed)

        self._cubes = self._new_states(self.num_envs)
        self._goals = self._new_states(self.num_envs)
        self._scramble(self._goals)
        self._made_steps[:] = 0
        self._auto_reseting &= False

        obs = self._get_observations()
        info = self._get_info()
        return obs, info

    def step(
        self, actions: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, Dict[str, Any]]:
        assert len(actions.shape) == 1, "Actions should be a 1D vector of actions."
        assert actions.shape[0] == self.num_envs, "Invalid number of actions provided."

        self._perform_actions(actions, self._states)
        self._made_steps += 1
        self._autoreset()

        rewards = -np.ones(self.num_envs)
        terminated = self._is_solved()
        truncated = self._made_steps >= self._ep_limit# - 1

        self._auto_reseting = terminated | truncated

        obs = self._get_observations()
        info = self._get_info()

        return obs, rewards, terminated, truncated, info

