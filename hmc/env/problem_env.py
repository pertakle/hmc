from typing import Dict, Tuple, Optional, Any, SupportsFloat
import gymnasium as gym
import torch
import numpy as np
from hmc.problems.baseproblem import Problem



class ProblemEnv(gym.Env):
    def __init__(self, problem: Problem, scramble_len: int, ep_limit: int) -> None:
        super().__init__()

        self._problem = problem
        self._scramble_len = scramble_len
        self._ep_limit = ep_limit
        self._made_steps = 0

        self._state = self._problem.new_state()
        self._goal = self._problem.new_state()

        self.observation_space = gym.spaces.MultiDiscrete(
            np.full(2 * problem.state_size, problem.state_bound)
        )
        self.action_space = gym.spaces.Discrete(self._problem.num_actions)

    def _get_observation(self) -> torch.Tensor:
        return self._problem.make_stategoal(self._state, self._goal)

    def _get_info(self) -> dict:
        return {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        super().reset(seed=seed, options=options)

        if seed is not None:
            self._problem.seed(seed)

        self._state = self._problem.new_state()
        self._goal = self._problem.new_state()
        self._problem.scramble(self._goal, self._scramble_len)
        self._made_steps = 0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[torch.Tensor, SupportsFloat, bool, bool, Dict[str, Any]]:
        self._problem.perform_action(self._state, action)
        self._made_steps += 1

        reward = -1
        terminated = self._problem.is_solved(self._state, self._goal)
        truncated = self._made_steps >= self._ep_limit

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info


class ProblemEnvVec(gym.vector.VectorEnv):

    metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP}

    def __init__(
        self, num_envs: int, problem: Problem, scramble_len: int, ep_limit: int
    ) -> None:
        super().__init__()

        self._problem = problem
        self.num_envs = num_envs
        self._scramble_len = scramble_len
        self._ep_limit = ep_limit

        self._made_steps = torch.zeros(num_envs, dtype=torch.long, device=self._problem.device)
        self._autoreseting = torch.zeros(num_envs, dtype=torch.bool, device=self._problem.device)
        self._states = self._problem.new_state_vec(num_envs)
        self._goals = self._problem.new_state_vec(num_envs)

        self.single_observation_space = gym.spaces.MultiDiscrete(
            np.full([2 * self._problem.state_size], self._problem.state_bound)
        )
        self.single_action_space = gym.spaces.Discrete(self._problem.num_actions)
        self.observation_space = gym.spaces.MultiDiscrete(
            np.full([num_envs, 2 * self._problem.state_size], self._problem.state_bound)
        )
        self.action_space = gym.spaces.MultiDiscrete(np.full([num_envs], self._problem.state_bound))

    def _get_observations(self) -> torch.Tensor:
        return self._problem.make_stategoal_vec(self._states, self._goals)

    def _get_info(self) -> dict:
        return {}

    def _reset(self, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self._problem.device)

        num_resets = mask.count_nonzero()
        new_states = self._problem.new_state_vec(num_resets)

        self._states[mask] = new_states

        self._problem.scramble_vec(new_states, self._scramble_len)
        self._goals[mask] = new_states
        self._made_steps[mask] = 0

    def reset(
        self,
        *,
        mask: torch.Tensor | None = None,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        super().reset(seed=seed, options=options)

        self._reset(mask)
        obs = self._get_observations()
        info = self._get_info()
        return obs, info

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        assert len(actions.shape) == 1, "Actions should be a 1D vector of actions."
        assert (
            actions.shape[0] == self.num_envs
        ), f"Invalid number of actions provided ({actions.shape[0]} != {self.num_envs})."

        self._problem.perform_action_vec(self._states, actions)
        self._made_steps += 1
        self._reset(mask=self._autoreseting)

        rewards = -torch.ones(self.num_envs, device=self._problem.device)
        terminated = self._problem.is_solved_vec(self._states, self._goals)
        truncated = self._made_steps >= self._ep_limit
        self._autoreseting = (terminated | truncated) & ~self._autoreseting

        obs = self._get_observations()
        info = self._get_info()

        return obs, rewards, terminated, truncated, info


