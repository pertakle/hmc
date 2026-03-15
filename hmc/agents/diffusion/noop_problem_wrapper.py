from hmc.problems.baseproblem import Problem
import torch

class NoOpProblemWrapper:
    """Wrapper that adds a no-op action.
    The no-op action has the index 0, the old actions are increased by 1.
    """

    def __init__(self, problem: Problem) -> None:
        self._problem = problem
        self._all_actions = torch.arange(self.num_actions, dtype=torch.long, device=self.device)

    @property
    def device(self) -> torch.device:
        return self._problem._device

    @property
    def num_actions(self) -> int:
        return self._problem._num_actions + 1

    @property
    def state_size(self) -> int:
        return self._problem._state_size

    @property
    def state_bound(self) -> int:
        return self._problem._state_bound

    @property
    def num_features(self) -> int:
        return self._problem._num_features

    def seed(self, seed: int) -> None:
        self._problem._rng.manual_seed(seed)

    def new_state(self) -> torch.Tensor:
        """Returns new state."""
        return self._problem.new_state()

    def new_state_vec(self, num_states: int) -> torch.Tensor:
        """Returns `num_states` new states."""
        return self._problem.new_state_vec(num_states)

    def sample_actions(self, shape: tuple[int, ...] | list[int]) -> torch.Tensor:
        return self._problem.sample_actions(shape) + 1

    def sample_actions_noop(self, shape: tuple[int, ...] | list[int]) -> torch.Tensor:
        """Samples uniformly random actions."""
        original_actions = self._problem.sample_actions(shape)
        noise = torch.rand(original_actions.shape, generator=self._problem._rng, device=self.device)
        noop = noise < 1 / self.num_actions
        return torch.where(noop, 0, original_actions + 1)

    def scramble(self, state: torch.Tensor, scramble_len: int) -> torch.Tensor:
        """Scramble `state` in-place with `scramble_len` random moves excluding no-op.
        Returns the vector of applied actions.
        """
        return self._problem.scramble(state, scramble_len)

    def scramble_vec(self, states: torch.Tensor, scramble_len: int) -> torch.Tensor:
        """Scramble `states` in-place with `scramble_len` random moves excluding no-op.
        Returns batched vectors of applied actions (scramble_len, num_states).
        """
        return self._problem.scramble_vec(states, scramble_len)

    def perform_action(self, state: torch.Tensor, action: int) -> None:
        """Perform `action` on the `state` in-place."""
        if action > 0:
            self._problem.perform_action(state, action - 1)

    def perform_action_vec(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        """Perform `actions` on the corresponding `states` in-place."""
        # TODO: check if perform_action_vec(states[mask], actions) is enough
        mask = actions > 0
        masked_states = states[mask]
        self._problem.perform_action_vec(masked_states, actions[mask] - 1)
        states[mask] = masked_states

    def invert_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Returns a tensor of the same shape with inverse actions."""
        mask = actions > 0
        inverted = torch.zeros_like(actions)
        inverted[mask] = self._problem.invert_actions(actions[mask] - 1) + 1
        return inverted

    def perform_all_actions_vec(self, states: torch.Tensor) -> torch.Tensor:
        """Performs all possible action on every state.
        Params: `states`: (B, *S)
        Returns: (B, A, *S)
        """
        # this is too dirty
        # type(self._problem).perform_all_actions_vec(self, states)

        B = states.shape[0]
        S = states.shape[1:]
        all_children = torch.repeat_interleave(states, self.num_actions, dim=0)
        all_actions = self._all_actions.repeat(len(states))
        self.perform_action_vec(all_children, all_actions)
        return all_children.reshape(B, self.num_actions, *S)

    def is_solved(self, state: torch.Tensor, goal: torch.Tensor) -> bool:
        """Returns `True` if `state` reached `goal`."""
        return self._problem.is_solved(state, goal)

    def is_solved_vec(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """Returns a vector of bools whether each state reached its goal."""
        return self._problem.is_solved_vec(states, goals)

    def make_features(self, state: torch.Tensor) -> torch.Tensor:
        """Returns read-only vector of features of the `state`."""
        return self._problem.make_features(state)

    def make_features_vec(self, states: torch.Tensor) -> torch.Tensor:
        """Returns read-only vectors of features of the `states`."""
        return self._problem.make_features_vec(states)

    def make_stategoal(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Returns a vector representing the complete state-goal."""
        return self._problem.make_stategoal(state, goal)

    def make_stategoal_vec(
        self, states: torch.Tensor, goals: torch.Tensor
    ) -> torch.Tensor:
        """Returns a vector representing the complete state-goal.

        Params:
            - `states`: vector of states
            - `goals`: vector of goals or a single goal
        """
        return self._problem.make_stategoal_vec(states, goals)
