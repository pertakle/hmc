import torch


class Problem:
    """General abstract base class for problems.

    States are represented as a tensor of arbitrary dimensionality.
    Once a state is merged with `make_stategoal` the representation is
    flattened into a single vector to maintain the `ProblemEnv` convention.
    """

    def __init__(
        self,
        size: int,
        num_actions: int,
        state_size: int,
        state_bound: int,
        num_features: int,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        self._size = size
        self._num_actions = num_actions
        self._state_size = state_size
        self._state_bound = state_bound
        self._num_features = num_features
        self._device = torch.device("cpu") if device is None else device
        self._rng = torch.Generator(device)
        if seed is not None:
            self.seed(seed)
        self._all_actions = torch.arange(num_actions, dtype=torch.long, device=device)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def state_size(self) -> int:
        return self._state_size

    @property
    def state_bound(self) -> int:
        return self._state_bound

    @property
    def num_features(self) -> int:
        return self._num_features

    def seed(self, seed: int) -> None:
        self._rng.manual_seed(seed)

    def new_state(self) -> torch.Tensor:
        """Returns new state."""
        raise NotImplementedError

    def new_state_vec(self, num_states: int) -> torch.Tensor:
        """Returns `num_states` new states."""
        raise NotImplementedError

    def sample_actions(self, shape: tuple[int, ...] | list[int]) -> torch.Tensor:
        """Samples uniformly random actions."""
        return torch.randint(
            0, self.num_actions, shape, generator=self._rng, device=self._device
        )

    def scramble(self, state: torch.Tensor, scramble_len: int) -> torch.Tensor:
        """Scramble `state` in-place with `scramble_len` random moves.
        Returns the vector of applied actions.
        """
        scramble = self.sample_actions([scramble_len])
        for action in scramble:
            self.perform_action(state, action.item())
        return scramble

    def scramble_vec(self, states: torch.Tensor, scramble_len: int) -> torch.Tensor:
        """Scramble `states` in-place with `scramble_len` random moves.
        Returns batched vectors of applied actions (scramble_len, num_states).
        """
        scrambles = self.sample_actions([scramble_len, len(states)])
        for actions in scrambles:
            self.perform_action_vec(states, actions)
        return scrambles

    def perform_action(self, state: torch.Tensor, action: int) -> None:
        """Perform `action` on the `state` in-place."""
        raise NotImplementedError

    def perform_action_vec(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        """Perform `actions` on the corresponding `states` in-place."""
        raise NotImplementedError

    def invert_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Returns a tensor of the same shape with inverse actions."""
        raise NotImplementedError

    def perform_all_actions_vec(self, states: torch.Tensor) -> torch.Tensor:
        """Performs all possible action on every state.
        Params: `states`: (B, *S)
        Returns: (B, A, *S)
        """
        B = states.shape[0]
        S = states.shape[1:]
        all_children = torch.repeat_interleave(states, self._num_actions, dim=0)
        all_actions = self._all_actions.repeat(len(states))
        self.perform_action_vec(all_children, all_actions)
        return all_children.reshape(B, self._num_actions, *S)

    def is_solved(self, state: torch.Tensor, goal: torch.Tensor) -> bool:
        """Returns `True` if `state` reached `goal`."""
        return bool(torch.all(state == goal).item())

    def is_solved_vec(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """Returns a vector of bools whether each state reached its goal."""
        return torch.all(states == goals, dim=tuple(range(1, len(states.shape))))

    def make_features(self, state: torch.Tensor) -> torch.Tensor:
        """Returns read-only vector of features of the `state`."""
        raise NotImplementedError

    def make_features_vec(self, states: torch.Tensor) -> torch.Tensor:
        """Returns read-only vectors of features of the `states`."""
        raise NotImplementedError

    def make_stategoal(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Returns a vector representing the complete state-goal."""
        state_features = self.make_features(state)
        goal_features = self.make_features(goal)
        return torch.hstack((state_features, goal_features))

    def make_stategoal_vec(
        self, states: torch.Tensor, goals: torch.Tensor
    ) -> torch.Tensor:
        """Returns a vector representing the complete state-goal.

        Params:
            - `states`: vector of states
            - `goals`: vector of goals or a single goal
        """
        states_f = self.make_features_vec(states)
        goals_f = self.make_features_vec(goals)
        states_b, goals_b = torch.broadcast_tensors(states_f, goals_f)
        stategoals = torch.hstack((states_b, goals_b))
        return stategoals

