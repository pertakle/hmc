from .baseproblem import Problem
import torch


class LightsOut(Problem):

    def __init__(
        self,
        size: int,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(size, size**2, size**2, 2, size**2, device, seed)
        self._row_shift = torch.as_tensor([-1, 0, 0, 0, 1], dtype=torch.long, device=device)
        self._col_shift = torch.as_tensor([0, -1, 0, 1, 0], dtype=torch.long, device=device)

    def new_state(self) -> torch.Tensor:
        # TODO: use -1 instead of 0 to drop unnecessary `.add_(1)`
        # TODO: use index-based flattened representation to avoid slow modulus
        return torch.zeros(
            [self._size, self._size], dtype=torch.long, device=self._device
        )

    def new_state_vec(self, num_states: int) -> torch.Tensor:
        return torch.zeros(
            [num_states, self._size, self._size], dtype=torch.long, device=self._device
        )

    # TODO: more efficeint `scramble` and `scramble_vec`

    def perform_action(self, state: torch.Tensor, action: int) -> None:
        row = action // self._size
        col = action % self._size

        # TODO: use in-place operators
        rows = torch.clip(row + self._row_shift, 0, self._size - 1)
        cols = torch.clip(col + self._col_shift, 0, self._size - 1)
        
        state[rows, cols] = 1 - state[rows, cols]
        # print(state.shape, rows, cols)
        # state[rows, cols].mul_(-1).add_(1)

    def perform_action_vec(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        # TODO: use in-place operators
        row = actions // self._size
        col = actions % self._size

        # TODO: use in-place operators
        rows = torch.clip(row[..., None] + self._row_shift[None], 0, self._size - 1)
        cols = torch.clip(col[..., None] + self._col_shift[None], 0, self._size - 1)
        b_range = torch.arange(len(actions), device=self._device)

        # states[b_range[:, None], rows, cols].mul_(-1).add_(1)
        states[b_range[:, None], rows, cols] = 1 - states[b_range[:, None], rows, cols]

    def invert_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def make_features(self, state: torch.Tensor) -> torch.Tensor:
        return state.flatten()

    def make_features_vec(self, states: torch.Tensor) -> torch.Tensor:
        return states.flatten(1)

    # def make_stategoal(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # return torch.hstack((state.flatten(), goal.flatten()))

    # def make_stategoal_vec(
        # self, states: torch.Tensor, goals: torch.Tensor
    # ) -> torch.Tensor:
        # states_b, goals_b = torch.broadcast_tensors(states, goals)
        # return torch.hstack((states_b.flatten(1), goals_b.flatten(1)))
