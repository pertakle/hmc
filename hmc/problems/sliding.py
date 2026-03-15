from .baseproblem import Problem
import torch


def swap_fn(hole: int, action: int, size: int) -> int:
    """Actions:
        0: right
        1: down
        2: left
        3: up
    """
    column = hole % size
    row = hole // size
    #   try right and space on the right     try left and NO space on the left
    if (action == 0 and column < size - 1) or (action == 2 and column == 0):
        return hole + 1  # move right

    #   try left and space on the left    try right and NO space on the right
    elif (action == 2 and column > 0) or (action == 0 and column == size - 1):
        return hole - 1  # move left

    #     try down and space below           try up and NO space above
    elif (action == 1 and row < size - 1) or (action == 3 and row == 0):
        return hole + size  # move down

    else:
        return hole - size  # move up


class Sliding(Problem):

    """ Lloyd's sliding n-puzzle

    Actions:
        0: right
        1: down
        2: left
        3: up
    """

    def __init__(
        self, size: int, device: torch.device | None = None, seed: int | None = None
    ) -> None:
        super().__init__(size, 4, size**2, size**2, size**2, device, seed)
        self._swaps = torch.tensor(
            [
                [swap_fn(hole, action, self._size) for action in range(self._num_actions)]
                for hole in range(self._size**2)
            ],
            dtype=torch.long,
            device=self._device,
        )
        self._inversions = torch.tensor([2, 3, 0, 1], dtype=torch.long, device=self._device)

    def new_state(self) -> torch.Tensor:
        state = torch.arange(self._size**2 + 1, dtype=torch.long, device=self._device)
        state[-1] = 0
        return state

    def new_state_vec(self, num_states: int) -> torch.Tensor:
        states = torch.arange(
            0, self._size**2 + 1, dtype=torch.long, device=self._device
        )
        states = states.repeat(num_states, 1)
        states[:, -1] = 0
        return states

    def perform_action(self, state: torch.Tensor, action: int) -> None:
        hole = state[-1]
        swap = self._swaps[hole, action]
        state[hole] = state[swap]
        state[swap] = 0  # hole
        state[-1] = swap

    def perform_action_vec(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        holes = states[:, -1].clone()  # torch complains without clone
        swaps = self._swaps[holes, actions]
        _brang = torch.arange(len(states), device=self._device)
        states[_brang, holes] = states[_brang, swaps]
        states[_brang, swaps] = 0  # hole
        states[:, -1] = swaps

    def invert_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self._inversions[actions]

    def make_features(self, state: torch.Tensor) -> torch.Tensor:
        res = state[:-1]
        return res

    def make_features_vec(self, states: torch.Tensor) -> torch.Tensor:
        res = states[:, :-1]
        return res

    # def make_stategoal(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # return torch.hstack((state[:-1], goal[:-1]))

    # def make_stategoal_vec(
        # self, states: torch.Tensor, goals: torch.Tensor
    # ) -> torch.Tensor:
        # states_b, goals_b = torch.broadcast_tensors(states, goals)
        # return torch.hstack((states_b[:, :-1], goals_b[:, :-1]))
