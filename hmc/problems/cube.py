from .baseproblem import Problem
import hmc.kostka.torch_cube as tcu
import hmc.kostka.torch_cube_vec as tcv
import hmc.utils.torch_utils as tut
import torch

class RubiksCube(Problem):

    def __init__(
        self,
        size: int,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(size, 12, 54, 6, 6*3*3, device, seed)
        if device is not None:
            tut.set_torch_cube_device(device)
        # tut.set_torch_cube_device(torch.device("cpu") if device is None else device)

    def new_state(self) -> torch.Tensor:
        """Returns new state."""
        return tcu.new_cube()

    def new_state_vec(self, num_states: int) -> torch.Tensor:
        """Returns `num_states` new states."""
        # assert num_states > 0
        return tcv.new_cube_vec(num_states)

    # def scramble(self, state: torch.Tensor, scramble_len: int) -> None:
        # """Scramble `state` in-place with `scramble_len` random moves."""
        # tcu.scramble(state, scramble_len)

    # def scramble_vec(self, states: torch.Tensor, scramble_len: int) -> None:
        # """Scramble `states` in-place with `scramble_len` random moves."""
        # tcv.scramble_vec(states, scramble_len)

    def _action_to_move(self, action: int) -> int:
        return action - 6 if action < 6 else action - 5

    def _actions_to_moves(self, actions: torch.Tensor) -> torch.Tensor:
        #  0  1  2  3  4  5  6  7  8  9 10 11
        # -5 -4 -3 -2 -1  0  1  2  3  4  5  6
        # -1 -2 -3 -4 -5 -6  1  2  3  4  5  6
        # return actions - 5 - (actions < 6).type(torch.long)
        return actions - torch.where(actions < 6, 6, 5)

    def _moves_to_actions(self, moves: torch.Tensor) -> torch.Tensor:
        return moves + torch.where(moves < 0, 6, 5)

    def perform_action(self, state: torch.Tensor, action: int) -> None:
        """Perform `action` on the `state` in-place."""
        move = self._action_to_move(action)
        tcu.make_move(state, move)

    def perform_action_vec(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        """Perform `actions` on the corresponding `states` in-place."""
        moves = self._actions_to_moves(actions)
        tcv.make_move_vec(states, moves)

    def invert_actions(self, actions: torch.Tensor) -> torch.Tensor:
        # TODO: inversion table could be faster
        moves = self._actions_to_moves(actions)
        inverted_actions = self._moves_to_actions(-moves)
        return inverted_actions

    def make_features(self, state: torch.Tensor) -> torch.Tensor:
        return state.flatten()

    def make_features_vec(self, states: torch.Tensor) -> torch.Tensor:
        return states.flatten(1)

    # def make_stategoal(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # """Returns a vector representing the complete state-goal."""
        # return torch.stack((state, goal)).flatten()

    # def make_stategoal_vec(
        # self, states: torch.Tensor, goals: torch.Tensor
    # ) -> torch.Tensor:
        # """Returns a vector representing the complete state-goal.
        
        # Params:
            # - `states`: vector of states
            # - `goals`: vector of goals or a single goal
        # """
        # states_b, goals_b = torch.broadcast_tensors(states.flatten(1), goals.flatten(1))
        # return torch.hstack((states_b, goals_b))
