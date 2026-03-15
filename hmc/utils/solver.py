import torch
from hmc.problems.baseproblem import Problem
from typing import Callable, Tuple


def unique(x: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implementaion of `unique` that returns indices instead of their inversion.

    Returns:
        tuple unique entries, indices in the original tensor `x`

    Taken from:
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-1072093200
    """
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)

def solve_beam_uniuniversal_stateaction(
    problem: Problem,
    goal: torch.Tensor,
    heuristic_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    time_limit: int,
    beam_size: int,
    init_h_value: float,
    filter_states: bool = False,
    init_state: torch.Tensor|None = None
) -> int:
    """
    Params:
        - `problem`: problem class
        - `goal`: goal to reach from the initial state
        - `heuristic_fn(stategoals, parent_h)`: calculates the heuristic values of every action
            stategoals shape (B, 2*S), parent_h shape (B,)
            returns: tensor (B, 2*S, A)
        - `time_limit`: maximum number of steps
        - `beam_size`: maximum size of the beam
        - `init_h_value`: initial heuristic value of the initial state
        - `filter_states`: if True the beam will contain only unique states
        - `init_state`: initial state of the search

    Returns:
        lenght of the shortest path found, -1 if the goal was not achieved
    """

    states = problem.new_state_vec(1) if init_state is None else init_state[None]
    h = torch.full([1], init_h_value, dtype=torch.float, device=problem.device)

    if problem.is_solved(states[0], goal):
        return 0

    # TODO: handle already solved state
    for t in range(time_limit):
        assert len(states) <= beam_size

        stategoals = problem.make_stategoal_vec(states, goal[None])
        ha = heuristic_fn(stategoals, h).flatten()

        next_states = problem.perform_all_actions_vec(states).flatten(0, 1)
        if filter_states:
            
            # Zp = 59393  # TODO: try higher prime numbers
            # hash_vector = torch.randint(1, Zp, [problem.state_size], device=problem.device)
            # hash_values = torch.sum(next_states.view(-1, problem.state_size) * hash_vector, -1)

            # _, uniq_indices = unique(hash_values)

            # next_states = next_states[uniq_indices]
            # ha = ha[uniq_indices]


            next_states, uniq_indices = unique(next_states, 0)
            ha = ha[uniq_indices]

        best_indices = ha.argsort()[-beam_size:]
        states = next_states[best_indices]
        h = ha[best_indices]

        if problem.is_solved_vec(states, goal).any():
            return t + 1
    return -1

def solve_beam_uniuniversal(
    problem: Problem,
    goal: torch.Tensor,
    heuristic_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    time_limit: int,
    beam_size: int,
    init_h_value: float,
    filter_states: bool = False,
    init_state: torch.Tensor|None = None
) -> int:
    """
    Params:
        - `heuristic_fn(stategoals, parent_h)`: calculates the heuristic values of the stategoals
            stategoals shape (B, 2*S), parent_h shape (B,)
        - `scramble_len`: scramble length
        - `beam_size`: maximum size of the beam
        - `time_limit`: maximum number of steps
        - `init_h_value`: initial heuristic value of the initial state
        - `filter_states`: if True the beam will contain only unique states
        - `init_state`: initial state of the search

    Returns:
        lenght of the shortest path found, -1 if the goal was not achieved
    """

    states = problem.new_state_vec(1) if init_state is None else init_state[None]
    h = torch.full([1], init_h_value, dtype=torch.float, device=problem.device)

    if problem.is_solved(states[0], goal):
        return 0

    # TODO: handle already solved state
    for t in range(time_limit):
        assert len(states) <= beam_size
        child_states = problem.perform_all_actions_vec(states)
        h = torch.repeat_interleave(h, problem.num_actions)

        next_states = child_states.flatten(0, 1)
        assert (len(next_states) == len(h)) or print(next_states.shape, h.shape)

        if filter_states:
            next_states, uniq_indices = unique(next_states, 0)
            h = h[uniq_indices]

        stategoals = problem.make_stategoal_vec(next_states, goal[None])
        h_next = heuristic_fn(stategoals, h)
        best_indices = h.argsort()[-beam_size:]

        states = next_states[best_indices]
        h = h_next[best_indices]

        if problem.is_solved_vec(states, goal).any():
            return t + 1
    return -1


