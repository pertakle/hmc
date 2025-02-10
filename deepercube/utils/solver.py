import numpy as np
import deepercube.kostka.kostka as ko
import deepercube.kostka.kostka_vek as kv
from typing import Callable

def actions_to_moves(actions: np.ndarray) -> np.ndarray:
    """
    Converts `actions` (0..11) to moves (-6..1,1..6).
    """
    minus_moves = (actions > 5).astype(np.int64)
    moves = actions + 1
    moves -= 2 * actions * minus_moves
    moves += 4 * minus_moves
    return moves

def solve_beam_universal(
    goal_cube: ko.Kostka,
    action_heuristic: Callable[[np.ndarray, np.ndarray], np.ndarray],
    beam_size: int,
    limit: int,
) -> int:
    def add_goal(cubes: kv.KostkaVek, goal: ko.Kostka) -> np.ndarray:
        CUBE_FEATURES = 6 * 3 * 3
        states_goals = np.empty([len(cubes), 2 * CUBE_FEATURES])
        states_goals[:, :CUBE_FEATURES] = cubes.reshape(len(cubes), CUBE_FEATURES)
        states_goals[:, CUBE_FEATURES:] = goal.reshape(1, CUBE_FEATURES)
        return states_goals

    ACTIONS = 12

    cubes = kv.nova_kostka_vek(1)
    values = np.zeros(1)
    for step in range(limit + 1):
        if kv.je_stejna(cubes, goal_cube).any():
            return step
        values = action_heuristic(add_goal(cubes, goal_cube), values)
        best_indices = values.reshape(-1).argsort()[-beam_size:]

        cubes = cubes[best_indices // ACTIONS]
        moves = actions_to_moves(best_indices % ACTIONS)

        kv.tahni_tah_vek(cubes, moves)

    return limit + 1


