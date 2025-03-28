import numpy as np
from hmc.nn.her_cube_agent import HERCubeAgent
from typing import Tuple, List
import hmc.kostka.kostka_vek as kv


def compute_ep_lengths(episodes: np.ndarray) -> np.ndarray:
    EPISODES = len(episodes)
    her_ep_lengths = np.zeros(EPISODES, dtype=int)
    not_dones = np.full(EPISODES, True)
    for step in range(episodes.shape[1] - 1):
        her_ep_lengths += not_dones
        not_terminated = np.logical_not(stategoal_terminal(episodes[:, step + 1]))
        not_dones = np.logical_and(not_dones, not_terminated)
    return her_ep_lengths


def get_last_next_stategoals(
    episodes: np.ndarray, ep_lengths: np.ndarray
) -> np.ndarray:
    return np.array([episodes[i, l] for i, l in enumerate(ep_lengths)])


def set_goals(episodes: np.ndarray, goals: np.ndarray) -> np.ndarray:
    CUBE_FEATURES = 6 * 3 * 3
    if len(goals.shape) != len(episodes.shape):
        goals = goals[:, np.newaxis]
    new_episodes = episodes.copy()
    new_episodes[:, :, CUBE_FEATURES:] = goals
    return new_episodes


EpData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def generate_episodes_vec(
    agent: HERCubeAgent, sample_moves: int, move_limit: int, episodes: int
) -> EpData:
    """
    Generates `episodes` vectorized episodes with `sample_moves` lengths.
    Goals are randomly scrambled cubes with `sample_moves` uniformly random moves.

    Returns:
        - State-Goals: ndarray, dtype=float, shape=[`episodes`, `move_limit + 1`, 2 * cube features]
        - Actions: ndarray, dtype=float, shape=[`episodes`, `move_limit`]
        - Rewards: ndarray, dtype=float, shape=[`episodes`, `move_limit`]
            reward is `-1` for each action, episode terminates when reaching goal
        - Episode lengths: ndarray, dtype=int, shape=[`episodes`]
    """

    STATE_GOAL_FEATURES = 2 * 6 * 3 * 3  # two flattened cubes
    goals = kv.nova_kostka_vek(episodes)
    kv.zamichej_nahodnymi_tahy_vek(goals, sample_moves)

    states = kv.nova_kostka_vek(episodes)
    dones = np.full(episodes, False)

    ep_stategoals = np.zeros([episodes, move_limit + 1, STATE_GOAL_FEATURES])
    ep_actions = np.zeros([episodes, move_limit])
    ep_rewards = -np.ones([episodes, move_limit])  # NOTE: will be handled by Env

    for step in range(move_limit + 1):
        solved = kv.je_stejna(states, goals)
        dones = np.logical_or(dones, solved)

        # NOTE: breaks last episode states
        # if np.all(dones):
        #     break

        # choose an action in the env
        stategoals = merge_to_stategoals(states, goals)  # reshape?
        actions = agent.predict_action(stategoals, False)

        # store the transitions
        ep_stategoals[:, step] = stategoals.reshape([episodes, -1])
        if step < move_limit:  # if not end of episode
            ep_actions[:, step] = actions

        # make the action
        moves = actions_to_moves(actions)
        kv.tahni_tah_vek(states, moves)

    ep_lengths = compute_ep_lengths(ep_stategoals)
    return ep_stategoals, ep_actions, ep_rewards, ep_lengths


def stategoal_terminal(stategoals: np.ndarray) -> np.ndarray:
    CUBE_FEATURES = 6 * 3 * 3
    return kv.je_stejna(
        stategoals[:, :CUBE_FEATURES].reshape(-1, 6, 3, 3),
        stategoals[:, CUBE_FEATURES:].reshape(-1, 6, 3, 3),
    )


def make_her_last_state(
    episodes: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    ep_lengths: np.ndarray,
) -> EpData:
    """
    Makes new transitions with goals reached at the end of episodes.
    NOTE: Keeps episodes.shape with the same episodes paddings.

    Params:
        - episodes:
            shape=[num episodes, padded ep lengths + 1, 2 * CUBE FEATURES],
            episodes contain the final state as the last state
            `state_goals.shape[1] > ep_lengths `
        - actions:
            shape=[num episodes, padded ep lengths]
        - returns:
            shape=[num episodes, padded ep lengths]
        - ep_lengths:
            vector of episodes lengths
            shape=[num episodes]

    Returns:
        - HER episodes with replaced goals
        - actions: just a copy
        - HER returns
        - ep lengths: jus a copy
    """

    CUBE_FEATURES = 6 * 3 * 3
    her_last_next_stategoals = get_last_next_stategoals(episodes, ep_lengths)

    her_goals = her_last_next_stategoals[:, :CUBE_FEATURES]
    her_episodes = set_goals(episodes, her_goals)

    her_actions = actions.copy()
    her_rewards = rewards.copy()
    her_ep_lengths = compute_ep_lengths(her_episodes)

    return her_episodes, her_actions, her_rewards, her_ep_lengths


def unroll_padded_episodes(
    episodes: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    ep_lengths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unrolls all episodes and actions into a batch of data.

    Params:
        - episodes:
        - actions
        - rewards
        - ep_lengths

    Returns:
        - stategoals: batch of stategoals
        - actions: 1d vector of actions
        - rewards: 1d vector of rewards
    """
    CUBE_FEATURES = 6 * 3 * 3
    num_transitions = ep_lengths.sum()
    unrolled_stategoals = np.zeros([num_transitions, CUBE_FEATURES * 2], dtype=int)
    unrolled_actions = np.zeros(num_transitions, dtype=int)
    unrolled_rewards = np.zeros(num_transitions)

    # TODO: vektorizace
    transition = 0
    for ep in range(len(ep_lengths)):
        for step in range(ep_lengths[ep]):
            unrolled_stategoals[transition] = episodes[ep, step]
            unrolled_actions[transition] = actions[ep, step]
            unrolled_rewards[transition] = rewards[ep, step]

            transition += 1
    return unrolled_stategoals, unrolled_actions, unrolled_rewards


BatchedData = Tuple[np.ndarray, np.ndarray, np.ndarray]


def merge_data(eps_data: List[BatchedData]) -> BatchedData:
    """Converts list of multiple episodes data into a single batch of stategoals, actions and rewards."""
    stategoals = np.vstack([x[0] for x in eps_data])
    actions = np.hstack([x[1] for x in eps_data])
    rewards = np.hstack([x[2] for x in eps_data])
    return stategoals, actions, rewards


def merge_to_stategoals(states: np.ndarray, goals: np.ndarray) -> np.ndarray:
    """
    Merges `states` with their corresponding `goals`.
    Returns:
        merged stategoals
    """
    assert np.prod(states.shape) == np.prod(goals.shape)
    batch_size = states.shape[0]
    CUBE_LEN = 6 * 3 * 3
    state_goal = np.empty([batch_size, 2 * CUBE_LEN], dtype=states.dtype)
    state_goal[:, :CUBE_LEN] = states.reshape(batch_size, CUBE_LEN)
    state_goal[:, CUBE_LEN:] = goals.reshape(-1, CUBE_LEN)
    return state_goal


def actions_to_moves(actions: np.ndarray) -> np.ndarray:
    """
    Converts `actions` (0..11) to moves (-6..1,1..6).
    """
    minus_moves = (actions > 5).astype(np.int64)
    moves = actions + 1
    moves -= 2 * actions * minus_moves
    moves += 4 * minus_moves
    return moves


def moves_to_actions(moves: np.ndarray) -> np.ndarray:
    """
    Converts `moves` (-6..1,1..6) to actions (0..11).
    """
    minus_moves = (moves < 0).astype(np.int64)
    actions = moves.astype(np.int64) - minus_moves * 6
    actions -= 2 * actions * minus_moves
    actions -= 1
    return actions


def generate_batch(
    agent: HERCubeAgent, episodes: int, sample_moves: int, move_limit: int
) -> BatchedData:

    ep_data = generate_episodes_vec(agent, sample_moves, move_limit, episodes)
    her_ep_data = make_her_last_state(*ep_data)

    data = unroll_padded_episodes(*ep_data)
    her_data = unroll_padded_episodes(*her_ep_data)

    return merge_data([data, her_data])
