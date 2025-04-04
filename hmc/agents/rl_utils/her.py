import numpy as np
from .buffers import ReplayEpData


def recompute_episode_lengths(
    next_states: np.ndarray, prev_lengths: np.ndarray
) -> np.ndarray:
    """
    Recomputes lengths of episodes based on next_states and previous episode lengths
    before modifying the episode.
    Useful when creating HER.


    Params:
        `next_states`: ndarray (num_episodes, max_ep_len, state_shape)
        `prev_lengths`: ndarray (num_episodes,)

    Parameter `prev_lengths` says where the `next_states` are still valid states.

    Returns:
        array of newly computed episode lengths
    """
    EPS, L, S = next_states.shape
    goal_index = S // 2

    next_states_split = next_states.reshape(EPS, L, 2, goal_index)
    new_ends = np.all(next_states_split[:, :, 0] == next_states_split[:, :, 1], axis=-1).astype(np.long)
    # new_ends: [num_episodes, max_ep_len], True if it is a terminal state
    # max_values, max_indices = new_ends.max(1)
    max_values = new_ends.max(1)
    max_indices = new_ends.argmax(1)
    new_ep_lengths = max_indices + 1

    #                   invalid transitions           goal not reached
    invalid_lengths = (max_indices >= prev_lengths) | (max_values == 0)
    new_ep_lengths = ~invalid_lengths * new_ep_lengths + invalid_lengths * prev_lengths
    return new_ep_lengths


def make_her_any(
    episodes: ReplayEpData, new_goals_indices: np.ndarray, reward_type: str
) -> ReplayEpData:
    """
    Creates a new fictious episodes with the *final* strategy.
    The new episodes will make a copy, so that `episodes` stays unchanged.

    Params:
        `episodes`: episodes from which will the new episodes be created

    Returns:
        created episodes
    """
    states, actions, rewards, next_states, ep_lengths = episodes

    # copy episodes
    her_states = states.copy()
    her_actions = actions.copy()
    # her_rewards = rewards.copy()  # rewards are all -1 anyway
    her_next_states = next_states.copy()

    goal_index = states.shape[2] // 2
    # new_goals_indices = ep_lengths - 1
    _brang = np.arange(episodes.batch_size())
    new_goals = next_states[_brang, new_goals_indices][:, :goal_index]

    # set new goals
    her_states[:, :, goal_index:] = new_goals[:, np.newaxis]
    her_next_states[:, :, goal_index:] = new_goals[:, np.newaxis]

    her_ep_lengths = recompute_episode_lengths(her_next_states, ep_lengths)

    if reward_type == "punish":
        her_rewards = rewards.copy()  # rewards are all -1 anyway
    elif reward_type == "reward":
        her_rewards = np.zeros_like(rewards)
        her_rewards[_brang, her_ep_lengths - 1] = 1
    else:
        assert False, "Unknown reward type!"

    return ReplayEpData(
        her_states, her_actions, her_rewards, her_next_states, her_ep_lengths
    )


def make_her_final(episodes: ReplayEpData, reward_type: str) -> ReplayEpData:
    return make_her_any(episodes, episodes.lengths - 1, reward_type)


def make_her_future(episodes: ReplayEpData, reward_type: str) -> ReplayEpData:
    lcm = np.lcm.reduce(episodes.lengths)  # for fair distribution
    goal_indices = np.random.randint(0, lcm, len(episodes.lengths)) % episodes.lengths
    return make_her_any(episodes, goal_indices, reward_type)
