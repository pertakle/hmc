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
    goal_index = next_states.shape[2] // 2
    new_lengths = np.zeros_like(prev_lengths)
    finished = np.full(len(prev_lengths), False)
    for t in range(prev_lengths.max()):
        # assuming all episodes have at least one transition
        new_lengths += ~finished
        next_states_t = next_states[:, t]

        # check if the new episodes ended
        finished_t = np.all(
            next_states_t[:, :goal_index] == next_states_t[:, goal_index:], axis=1
        )  # new goal reached
        finished_t |= prev_lengths <= t  # previous episode ended
        finished |= finished_t
    return new_lengths


def make_her_any(episodes: ReplayEpData, new_goals_indices: np.ndarray) -> ReplayEpData:
    """
    Creates a new fictious episodes with the *final* strategy.
    The new episodes will make a copy, so that `episodes` stays unchanged.

    Params:
        `episodes`: episodes from which will the new episodes be created

    Returns:
        created episodes
    """
    states, actions, rewards, next_states, ep_lengths = episodes
    num_episodes = len(states)

    # copy episodes
    her_states = states.copy()
    her_actions = actions.copy()
    her_rewards = rewards.copy()  # rewards are all -1 anyway
    her_next_states = next_states.copy()

    goal_index = states.shape[2] // 2
    # new_goals_indices = ep_lengths - 1
    new_goals = next_states[np.arange(num_episodes), new_goals_indices][:, goal_index:]

    # set new goals
    her_states[:, :, goal_index:] = new_goals[:, np.newaxis]
    her_next_states[:, :, goal_index:] = new_goals[:, np.newaxis]

    her_ep_lengths = recompute_episode_lengths(her_next_states, ep_lengths)

    return ReplayEpData(
        her_states, her_actions, her_rewards, her_next_states, her_ep_lengths
    )


def make_her_final(episodes: ReplayEpData) -> ReplayEpData:
    return make_her_any(episodes, episodes.lengths - 1)


def make_her_future(episodes: ReplayEpData) -> ReplayEpData:
    lcm = np.lcm.reduce(episodes.lengths)  # for fair distribution
    goal_indices = np.random.randint(0, lcm, len(episodes.lengths)) % episodes.lengths
    return make_her_any(episodes, goal_indices)
