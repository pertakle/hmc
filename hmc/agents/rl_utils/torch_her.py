import functools
import torch
from .torch_buffers import TorchReplayEpData


def torch_recompute_episode_lengths(
    next_states: torch.Tensor, prev_lengths: torch.Tensor
) -> torch.Tensor:
    """
    Recomputes lengths of episodes based on next_states and previous episode lengths
    before modifying the episode.
    Useful when creating HER.


    Params:
        `next_states`: tensor (num_episodes, max_ep_len, state_shape)
        `prev_lengths`: tensor (num_episodes,)

    Parameter `prev_lengths` says where the `next_states` are still valid states,
    i.e. it is an upper bound of the recomputed lengths.

    Returns:
        tensor of newly computed episode lengths
    """
    EPS, L, S = next_states.shape
    goal_index = S // 2

    next_states_split = next_states.reshape(EPS, L, 2, goal_index)
    new_ends = torch.all(next_states_split[:, :, 0] == next_states_split[:, :, 1], dim=-1).type(torch.long)
    # new_ends: [num_episodes, max_ep_len], True if it is a terminal state
    max_values, max_indices = new_ends.max(1)
    new_ep_lengths = max_indices + 1

    #                   invalid transitions           goal not reached
    invalid_lengths = (max_indices >= prev_lengths) | (max_values == 0)
    new_ep_lengths = ~invalid_lengths * new_ep_lengths + invalid_lengths * prev_lengths
    return new_ep_lengths


def torch_make_her_any(
    episodes: TorchReplayEpData, new_goals_indices: torch.Tensor
) -> TorchReplayEpData:
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
    her_states = states.clone()
    her_actions = actions.clone()
    her_rewards = rewards.clone()  # rewards are all -1 anyway
    her_next_states = next_states.clone()

    goal_index = states.shape[2] // 2
    # new_goals_indices = ep_lengths - 1
    new_goals = next_states[
        torch.arange(episodes.batch_size(), dtype=torch.long, device=states.device),
        new_goals_indices,
    ][:, goal_index:]

    # set new goals
    her_states[:, :, goal_index:] = new_goals[:, None]
    her_next_states[:, :, goal_index:] = new_goals[:, None]

    her_ep_lengths = torch_recompute_episode_lengths(her_next_states, ep_lengths)

    return TorchReplayEpData(
        her_states, her_actions, her_rewards, her_next_states, her_ep_lengths
    )


def torch_make_her_final(episodes: TorchReplayEpData) -> TorchReplayEpData:
    return torch_make_her_any(episodes, episodes.lengths - 1)


def torch_make_her_future(episodes: TorchReplayEpData) -> TorchReplayEpData:
    lcm = int(functools.reduce(torch.lcm, episodes.lengths))
    # lcm = np.lcm.reduce(episodes.lengths)  # for fair distribution
    goal_indices = (
        torch.randint(
            0,
            lcm,
            [episodes.batch_size()],
            dtype=torch.long,
            device=episodes.get_device(),
        )
        % episodes.lengths
    )
    return torch_make_her_any(episodes, goal_indices)
