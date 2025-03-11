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

    Parameter `prev_lengths` says where the `next_states` are still valid states.

    Returns:
        tensor of newly computed episode lengths
    """
    goal_index = next_states.shape[2] // 2
    new_lengths = torch.zeros_like(prev_lengths)
    finished = torch.full(
        [len(prev_lengths)], False, dtype=torch.bool, device=prev_lengths.device
    )
    for t in range(prev_lengths.max()):
        # assuming all episodes have at least one transition
        new_lengths += ~finished
        next_states_t = next_states[:, t]

        # check if the new episodes ended
        finished_t = torch.all(
            next_states_t[:, :goal_index] == next_states_t[:, goal_index:], dim=1
        )  # new goal reached
        finished_t |= prev_lengths <= t  # previous episode ended
        finished |= finished_t
    return new_lengths


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
