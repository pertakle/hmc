import numpy as np
from typing import NamedTuple


class ReplayData(NamedTuple):
    """
    Transitions from the environment stored as 6-tuple of batched ndarrays:
        - states: (B, state_shape), `Any`
        - actions: (B,), `int`
        - rewards: (B,), `float`
        - terminated: (B,), `bool`
        - truncated: (B,), `bool`
        - next_states: (B, state_shape), `Any`
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_states: np.ndarray

    def batch_size(self) -> int:
        return len(self.states)

    @staticmethod
    def concatenate(data: list["ReplayData"]) -> "ReplayData":
        return ReplayData(
            *(np.concatenate([x[i] for x in data], axis=0) for i in range(6))
        )


class ReplayEpData(NamedTuple):
    """
    Episodes stored as 4-tuple of ndarrays:
        - states: (num_envs, ep_limit, state_shape), `Any`
        - actions: (num_envs, ep_limit), `int`
        - rewards: (num_envs, ep_limit), `float`
        - next_states: (num_envs, ep_limit state_shape), `Any`
        - lengths: (num_envs,), `int`
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    lengths: np.ndarray

    def unroll(self) -> ReplayData:
        mask = np.arange(self.states.shape[1])[None, :] < self.lengths[:, None]
        states = self.states[mask]
        actions = self.actions[mask]
        rewards = self.rewards[mask]
        next_states = self.next_states[mask]

        num_eps = self.states.shape[0]
        _all_eps = np.arange(num_eps)
        last_indices = self.lengths - 1

        # split states and goals in a dimension
        stategoal_size = states.shape[-1]
        goal_index = stategoal_size // 2
        last_next_states = self.next_states[_all_eps, last_indices]
        # goal reached
        terminated_last = np.all(
            last_next_states[:, :goal_index]  # state
            == last_next_states[:, goal_index:],  # goal
            axis=1,
        )

        terminated = np.full(self.rewards.shape, False)
        terminated[_all_eps, last_indices] = terminated_last
        terminated = terminated[mask]

        # episode limit hit
        truncated = np.full(self.rewards.shape, False)
        truncated[:, -1] = True
        truncated = truncated[mask]

        return ReplayData(states, actions, rewards, terminated, truncated, next_states)


class HERBuffer:
    """
    Buffer to wait for the complete episode and
    augment transitions with HER.
    """

    def __init__(
        self,
        num_envs: int,
        state_shape: tuple[int, ...],
        max_ep_len: int,
        state_dtype=None,
    ) -> None:
        self.num_envs = num_envs
        self.states = np.empty([num_envs, max_ep_len, *state_shape], dtype=state_dtype)
        self.actions = np.empty([num_envs, max_ep_len], dtype=int)
        self.rewards = np.empty([num_envs, max_ep_len], dtype=float)
        self.next_states = np.empty(
            [num_envs, max_ep_len, *state_shape], dtype=state_dtype
        )
        self.next_indices = np.zeros(num_envs, dtype=int)
        self.reseting = np.full(num_envs, False)

    def _recompute_episode_lengths(
        self, next_states: np.ndarray, prev_lengths: np.ndarray
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

    def _make_her_final(self, episodes: ReplayEpData) -> ReplayEpData:
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
        new_goals_indices = ep_lengths - 1  # TODO: try future too
        new_goals = next_states[np.arange(num_episodes), new_goals_indices][
            :, goal_index:
        ]

        # set new goals
        her_states[:, :, goal_index:] = new_goals[:, np.newaxis]
        her_next_states[:, :, goal_index:] = new_goals[:, np.newaxis]

        her_ep_lengths = self._recompute_episode_lengths(her_next_states, ep_lengths)

        return ReplayEpData(
            her_states, her_actions, her_rewards, her_next_states, her_ep_lengths
        )

    def store_transitions(
        self,
        transitions: ReplayData,
    ) -> ReplayData | None:
        """
        Stores the newly observed transitions including those between episodes.

        Params:
            transitions: batched transitions, see `ReplayData`

        Returns:
            Transitions to store into replay buffer,
            or None if there aren't any.
        """
        states, actions, rewards, terminated, truncated, next_states = transitions
        # store transition for each env
        _all = np.arange(self.num_envs)
        self.states[_all, self.next_indices] = states
        self.actions[_all, self.next_indices] = actions
        self.rewards[_all, self.next_indices] = rewards
        self.next_states[_all, self.next_indices] = next_states
        self.next_indices += 1

        # get all terminated or truncated episodes
        finished = (terminated | truncated) & ~self.reseting
        finished_ep = ReplayEpData(
            self.states[finished],
            self.actions[finished],
            self.rewards[finished],
            self.next_states[finished],
            self.next_indices[finished],
        )
        self.reseting = finished
        self.next_indices *= (~finished) | (~self.reseting)

        if np.count_nonzero(finished) == 0:
            return None

        # Augment with HER transitions
        # create k fictious episodes
        # TODO: k times
        her_replay = self._make_her_final(finished_ep)

        # unroll them and concatenate them
        data = finished_ep.unroll()
        her_data = her_replay.unroll()

        return ReplayData.concatenate([data, her_data])


class ReplayBuffer:
    def __init__(
        self, size: int, state_shape: tuple[int, ...], state_dtype=None
    ) -> None:
        self.size = size
        self.next_index = 0
        self.filled = False
        self.states = np.empty([size, *state_shape], dtype=state_dtype)
        self.actions = np.empty([size], dtype=int)
        self.rewards = np.empty([size], dtype=float)
        self.terminated = np.empty([size], dtype=bool)
        self.truncated = np.empty([size], dtype=bool)
        self.next_states = np.empty([size, *state_shape], dtype=state_dtype)

        self.buff_data = ReplayData(
            self.states,
            self.actions,
            self.rewards,
            self.terminated,
            self.truncated,
            self.next_states,
        )

        self.priorities = np.ones(size)

    def __len__(self) -> int:
        if self.filled:
            return self.size
        return self.next_index

    def print(self) -> None:
        print(f"{self.size=}, {self.next_index=}, {self.filled=}")
        print("States:")
        print(self.states)
        print()
        print("Actions")
        print(self.actions)
        print()
        print("Rewards")
        print(self.rewards)
        print()
        print("Terminated", self.terminated)
        print("Truncated", self.truncated)
        print("Next states")
        print(self.next_states)

    def _store_replay(self, replay_data: ReplayData, start: int) -> None:
        # if the batch fills the buffer many times, store only the last part
        data_size = replay_data[0].shape[0]
        batch_size = data_size - start
        if batch_size > self.size:
            start = data_size - self.size
            batch_size = data_size - start

        # batch_size <= self.size
        remaining_size = self.size - self.next_index
        clamped_batch_size = min(remaining_size, batch_size)

        for buf, data in zip(self.buff_data, replay_data):
            buf[self.next_index : self.next_index + clamped_batch_size] = data[
                start : start + clamped_batch_size
            ]
        self.priorities[self.next_index : self.next_index + clamped_batch_size] = (
            self.priorities[: len(self)].max() if len(self) > 0 else 1
        )

        if self.next_index + clamped_batch_size >= self.size:
            self.filled = True
        self.next_index = (self.next_index + clamped_batch_size) % self.size

        # if the batch hit the end of buffer
        # this will be called max once, thanks to optimization at the beginning
        if clamped_batch_size < batch_size:
            self._store_replay(replay_data, start + clamped_batch_size)

    def store_replay(
        self,
        replay_data: ReplayData,
    ) -> None:
        """
        Stores replay data to the buffer.

        Parameters:
            `replay_data`: data to be stored, see `ReplayData`
        """
        self._store_replay(replay_data, 0)

    def sample_transitions(
        self, num_samples: int, alpha: float = 0.7, beta: float = 0.5
    ) -> tuple[ReplayData, np.ndarray, np.ndarray]:

        num_stored = len(self)
        probs = self.priorities[:num_stored] ** alpha
        probs /= probs.sum()
        indices = np.random.choice(num_stored, num_samples, p=probs)

        isw = (num_stored * probs[indices]) ** (-beta)
        isw /= isw.max()

        return ReplayData(*(b[indices] for b in self.buff_data)), indices, isw

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.priorities[indices] = priorities
