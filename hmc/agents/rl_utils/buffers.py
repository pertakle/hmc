import numpy as np
from typing import NamedTuple, Iterable


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

    def filter(self, mask: np.ndarray) -> "ReplayData":
        return ReplayData(*(d[mask] for d in self))

    @staticmethod
    def concatenate(data: Iterable["ReplayData"]) -> "ReplayData":
        return ReplayData(*(np.concatenate(d, axis=0) for d in zip(*data)))

    @staticmethod
    def empty() -> "ReplayData":
        return ReplayData(*((np.array([]),) * 6))


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

    @staticmethod
    def concatenate(data: Iterable["ReplayEpData"]) -> "ReplayEpData":
        return ReplayEpData(*(np.concatenate(d, axis=0) for d in zip(*data)))

    def batch_size(self) -> int:
        return len(self.states)

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

class NStepBufferVec:

    def __init__(self, n: int, env_num: int, gamma: float) -> None:
        self.n = n
        self.env_num = env_num
        self.gamma = gamma

        self.next_index = 0
        self.filled = False

        self.states = np.array([])
        self.rewards = np.array([])
        self.actions = np.array([])
        self.terminated = np.array([])
        self.truncated = np.array([])
        self.next_states = np.array([])

    def __init(self, states_shape: tuple[int, ...], state_dtype=None) -> None:
        self.states = np.empty([self.env_num, self.n, *states_shape], dtype=state_dtype)
        self.rewards = np.empty([self.env_num, self.n], dtype=float)
        self.actions = np.empty([self.env_num, self.n], dtype=int)
        self.terminated = np.empty([self.env_num, self.n], dtype=bool)
        self.truncated = np.empty([self.env_num, self.n], dtype=bool)
        self.next_states = np.empty(
            [self.env_num, self.n, *states_shape], dtype=state_dtype
        )

    def _inc_next_index(self) -> None:
        self.next_index += 1
        if self.next_index >= self.n:
            self.next_index = 0
            self.filled = True

    def _store_transitions(self, transitions: ReplayData) -> None:
        self.states[:, self.next_index] = transitions.states
        self.actions[:, self.next_index] = transitions.actions
        self.rewards[:, self.next_index] = transitions.rewards
        self.terminated[:, self.next_index] = transitions.terminated
        self.truncated[:, self.next_index] = transitions.truncated
        self.next_states[:, self.next_index] = transitions.next_states

        self._inc_next_index()

    def _get_transitions(self) -> ReplayData | None:
        if not self.filled:
            return None
        rewards = self.rewards[:, self.next_index - 1]
        next_states = self.next_states[:, self.next_index - 1]
        terminated = self.terminated[:, self.next_index - 1]
        truncated = self.truncated[:, self.next_index - 1]

        for ti in range(self.next_index - 2, self.next_index - self.n, -1):
            term = self.terminated[:, ti]
            trun = self.truncated[:, ti]
            done = term | trun
            done_unsq = np.expand_dims(
                done, tuple(range(1, len(self.next_states[:, ti].shape)))
            )

            rewards = self.rewards[:, ti] + self.gamma * rewards * (1 - term)
            next_states = (
                done_unsq * self.next_states[:, ti] + (1 - done_unsq) * next_states
            )
            terminated = done * term + (1 - done) * terminated
            truncated = done * trun + (1 - done) * truncated

        states = self.states[:, self.next_index]
        actions = self.actions[:, self.next_index]
        return ReplayData(
            states,
            actions,
            rewards,
            terminated.astype(bool),
            truncated.astype(bool),
            next_states,
        )

    def step(self, transitions: ReplayData) -> ReplayData | None:
        """
        Returns the new n-step transitions.
        """
        if len(self.states) == 0:
            self.__init(transitions.states.shape[1:], transitions.states.dtype)
        self._store_transitions(transitions)
        return self._get_transitions()


class ReplayBuffer:
    def __init__(self, size: int) -> None:
        self.size = size
        self.next_index = 0
        self.filled = False
        self.states = None
        self.actions = None
        self.rewards = None
        self.terminated = None
        self.truncated = None
        self.next_states = None

        self.buff_data = None

    def __init(self, state_shape: tuple[int, ...], state_dtype=None) -> None:
        self.next_index = 0
        self.filled = False
        self.states = np.empty([self.size, *state_shape], dtype=state_dtype)
        self.actions = np.empty([self.size], dtype=int)
        self.rewards = np.empty([self.size], dtype=float)
        self.terminated = np.empty([self.size], dtype=bool)
        self.truncated = np.empty([self.size], dtype=bool)
        self.next_states = np.empty([self.size, *state_shape], dtype=state_dtype)

        self.buff_data = ReplayData(
            self.states,
            self.actions,
            self.rewards,
            self.terminated,
            self.truncated,
            self.next_states,
        )

        self.priorities = np.ones(self.size)

    def __len__(self) -> int:
        if self.filled:
            return self.size
        return self.next_index

    def _store_replay(self, replay_data: ReplayData, start: int) -> None:
        if self.buff_data is None:
            self.__init(replay_data.states.shape[1:], replay_data.states.dtype)

        # if the batch fills the buffer many times, store only the last part
        data_size = replay_data[0].shape[0]
        batch_size = data_size - start
        if batch_size > self.size:
            start = data_size - self.size
            batch_size = data_size - start

        # batch_size <= self.size
        remaining_size = self.size - self.next_index
        clamped_batch_size = min(remaining_size, batch_size)

        assert self.buff_data is not None
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
        self, num_samples: int, alpha: float = 0.5, beta: float = 0.5
    ) -> tuple[ReplayData, np.ndarray, np.ndarray]:
        """
        for alpha=0, beta=0 it is the same as normal replay buffer

        Returns:
            data, indices, isw
        """

        num_stored = len(self)
        probs = self.priorities[:num_stored] ** alpha
        probs = np.nan_to_num(probs, nan=0)
        probs = np.clip(probs, 0, 1)
        probs /= probs.sum()
        indices = np.random.choice(num_stored, num_samples, p=probs)

        isw = (num_stored * probs[indices]) ** (-beta)
        isw /= isw.max()

        assert self.buff_data is not None
        return ReplayData(*(b[indices] for b in self.buff_data)), indices, isw

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.priorities[indices] = priorities


class EpisodeBuffer:
    """
    Buffer to wait for the complete episode.
    """

    def __init__(
        self,
        num_envs: int,
        state_shape: tuple[int, ...],
        max_ep_len: int,
        state_dtype=None,
    ) -> None:
        padded_max_ep_len = max_ep_len + 1
        self.num_envs = num_envs
        self.states = np.empty(
            [num_envs, padded_max_ep_len, *state_shape], dtype=state_dtype
        )
        self.actions = np.empty([num_envs, padded_max_ep_len], dtype=int)
        self.rewards = np.empty([num_envs, padded_max_ep_len], dtype=float)
        self.next_states = np.empty(
            [num_envs, padded_max_ep_len, *state_shape], dtype=state_dtype
        )
        self.next_indices = np.zeros(num_envs, dtype=int)
        self.reseting = np.full(num_envs, False)

    def store_transitions(
        self,
        transitions: ReplayData,
    ) -> ReplayEpData | None:
        """
        Stores the newly observed transitions including those between episodes.
        It automatically takes care of auto-reset transitions.

        Params:
            transitions: batched transitions, see `ReplayData`

        Returns:
            ReplayEpData to store into replay buffer,
            or None if there aren't any.
        """
        # store transition for each env
        _all = np.arange(self.num_envs)
        self.states[_all, self.next_indices] = transitions.states
        self.actions[_all, self.next_indices] = transitions.actions
        self.rewards[_all, self.next_indices] = transitions.rewards
        self.next_states[_all, self.next_indices] = transitions.next_states
        self.next_indices += 1

        # get all terminated or truncated episodes
        finished = (transitions.terminated | transitions.truncated) & ~self.reseting
        finished_ep = ReplayEpData(
            self.states[finished],
            self.actions[finished],
            self.rewards[finished],
            self.next_states[finished],
            self.next_indices[finished],
        )
        self.next_indices *= ~(finished | self.reseting)
        self.reseting = finished

        if np.count_nonzero(finished) == 0:
            return None

        return finished_ep

