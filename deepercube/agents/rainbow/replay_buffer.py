import numpy as np


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

        self.buff_data = [
            self.states,
            self.actions,
            self.rewards,
            self.terminated,
            self.truncated,
            self.next_states,
        ]

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

    def _store_replay(self, replay_data: list[np.ndarray], start: int) -> None:
        # if the batch fills the buffer many times, store only the last part
        batch_size = replay_data[0].shape[0] - start
        if batch_size > self.size:
            start = replay_data[0].shape[0] - self.size
            batch_size = replay_data[0].shape[0] - start

        # batch_size <= self.size
        remaining_size = self.size - self.next_index
        clamped_batch_size = min(remaining_size, batch_size)

        for buf, data in zip(self.buff_data, replay_data):
            buf[self.next_index : self.next_index + clamped_batch_size] = data[
                start : start + clamped_batch_size
            ]

        if self.next_index + clamped_batch_size >= self.size:
            self.filled = True
        self.next_index = (self.next_index + clamped_batch_size) % self.size

        # if the batch hit the end of buffer
        # this will be called max once, thanks to optimization at the beginning
        if clamped_batch_size < batch_size:
            self._store_replay(replay_data, start + clamped_batch_size)

    def store_replay(
        self,
        replay_data: list[np.ndarray],
    ) -> None:
        """
        Stores replay data to the buffer.

        Parameters:
            `replay_data`: list of ndarrays [states, actions, rewards, terminated, truncated, next_states]

        Shapes:
            states: (B, state_shape)
            actions: (B,)
            rewards: (B,)
            terminated: (B,)
            truncated: (B,)
            next_states: (B, state_shape)
        """
        self._store_replay(replay_data, 0)

    def sample_transitions(
        self, num_samples: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(
            self.size if self.filled else self.next_index, num_samples
        )
        return tuple(b[indices] for b in self.buff_data)
