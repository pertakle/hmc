import numpy as np


class ReplayBuffer:
    def __init__(self, size: int, state_shape: tuple[int, ...], dtype=None) -> None:
        self.size = size
        self.next_index = 0
        self.filled = False
        self.states = np.empty([size, *state_shape], dtype=dtype)
        self.actions = np.empty([size], dtype=int)
        self.rewards = np.empty([size], dtype=float)
        self.next_states = np.empty([size, *state_shape], dtype=dtype)

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
        print("Next states")
        print(self.next_states)

    def store_replay(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
    ) -> None:

        # if the batch fills the buffer many times, store only the last part
        if len(states) > self.size:
            states = states[-self.size :]
            actions = actions[-self.size :]
            rewards = rewards[-self.size :]
            next_states = next_states[-self.size :]

        # batch_size <= self.size
        batch_size = states.shape[0]
        remaining_size = self.size - self.next_index
        clamped_batch_size = min(remaining_size, batch_size)
        remaining_slice = slice(
            self.next_index, self.next_index + clamped_batch_size, 1
        )

        self.states[remaining_slice] = states[:clamped_batch_size]
        self.actions[remaining_slice] = actions[:clamped_batch_size]
        self.rewards[remaining_slice] = rewards[:clamped_batch_size]
        self.next_states[remaining_slice] = next_states[:clamped_batch_size]

        if self.next_index + clamped_batch_size >= self.size:
            self.filled = True
        self.next_index = (self.next_index + clamped_batch_size) % self.size

        # if the batch hits the end of buffer
        if clamped_batch_size < batch_size:
            self.store_replay(
                states[clamped_batch_size:],
                actions[clamped_batch_size:],
                rewards[clamped_batch_size:],
                next_states[clamped_batch_size:],
            )

    def sample_transitions(
        self, num_samples: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(
            self.size if self.filled else self.next_index, num_samples
        )
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
        )

