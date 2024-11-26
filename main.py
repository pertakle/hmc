import deepercube.env
import gymnasium as gym
from deepercube.agents.rainbow.train_rainbow import train_rainbow

venv = gym.make_vec("deepercube/RubiksCube-v0", num_envs=64, scramble_len=1, ep_limit=5)
train_rainbow(
    venv, batch_size=1024, update_freq=32_000, replay_buffer_size=1_000_000, eval_each=10_000
)
exit()

from deepercube.agents.rainbow.replay_buffer import ReplayBuffer
import numpy as np

buff = ReplayBuffer(5, (2,), int)
buff.print()
for _ in range(8):
    states = np.random.choice(10, [1, 2])
    actions = np.random.choice(2, [1])
    rewards = np.random.choice(2, [1])
    terminated = np.full([1], False)
    truncated = np.full([1], True)
    next_states = np.random.choice(10, [1, 2])
    replay_data = [states, actions, rewards, terminated, truncated, next_states]
    print(f"adding {replay_data}")
    buff.store_replay(replay_data)
    buff.print()
    input()
print("sampled")
print(buff.sample_transitions(10))
exit()


from deepercube.env.cube_env import RubiksCubeEnvVec
import gymnasium as gym
import deepercube.kostka.kostka_vek as kv
import numpy as np


def print_state(state):
    kv.print_kostku_vek(state.reshape(-1, 6, 3, 3))


env = gym.wrappers.Autoreset(
    gym.make("deepercube/RubiksCube-v0", scramble_len=1, ep_limit=1)
)
env = gym.make_vec("deepercube/RubiksCube-v0", num_envs=2, scramble_len=1, ep_limit=2)

state = env.reset()[0]
terminated, truncated = False, False
while True:
    print_state(state)
    action = np.array(input(f"{env.num_envs} actions: ").split(), dtype=int)
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"{terminated=}")
    print(f"{truncated=}")
    state = next_state
