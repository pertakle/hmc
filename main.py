from deepercube.env.cube_env import RubiksCubeEnvVec
import gymnasium as gym
import deepercube.kostka.kostka_vek as kv
import numpy as np


def print_state(state):
    kv.print_kostku_vek(state.reshape(-1, 6, 3, 3))

env = gym.wrappers.Autoreset(gym.make("deepercube/RubiksCube-v0", scramble_len=1, ep_limit=1))
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







