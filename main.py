from deepercube.env.cube_env import RubiksCubeEnvVec
import gymnasium as gym
import deepercube.kostka.kostka_vek as kv
import numpy as np


ENVS = 3
env = gym.make("deepercube/RubiksCube-v0", scramble_len=1, ep_limit=2)
venv = gym.make_vec("deepercube/RubiksCube-v0", num_envs=ENVS, vectorization_mode="sync", scramble_len=1, ep_limit=2)
venv = RubiksCubeEnvVec(ENVS, 1, 2)
print(venv.single_action_space, venv.action_space)
print(venv.observation_space)
s = venv.reset()
ns = venv.step(np.ones(ENVS, dtype=int))
kv.print_kostku_vek(venv._cubes)

print()
print()
print(env.action_space)
print(env.observation_space)


exit()
def reset(env):
    print("reseting")
    state = env.reset()[0]
    ended = np.full(ENVS, False)
    return state, ended

def print_state(state):
    kv.print_kostku_vek(state.reshape(-1, 6, 3, 3))

state, ended = reset(env)
while True:
    print_state(state)
    action = int(input("action: "))
    action = np.array([action], dtype=int)
    next_state, reward, terminated, truncated, info = env.step(action)
    print(reward, terminated, truncated, info)

    dones = np.logical_or(terminated, truncated) 
    ended = np.logical_or(ended, dones)
    if np.all(ended):
        print_state(next_state)
        next_state, ended = reset(env)

    state = next_state

