from copy import deepcopy
import numpy as np
import gymnasium as gym
from .rainbow import Rainbow
from .replay_buffer import ReplayBuffer


def train_rainbow(
    venv: gym.vector.VectorEnv,
    batch_size: int,
    update_freq: int,
    replay_buffer_size: int,
    eval_each: int,
) -> Rainbow:

    eval_env = deepcopy(venv)
    def evaluate_episode() -> None:
        rewards_total = 0
        dones = np.full(eval_env.num_envs, False)
        states, _ = eval_env.reset()
        while not np.all(dones):
            actions = agent.predict_moves(states, True)
            next_states, rewards, terminated, truncated, _ = eval_env.step(actions)
            rewards_total += rewards @ (~dones)
            dones |= terminated | truncated
            states = next_states
        rewards_total /= eval_env.num_envs
        print(f"Mean {rewards_total:.4f}")



    
    ob_space = venv.single_observation_space
    action_space = venv.single_action_space
    agent = Rainbow(ob_space.nvec[0], ob_space.shape[0], action_space.n)  # type: ignore
    replay_buffer = ReplayBuffer(replay_buffer_size, ob_space.shape)  # type: ignore

    states, _ = venv.reset()
    training = True
    step = 0
    not_reseting = np.full(venv.num_envs, True)
    while training:
        actions = agent.predict_moves(states, False)
        next_states, rewards, terminated, truncated, _ = venv.step(actions)

        replay_buffer.store_replay(
            [
                states[not_reseting],
                actions[not_reseting],
                rewards[not_reseting],
                terminated[not_reseting],
                truncated[not_reseting],
                next_states[not_reseting],
            ]
        )

        if len(replay_buffer) >= batch_size:
            replay = replay_buffer.sample_transitions(batch_size)
            agent.train(*replay)
            if (step % update_freq) == 0:
                agent.copy_weights()

        not_reseting = ~(terminated | truncated)
        states = next_states

        if (step % eval_each) == 0:
            evaluate_episode()

    return agent
