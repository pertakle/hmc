from copy import deepcopy
import numpy as np
import gymnasium as gym
from .rainbow import Rainbow
from .replay_buffer import ReplayBuffer, ReplayData, HERBuffer


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
        print(f"Evaluation after {step} steps: {rewards_total:.4f}")

    ob_space = venv.single_observation_space
    action_space = venv.single_action_space
    agent = Rainbow(ob_space.nvec[0], ob_space.shape[0], action_space.n)  # type: ignore
    replay_buffer = ReplayBuffer(replay_buffer_size, ob_space.shape)  # type: ignore
    her_buffer = HERBuffer(venv.num_envs, ob_space.shape, venv._ep_limit)

    states, _ = venv.reset()
    training = True
    step = 0
    while training:
        step += 1
        actions: np.ndarray = agent.predict_moves(states, False)  # type: ignore
        next_states, rewards, terminated, truncated, _ = venv.step(actions)

        transition = ReplayData(
            states,
            actions,
            rewards,
            terminated,
            truncated,
            next_states,
        )
        replay_data = her_buffer.store_transitions(transition)
        if replay_data is not None:
            replay_buffer.store_replay(replay_data)

        if len(replay_buffer) >= batch_size:
            replay, replay_indices, replay_isw = replay_buffer.sample_transitions(
                batch_size
            )
            td_errors: np.ndarray = agent.train(*replay, replay_isw)  # type: ignore
            replay_buffer.update_priorities(replay_indices, td_errors)
            if (step % update_freq) == 0:
                agent.copy_weights()

        states = next_states

        if (step % eval_each) == 0:
            evaluate_episode()

    return agent
