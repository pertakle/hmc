import argparse
from collections import deque
from hmc.utils.wrappers import PositiveWrapperVec
import gymnasium as gym
from .ppo import PPO
import torch
import numpy as np
import hmc.utils.torch_utils as tut
import hmc.agents.rl_utils.torch_buffers as tbuf
import hmc.agents.rl_utils.torch_her as ther

def evaluate(agent: PPO, env: gym.vector.VectorEnv) -> tuple[float, float]:
    dones = torch.full([env.num_envs], False, dtype=torch.bool, device=tut.get_torch_cube_device())
    returns = torch.zeros(env.num_envs, dtype=torch.float32, device=tut.get_torch_cube_device())

    states = env.reset()[0]
    while not dones.all():
        actions = agent.predict_action_probs(states).argmax(-1)
        states, rewards, terminations, truncations, _ = env.step(actions)

        returns += rewards * ~dones
        dones |= terminations | truncations

    return returns.mean().item(), returns.std().item()


def train_ppo(args: argparse.Namespace) -> PPO:

    env = gym.make_vec(
        args.env,
        size=args.env_size,
        num_envs=args.num_envs,
        scramble_len=args.scramble_len,
        ep_limit=args.ep_limit,
        device=tut.get_torch_cube_device(),
    )
    eval_env = gym.make_vec(
        args.env,
        size=args.env_size,
        num_envs=args.eval_num_envs,
        scramble_len=args.eval_scramble_len,
        ep_limit=args.eval_ep_limit,
        device=tut.get_torch_cube_device(),
    )
    # from hmc.env.cube_env import RubiksCube2x2WrapperVec
    # env = RubiksCube2x2WrapperVec(env)
    # eval_env = RubiksCube2x2WrapperVec(eval_env)
    if args.reward_type == "reward":
        env = PositiveWrapperVec(env)
        eval_env = PositiveWrapperVec(eval_env)

    assert type(env.single_observation_space) == gym.spaces.MultiDiscrete
    assert type(env.single_action_space) == gym.spaces.Discrete

    ob_space = env.single_observation_space
    act_space = env.single_action_space

    agent = PPO(
        args,
        ob_space.nvec[0],
        ob_space.shape[0],
        act_space.n.item(),
        tut.get_torch_cube_device(),
    )

    states = env.reset()[0]
    replay_ep_buffer = tbuf.TorchEpisodeBuffer(
        args.num_envs,
        states[0].shape,
        args.ep_limit,
        tut.get_torch_cube_device(),
        states.dtype,
    )
    training = True
    step = 0
    ep_buffer = []
    # res = False
    while training:
        step += 1
        
        # print(states[:, 1].count_nonzero())
        probs = agent.predict_action_probs(states)
        actions = torch.distributions.Categorical(probs).sample()
        next_states, rewards, terminations, truncations, _ = env.step(actions)
        # if not res:
            # print(states[0].cpu().numpy().reshape(2, 2, 2))
            # print(probs[0].cpu().numpy())
            # input()
        # res = not res and (terminations[0] or truncations[0])

        # gather train_data
        replay_data = tbuf.TorchReplayData(
            states, actions, rewards, terminations, truncations, next_states
        )
        eps = replay_ep_buffer.store_transitions(replay_data)
        ep_buffer.append(eps)


        # train
        if eps.batch_size() > 0:
            # augment with HER
            # train_eps = [eps]
            for _ in range(args.her_future):
                ep_buffer.append(ther.torch_make_her_future(eps, args.reward_type))
            for _ in range(args.her_final):
                ep_buffer.append(ther.torch_make_her_final(eps, args.reward_type))
            if len(ep_buffer) > args.num_envs / 2:
                train_eps = tbuf.TorchReplayEpData.concatenate(ep_buffer)
                ep_buffer.clear()
                # train step
                agent.train(train_eps)


        if step % args.eval_each == 0:
            mean, std = evaluate(agent, eval_env)
            print(f"Evaluation after {step} steps: {mean:.2f} +-{std:.2f}")

        states = next_states

    return agent
