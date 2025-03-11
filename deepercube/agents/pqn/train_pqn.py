import argparse
import gymnasium as gym
from .pqn import PQN
import torch
import numpy as np
import deepercube.kostka.torch_cube_vec as tcv
import deepercube.utils.torch_utils as tut
import deepercube.agents.rl_utils.torch_buffers as tbuf
import deepercube.agents.rl_utils.torch_her as ther


def make_train_data(
    agent: PQN, episodes: tbuf.TorchReplayEpData, l: float, gamma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns batched data for training
    as triple: states, actions, target_values

    Args:
        `agent`: PQN agent
        `episodes`: episodes to train on
        `l`: lambda for Q(lambda) return
        `gamma`: discount factor
    """
    B = episodes.batch_size()
    T = episodes.states.shape[1]
    next_q = agent.predict_q(episodes.next_states.reshape(B*T, -1)).reshape(B, T, -1)
    next_v = next_q.max(-1).values

    end_next_stategoals = episodes.next_states[range(B), episodes.lengths].reshape(B, 2, -1)
    end_terminated = torch.all(end_next_stategoals[:, 0] == end_next_stategoals[:, 1], dim=1)

    returns = torch.zeros_like(episodes.rewards)
    for t in range(T-1, -1, -1):
        # this transition terminated
        terminated = (t == (episodes.lengths - 1)) & end_terminated

        # continue: l * ret[t+1] + (1 - l) * v_next[t]
        # term: 0
        # trun: v_next
        ql_next_continue = l * returns[:, t+1] + (1 - l) * next_v[:, t]
        ql_next_term = torch.zeros_like(returns[:, t])
        ql_next_trun = next_v[:, t]

        # TODO
        ql_next = ...
        returns[:, t] = episodes.rewards[:, t] + gamma * ql_next
    return ...


def train_pqn(args: argparse.Namespace) -> PQN:

    env = gym.make_vec(
        "deepercube/TorchRubiksCube-v0",
        num_envs=args.num_envs,
        scramble_len=args.scramble_len,
        ep_limit=args.ep_limit,
    )
    eval_env = gym.make_vec(
        "deepercube/TorchRubiksCube-v0",
        num_envs=args.eval_num_envs,
        scramble_len=args.eval_scramble_len,
        ep_limit=args.eval_ep_limit,
    )

    assert type(env.single_observation_space) == gym.spaces.MultiDiscrete
    assert type(env.single_action_space) == gym.spaces.Discrete

    ob_space = env.single_observation_space
    act_space = env.single_action_space

    agent = PQN(
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
    while training:
        step += 1

        # make steps in the environments
        epsilon = np.interp(
            step, [1, args.epsilon_final_at], [args.epsilon, args.epsilon_final]
        )
        actions = agent.predict_egreedy_actions(states, epsilon)
        next_states, rewards, terminations, truncations, _ = env.step(actions)

        # gather train_data
        replay_data = tbuf.TorchReplayData(
            states, actions, rewards, terminations, truncations, next_states
        )
        eps = replay_ep_buffer.store_transitions(replay_data)

        # train
        if eps.batch_size() > 0:
            # augment with HER
            train_eps = [eps]
            for _ in range(args.her_future):
                train_eps.append(ther.torch_make_her_future(eps))
            for _ in range(args.her_final):
                train_eps.append(ther.torch_make_her_final(eps))
            train_eps = tbuf.TorchReplayEpData.concatenate(train_eps)

            # compute target values
            train_states, train_actions, train_targets = make_train_data(
                    agent, train_eps, args.lambd, args.gamma
                    )

            # train step
            agent.train(train_states, train_actions, train_targets)

        states = next_states

    return agent
