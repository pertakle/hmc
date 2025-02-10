from copy import deepcopy
import numpy as np
import gymnasium as gym
import deepercube.kostka.kostka as ko
from .rainbow import Rainbow
from . import her, buffers
import argparse
import deepercube.utils.solver as sol


class Queue:
    def __init__(self, size: int) -> None:
        self._queue = np.empty(size)
        self._size = size
        self._next = 0
        self._filled = False

    def __repr__(self) -> str:
        if self._filled:
            queue = np.roll(self._queue, -self._next)
        else:
            queue = self._queue[:self._next]
        return str(queue)
        

    def push(self, x: float) -> None:
        self._queue[self._next] = x
        self._next += 1
        if self._next >= self._size:
            self._next -= self._size
            self._filled = True

    def mean(self) -> float:
        size = self._size if self._filled else self._next
        return np.mean(self._queue[:size]).item()


def distr_bellman(
    args: argparse.Namespace,
    network: Rainbow,
    target_network: Rainbow,
    rewards: np.ndarray,
    next_states: np.ndarray,
    not_terminated: np.ndarray,
) -> np.ndarray:
    PT_SS: np.ndarray = target_network.predict_probs(next_states, True)  # type: ignore
    QB_SS: np.ndarray = network.predict_q_values(next_states, True)  # type: ignore
    a_star = QB_SS.argmax(1)

    # nice trick from https://github.com/ShangtongZhang/DeepRL
    atoms_target = (
        rewards[..., None]
        + args.gamma**args.n_step * not_terminated[:, None] * network.atoms_np[None]
    )
    atoms_target = np.clip(atoms_target, args.v_min, args.v_max)[:, None, :]
    # atoms_target: [B, 1, dist]
    target_prob = (
        np.clip(
            1
            - np.abs(atoms_target - network.atoms_np[None, :, None])
            / network.atom_delta,
            0,
            1,
        )
        * PT_SS[np.arange(len(PT_SS)), a_star][:, None]
    ).sum(-1)
    return target_prob

import deepercube.kostka.kostka_vek as kv
def print_ep(episodes: buffers.ReplayEpData, ep_index: int) -> None:
    def print_state_goal(sg: np.ndarray) -> None:
        sg_space = 5
        sg = sg.reshape(2, 6, 3, 3).astype(int)
        for ls, lg in zip(sg[0, 0], sg[1, 0]):
            print(" " * 7, ls, " " * (21 + sg_space), lg, sep="")
        for i in range(3):
            for s in sg[0, 1:5]:
                print(s[i], end="")
            print(" " * sg_space, end="")
            for s in sg[1, 1:5]:
                print(s[i], end="")
            print()
        for ls, lg in zip(sg[0, 5], sg[1, 5]):
            print(" " * 7, ls, " " * (21 + sg_space), lg, sep="")

    def print_transition(ep: buffers.ReplayEpData, ep_i: int, tr: int) -> None:
        print("State-goal")
        print_state_goal(ep.states[ep_i, tr])
        print("Action", ep.actions[ep_i, tr])
        print("Reward", ep.rewards[ep_i, tr])
        print("Next state-goal")
        print_state_goal(ep.next_states[ep_i, tr])

    for t in range(episodes.lengths[ep_index]):
        print_transition(episodes, ep_index, t)
        input()
    input("Konec episody")


def train_rainbow(args: argparse.Namespace) -> Rainbow:

    env = gym.make_vec(
        "deepercube/RubiksCube-v0",
        num_envs=args.num_envs,
        scramble_len=args.scramble_len,
        ep_limit=args.ep_limit,
    )
    eval_env = gym.make_vec(
        "deepercube/RubiksCube-v0",
        num_envs=args.eval_num_envs,
        scramble_len=args.eval_scramble_len,
        ep_limit=args.eval_ep_limit,
    )

    def evaluate_beam(agent: Rainbow, args: argparse.Namespace) -> float:
        def heuristic(states: np.ndarray, prev_values: np.ndarray) -> np.ndarray:
            q_values = agent.predict_q_values(states, True)
            return q_values

        made_moves = 0
        for _ in range(args.eval_num_envs):
            cube = ko.nova_kostka()
            ko.zamichej(cube, args.eval_scramble_len)
            made_moves += sol.solve_beam_universal(cube, heuristic, args.beam_size, args.eval_ep_limit)
        return -made_moves / args.eval_num_envs

    def evaluate_episode() -> float:
        rewards_total = 0
        dones = np.full(eval_env.num_envs, False)
        states, _ = eval_env.reset()
        while not np.all(dones):
            actions = agent.predict_q_values(states, True).argmax(-1)
            next_states, rewards, terminated, truncated, _ = eval_env.step(actions)
            rewards_total += rewards @ (~dones)
            dones |= terminated | truncated
            states = next_states
        rewards_total /= eval_env.num_envs
        return rewards_total

    assert type(env.single_observation_space) == gym.spaces.MultiDiscrete
    assert type(env.single_action_space) == gym.spaces.Discrete

    ob_space = env.single_observation_space
    action_space = env.single_action_space

    agent = Rainbow(args, ob_space.nvec[0], ob_space.shape[0], action_space.n.item())
    target_agent = deepcopy(agent)

    replay_buffer = buffers.ReplayBuffer(args.replay_buffer_size)
    ep_buffer = buffers.EpisodeBuffer(args.num_envs, ob_space.shape, args.ep_limit)
    n_step_buffer = buffers.NStepBufferVec(args.n_step, env.num_envs, args.gamma)
    mean_queue = Queue(30)

    states, _ = env.reset()
    training = True
    replay_filled = False
    step = 0
    while training:
        step += 1

        # Make a step in the environments
        actions = agent.predict_q_values(states, False).argmax(-1)
        next_states, rewards, terminated, truncated, _ = env.step(actions)

        # Store new transitions
        transition = buffers.ReplayData(
            states,
            actions,
            rewards,
            terminated,
            truncated,
            next_states,
        )
        #nstep_transitions = n_step_buffer.step(transition)
        nstep_transitions = transition
        if nstep_transitions is not None:
            eps = ep_buffer.store_transitions(nstep_transitions)
            if eps is not None:
                train_eps = [eps]
                for _ in range(args.her_future):
                    train_eps.append(her.make_her_future(eps))
                for _ in range(args.her_final):
                    train_eps.append(her.make_her_final(eps))
                train_eps = buffers.ReplayEpData.concatenate(train_eps)
                #print("normal")
                #print_ep(train_eps, 0)
                #print("final")
                #print_ep(train_eps, -1)
                replay_buffer.store_replay(train_eps.unroll())

        # Train on sampled batch
        if len(replay_buffer) >= args.replay_start_size:
            if not replay_filled:
                replay_filled = True
                print("Replay buffer filled.")

            alpha = args.alpha  #np.interp(step, [0, args.steps], [args.alpha, 0]).item()
            beta = np.interp(step, [0, args.max_steps], [args.beta, 1]).item()
            data, indices, isw = replay_buffer.sample_transitions(
                args.batch_size, alpha, beta
            )
            target = distr_bellman(
                args,
                agent,
                target_agent,
                data.rewards,
                data.next_states,
                ~data.terminated,
            )
            priorities: np.ndarray = agent.train(data.states, data.actions, target, isw)  # type: ignore
            replay_buffer.update_priorities(indices, priorities)

        states = next_states

        if (step % args.target_update_each) == 0:
            target_agent.copy_weights_from(agent, args.tau)

        if (step % args.eval_each) == 0:
            #eval_rewards = evaluate_episode()
            eval_rewards = evaluate_beam(agent, args)
            mean_queue.push(eval_rewards)
            eval_mean_rewards = mean_queue.mean()
            print(
                f"Evaluation after {step} steps: {eval_rewards:.4f}, mean {mean_queue._size}: {eval_mean_rewards:.4f}"
            )

        if step >= args.max_steps:
            training = False

    return agent
