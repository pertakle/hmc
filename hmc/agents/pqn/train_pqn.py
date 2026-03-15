import argparse
import gymnasium as gym
from .pqn import PQN
from .pqn_plus import PQNPlus
import torch
import numpy as np
import hmc.utils.torch_utils as tut
import hmc.agents.rl_utils.torch_buffers as tbuf
import hmc.agents.rl_utils.torch_her as ther
from hmc.utils.wrappers import PositiveWrapperVec
import hmc.utils.solver as sol
from hmc.problems.baseproblem import Problem
from hmc.problems.sliding import Sliding
from hmc.problems.cube import RubiksCube
from hmc.problems.lightsout import LightsOut

# TODO: move `evaluate` and `evaluate_beam` to hmc/utils/solver.py to have only a single implementation
def evaluate_beam(
    agent: PQN | PQNPlus,
    problem: Problem,
    scramble_len: int,
    time_limit: int,
    beam_width: int,
    num_evals: int,
) -> tuple[float, float]:
    def heuristic(stategoals: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return agent.predict_q(stategoals).max(1).values

    def heuristic_sa(stategoals: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return agent.predict_q(stategoals)

    solved = 0
    solution_len_sum = 0
    for _ in range(num_evals):
        goal = problem.new_state()
        problem.scramble(goal, scramble_len)

        # solution_len = sol.solve_beam_uniuniversal(problem, goal, heuristic, time_limit, beam_width, 0, True)
        solution_len = sol.solve_beam_uniuniversal_stateaction(
            problem, goal, heuristic_sa, time_limit, beam_width, 0, True
        )

        if solution_len >= 0:
            solved += 1
            solution_len_sum += solution_len
    return solved / num_evals, solution_len_sum / (solved + 1e-8)


def train_pqn(args: argparse.Namespace) -> PQN | PQNPlus:

    if args.problem == "sliding":
        problem = Sliding(args.env_size, tut.get_torch_cube_device(), args.seed)
    elif args.problem == "cube":
        problem = RubiksCube(args.env_size, tut.get_torch_cube_device(), args.seed)
    elif args.problem == "lightsout":
        problem = LightsOut(args.env_size, tut.get_torch_cube_device(), args.seed)
    else:
        raise ValueError

    env = gym.make_vec(
        "hmc/ProblemEnv-v0",
        args.num_envs,
        problem=problem,
        scramble_len=args.scramble_len,
        ep_limit=args.ep_limit,
    )

    assert type(env.single_observation_space) == gym.spaces.MultiDiscrete
    assert type(env.single_action_space) == gym.spaces.Discrete

    ob_space = env.single_observation_space
    act_space = env.single_action_space

    if args.agent == "pqn":
        agent = PQN(
            args,
            ob_space.nvec[0],
            ob_space.shape[0],
            act_space.n.item(),
            False,
            tut.get_torch_cube_device(),
        )
    elif args.agent == "pqn_plus":
        agent = PQNPlus(
            args,
            ob_space.nvec[0],
            ob_space.shape[0],
            act_space.n.item(),
            tut.get_torch_cube_device(),
        )
    else:
        raise ValueError(f"Invalid agent name '{args.agent}'!")

    states = env.reset()[0]
    ep_buffer = tbuf.TorchEpisodeBuffer(
        args.num_envs,
        states[0].shape,
        args.ep_limit,
        problem._device,
        states.dtype,
    )
    replay_buffer = tbuf.TorchReplayEpBuffer()

    if args.reward_type == "reward":
        env = PositiveWrapperVec(env)

    training = True
    step = 0
    while training:
        step += 1

        # make steps in the environments
        epsilon = np.interp(
            step, [1, args.epsilon_final_at], [args.epsilon, args.epsilon_final]
        )
        actions = agent.predict_actions(states, epsilon)
        next_states, rewards, terminations, truncations, _ = env.step(actions)

        # gather train_data
        transitions = tbuf.TorchReplayData(
            states, actions, rewards, terminations, truncations, next_states
        )
        eps = ep_buffer.store_transitions(transitions)

        # train
        if eps.batch_size() > 0: # if any episode finished
            # augment with HER
            replay_buffer.add(eps)
            for _ in range(args.her_future):
                replay_buffer.add(ther.torch_make_her_future(eps, args.reward_type))
            for _ in range(args.her_final):
                replay_buffer.add(ther.torch_make_her_final(eps, args.reward_type))

            # train if enough data has been collected
            if replay_buffer.current_data_size() >= args.min_train_size:
                train_eps = replay_buffer.pop_data()
                agent.train(train_eps)

        if step % args.eval_each == 0:
            print(f"Evaluation after {step} steps: ", end="", flush=True)
            solved, length = evaluate_beam(
                agent,
                problem,
                args.eval_scramble_len,
                args.eval_ep_limit,
                args.beam_size,
                args.eval_num_envs
            )
            print(f"solved {solved:.2f}, length {length:.2f}.")
            # mean, std = evaluate(agent, eval_env)
            # print(f"Evaluation after {step} steps: {mean:.2f} +-{std:.2f}")

        if step >= args.max_steps:
            training = False

        states = next_states

    return agent
