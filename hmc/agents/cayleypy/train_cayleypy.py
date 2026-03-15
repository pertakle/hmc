import torch
from hmc.problems.sliding import Sliding
from hmc.problems.cube import RubiksCube
from hmc.problems.lightsout import LightsOut
from hmc.problems.baseproblem import Problem
import argparse
from hmc.agents.cayleypy.cayleypy import CayleyPy
import hmc.utils.torch_utils as tut
import hmc.utils.solver as sol


def evaluate(
    agent: CayleyPy,
    problem: Problem,
    num_evals: int,
    scramble_len: int,
    time_limit: int,
) -> tuple[float, float]:
    states = problem.new_state_vec(num_evals)
    problem.scramble_vec(states, scramble_len)
    goals = problem.new_state_vec(num_evals)
    _rang = torch.arange(num_evals, device=problem.device)
    solved = problem.is_solved_vec(states, goals)
    for step in range(time_limit):
        states = problem.perform_all_actions_vec(states)
        features = problem.make_features_vec(states.flatten(0, 1))
        dist_a = agent.predict(features).reshape(num_evals, problem.num_actions)
        actions = torch.min(dist_a, dim=-1).indices  # min distance to the solved

        states = states[_rang, actions]
        solved = solved | problem.is_solved_vec(states, goals)
    return torch.count_nonzero(solved).item() / num_evals, 0.



def evaluate_beam(
    agent: CayleyPy,
    problem: Problem,
    scramble_len: int,
    time_limit: int,
    beam_width: int,
    num_evals: int,
) -> tuple[float, float]:

    def heuristic(stategoals: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return -agent.predict(stategoals[:, :stategoals.shape[1] // 2])

    solved = 0
    solution_len_sum = 0
    for run in range(num_evals):
        goal = problem.new_state()
        problem.scramble(goal, scramble_len)

        solution_len = sol.solve_beam_uniuniversal(
            problem, problem.new_state(), heuristic, time_limit, beam_width, 0, True, init_state=goal
        )

        if solution_len >= 0:
            solved += 1
            solution_len_sum += solution_len
    return solved / num_evals, solution_len_sum / (solved + 1e-8)


def generate_data(
    problem: Problem, num_paths: int, path_length: int
) -> tuple[torch.Tensor, torch.Tensor]:

    all_targets = torch.arange(0, path_length, device=problem.device)[:, None].repeat(1, num_paths)
    all_states = torch.empty(
        [path_length, num_paths, problem.state_size],
        dtype=torch.long,
        device=problem.device,
    )

    states = problem.new_state_vec(num_paths)
    for i in range(path_length):
        all_states[i] = problem.make_features_vec(states)
        _ = problem.scramble_vec(states, 1)

    return all_states.flatten(0, 1), all_targets.flatten().type(torch.float32)



def train_cayleypy(args: argparse.Namespace) -> CayleyPy:

    if args.problem == "sliding":
        problem = Sliding(args.env_size, tut.get_torch_cube_device(), args.seed)
    elif args.problem == "cube":
        problem = RubiksCube(args.env_size, tut.get_torch_cube_device(), args.seed)
    elif args.problem == "lightsout":
        problem = LightsOut(args.env_size, tut.get_torch_cube_device(), args.seed)
    else:
        raise ValueError

    # print(
        # problem.state_bound,
        # problem.state_size,
        # problem.num_actions,
        # tut.get_torch_cube_device(),
    # )
    # exit()
    # 9 9 4 cuda

    agent = CayleyPy(
        args,
        problem.state_bound,
        problem.state_size,
        problem.num_actions,
        tut.get_torch_cube_device(),
    )

    for step in range(1, args.max_steps + 1):
        states, targets = generate_data(problem, args.num_envs, args.ep_limit)
        agent.train(states, targets)
        if step % args.eval_each == 0:
            solved, length = evaluate_beam(
            # solved, length = evaluate(
                agent,
                problem,
                args.eval_scramble_len,
                args.eval_ep_limit,
                args.beam_size,
                args.eval_num_envs,
            )
            print(
                f"Evaluation after {step} steps: solved {solved:.2f}, length {length:.2f}."
            )
    return agent
