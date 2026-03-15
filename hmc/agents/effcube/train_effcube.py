from hmc.agents.effcube.effcube import EffCube
from hmc.problems.sliding import Sliding
from hmc.problems.cube import RubiksCube
from hmc.problems.lightsout import LightsOut
from hmc.problems.baseproblem import Problem
import hmc.utils.solver as sol
import hmc.utils.torch_utils as tut
import argparse
import torch



def generate_data(
    problem: Problem, num_scrambles: int, max_scramble_len: int
) -> tuple[torch.Tensor, torch.Tensor]:

    features = torch.empty(
        [max_scramble_len, num_scrambles, problem.num_features],
        dtype=torch.long,
        device=problem.device,
    )

    states = problem.new_state_vec(num_scrambles)
    scrambles = problem.sample_actions([max_scramble_len, num_scrambles])
    for step in range(max_scramble_len):
        problem.perform_action_vec(states, scrambles[step])
        features[step] = problem.make_features_vec(states)

    targets = problem.invert_actions(scrambles)

    # the time ordering in the 1st dim is 1, 2, 3, ..., T-1, T
    targets = targets.flip(0)
    features = features.flip(0)
    # the time ordering in the 1st dim is T, T-1, T-2, ..., 1

    return features, targets


def evaluate_beam(
    agent: EffCube,
    problem: Problem,
    scramble_len: int,
    time_limit: int,
    beam_width: int,
    num_evals: int,
) -> tuple[float, float]:

    def heuristic(stategoals: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        assert len(h.shape) == 1
        features = problem.make_features_vec(stategoals)
        F = features.shape[-1] // 2
        probs = agent.predict_action_probs(features[:, :F])
        return h[:, None] * probs

    solved = 0
    solution_len_sum = 0
    for run in range(num_evals):
        goal = problem.new_state()
        problem.scramble(goal, scramble_len)

        solution_len = sol.solve_beam_uniuniversal_stateaction(
            problem, problem.new_state(), heuristic, time_limit, beam_width, 1, True, init_state=goal
        )

        if solution_len >= 0:
            solved += 1
            solution_len_sum += solution_len
    return solved / num_evals, solution_len_sum / (solved + 1e-8)


def train_effcube(args: argparse.Namespace) -> EffCube:

    if args.problem == "sliding":
        assert False, "sliding has broken inverse actions due to boundaries"
        problem = Sliding(args.env_size, tut.get_torch_cube_device(), args.seed)
    elif args.problem == "cube":
        problem = RubiksCube(args.env_size, tut.get_torch_cube_device(), args.seed)
    elif args.problem == "lightsout":
        problem = LightsOut(args.env_size, tut.get_torch_cube_device(), args.seed)
    else:
        raise ValueError

    agent = EffCube(
        args,
        problem,
        problem.state_bound,
        problem.state_size,
        problem.num_actions,
        tut.get_torch_cube_device(),
    )

    for step in range(1, args.max_steps + 1):
        states, targets = generate_data(problem, args.num_envs, args.scramble_len)

        agent.train(states, targets)
        if step % args.eval_each == 0:
            solved, length = evaluate_beam(
                agent,
                problem,
                args.eval_scramble_len,
                args.eval_ep_limit,
                args.beam_size,
                args.eval_num_envs
            )
            print(
                f"Evaluation after {step} steps: solved {solved:.2f}, length {length:.2f}."
            )
    return agent
