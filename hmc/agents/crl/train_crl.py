import torch
import gymnasium as gym
from hmc.problems.sliding import Sliding
from hmc.problems.cube import RubiksCube
from hmc.problems.lightsout import LightsOut
from hmc.problems.baseproblem import Problem
import argparse
from hmc.agents.crl.crl import CRL
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


class DataGen:
    def __init__(self, problem: Problem, args: argparse.Namespace) -> None:
        self.problem = problem
        self.args = args
        self.states = problem.new_state_vec(args.num_envs)
        self.goals = problem.new_state_vec(args.num_envs)
        self._reset()

    def _reset(self, mask: torch.Tensor|None = None) -> None:
        if mask is None:
            mask = torch.ones(self.args.num_envs, dtype=torch.bool)

        num_resets = torch.count_nonzero(mask)
        if num_resets == 0:
            return

        new_states = self.problem.new_state_vec(self.args.num_envs)
        new_goals = self.problem.new_state_vec(self.args.num_envs)
        self.problem.scramble_vec(new_goals, self.args.scramble_len)

        self.states[mask] = new_states
        self.goals[mask] = new_goals

    def generate_data(self, agent: CRL, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_states = []
        curr_batch_size = 0
        while curr_batch_size < batch_size:
            # get new transitions

    def _get_random_goals(self, n: int) -> torch.Tensor:
        goals = self.problem.new_state_vec(n)
        self.problem.scramble_vec(goals, self.args.scramble_len)
        return goals

def train_crl(args: argparse.Namespace) -> CRL:

    if args.problem == "sliding":
        problem = Sliding(args.env_size, tut.get_torch_cube_device(), args.seed)
    elif args.problem == "cube":
        problem = RubiksCube(args.env_size, tut.get_torch_cube_device(), args.seed)
    elif args.problem == "lightsout":
        problem = LightsOut(args.env_size, tut.get_torch_cube_device(), args.seed)
    else:
        raise ValueError

    agent = CRL(
        args,
        problem.state_bound,
        problem.state_size,
        32,
        tut.get_torch_cube_device(),
    )
    env = gym.make_vec(
        "hmc/ProblemEnv-v0",
        args.num_envs,
        problem=problem,
        scramble_len=args.scramble_len,
        ep_limit=args.ep_limit,
    )
    last_states = env.reset()[0]

    K = 26  # idk
    for step in range(1, args.max_steps + 1):
        batch_states, batch_goals, last_states = generate_data(agent, env, args.batch_size, last_states)
        random_goals = problem.new_state_vec(len(states) * K)
        problem.scramble_vec(random_goals, args.scramble_len)
        random_goals = random_goals.reshape(len(states), K, -1)

        agent.train(states, goals, random_goals)
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
