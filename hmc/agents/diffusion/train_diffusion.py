from hmc.agents.diffusion.diffusion import Diffusion
from hmc.agents.diffusion.noop_problem_wrapper import NoOpProblemWrapper
from hmc.problems.sliding import Sliding
from hmc.problems.cube import RubiksCube
from hmc.problems.lightsout import LightsOut
import hmc.utils.torch_utils as tut
import argparse
import torch


import hmc.kostka.torch_cube_vec as tcv
import hmc.kostka.torch_cube as tcu

def sample_noops(
    problem: NoOpProblemWrapper, features: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    T = features.shape[0]
    N = features.shape[1]

    # sampled_lengths = torch.randint(
        # 1, T, [N], generator=problem._problem._rng, device=problem.device
    # )

    # on average there are N/A 'bad' pairs of moves (x, -x) in a single scramble
    # so binomial distribution will add N/A no-ops on average
    # T - 1 because we want to have at least one transition (~ we are not learning that solved states are solved)
    dist = torch.distributions.Binomial(T - 1, torch.tensor(1/problem.num_actions, device=problem.device))
    # this distribution will sample integers from [0, T - 1]

    sampled_lengths = T - dist.sample([N]).type(torch.long)
    _rang_n = torch.arange(N, device=problem.device)
    final_features = features[sampled_lengths-1, _rang_n]

    _rang_t = torch.arange(T, 0, -1, device=problem.device)
    noop_mask = _rang_t[:, None] > sampled_lengths[None]
    features = torch.where(noop_mask[..., None], final_features[None], features)
    targets[noop_mask] = 0

    return features, targets


def generate_data(
    problem: NoOpProblemWrapper, num_scrambles: int, max_scramble_len: int
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

    # return features, targets
    noop_features, noop_targets = features, targets
    noop_features, noop_targets = sample_noops(problem, features, targets)
    return noop_features, noop_targets


def evaluate(
    agent: Diffusion,
    problem: NoOpProblemWrapper,
    num_evals: int,
    scramble_len: int,
) -> tuple[float, float]:
    states = problem.new_state_vec(num_evals)
    _ = problem.scramble_vec(states, scramble_len)

    solved_states = problem.new_state_vec(1)
    predicted_states = agent.predict(states, True)
    solved = problem.is_solved_vec(predicted_states, solved_states)
    return torch.count_nonzero(solved).item() / num_evals, -1


def train_diffusion(args: argparse.Namespace) -> Diffusion:

    if args.problem == "sliding":
        assert False, "sliding has broken inverse actions due to boundaries"
        problem = Sliding(args.env_size, tut.get_torch_cube_device(), args.seed)
    elif args.problem == "cube":
        problem = RubiksCube(args.env_size, tut.get_torch_cube_device(), args.seed)
    elif args.problem == "lightsout":
        problem = LightsOut(args.env_size, tut.get_torch_cube_device(), args.seed)
    else:
        raise ValueError
    problem = NoOpProblemWrapper(problem)

    agent = Diffusion(
        args,
        problem,
        problem.state_bound,
        problem.state_size,
        problem.num_actions,
        args.scramble_len,
        tut.get_torch_cube_device(),
    )

    import hmc.kostka.torch_cube_vec as tcv

    for step in range(1, args.max_steps + 1):
        states, targets = generate_data(problem, args.num_envs, args.scramble_len)
        # pj = 0
        # pi = [0, -1]
        # tcv.print_cube_vec(states[pi, pj].reshape(-1, 6, 3, 3))
        # print(targets[pi, pj])
        # print()
        # input()

        agent.train(states, targets)
        if step % args.eval_each == 0:
            solved, length = evaluate(
                agent,
                problem,
                args.eval_num_envs,
                args.eval_scramble_len,
            )
            print(
                f"Evaluation after {step} steps: solved {solved:.2f}, length {length:.2f}."
            )
    return agent
