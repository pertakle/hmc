from deepercube.agents.rainbow.train_rainbow import train_rainbow
from deepercube.agents.pqn.train_pqn import train_pqn
import deepercube.env
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--beam_size", type=int, default=256, help="size of beam for beam search")
parser.add_argument("--num_envs", type=int, default=256, help="number of parallel envs")
parser.add_argument("--scramble_len", type=int, default=100, help="length of puzzle scramble")
parser.add_argument("--ep_limit", type=int, default=26, help="maximum length of an episode")
parser.add_argument("--max_steps", type=int, default=100_000, help="max number of train steps")

parser.add_argument("--eval_each", type=int, default=100, help="eval after each n steps")
parser.add_argument("--eval_num_envs", type=int, default=128, help="number of eval envs")
parser.add_argument("--eval_scramble_len", type=int, default=100, help="length of eval puzzle scramble")
parser.add_argument("--eval_ep_limit", type=int, default=64, help="maximum length of an eval episode")


parser.add_argument("--hidden_size", type=int, default=1024, help="size of hidden layer")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
parser.add_argument("--clip_grad_norm", type=float, default=0.0, help="clip grad l2 norm")
parser.add_argument("--replay_buffer_size", type=int, default=120_000, help="size of replay buffer")
parser.add_argument("--replay_start_size", type=int, default=10_000, help="min RB to train from")
parser.add_argument("--target_update_each", type=int, default=1, help="frequency of target update")

parser.add_argument("--alpha", type=float, default=0.6, help="alpha for PER")
parser.add_argument("--beta", type=float, default=0.4, help="beta for PER")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--tau", type=float, default=0.001, help="smooth target update coef")
parser.add_argument("--lambd", type=float, default=0.4, help="Q(lambda) coef")
parser.add_argument("--epsilon", type=float, default=0.5, help="epsilon-greedy exploration")
parser.add_argument("--epsilon_final", type=float, default=0.1, help="final epsilon")
parser.add_argument("--epsilon_final_at", type=int, default=1_000_000, help="steps to reach final eps.")

parser.add_argument("--n_step", type=int, default=1, help="n-step return")
parser.add_argument("--atoms", type=int, default=26, help="number of atoms for dist. dqn")
parser.add_argument("--v_min", type=float, default=-26.0, help="min atom value")
parser.add_argument("--v_max", type=float, default=-1.0, help="max atom value")

parser.add_argument("--her_future", type=int, default=0, help="number of future HER episodes")
parser.add_argument("--her_final", type=int, default=1, help="number of final HER episodes")


if __name__ == "__main__":
    from deepercube.env.lights_out import LightsOutVec
    import torch
    env = LightsOutVec(3, 2, 1, 10000)
    env.reset()
    done = False
    while True:
        env.print()
        actions = torch.as_tensor(list(map(int, input("Action: ").split(" "))), dtype=torch.long)
        _, _, term, trun, _  = env.step(actions)
        done = torch.any(term | trun)
    env.print()
    exit()



    args = parser.parse_args()
    train_pqn(args)

