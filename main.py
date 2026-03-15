#! /usr/bin/env python3

from hmc.agents.rainbow.train_rainbow import train_rainbow
from hmc.agents.pqn.train_pqn import train_pqn
from hmc.agents.rainbow.train_rainbow import train_rainbow
from hmc.agents.ppo.train_ppo import train_ppo
from hmc.agents.cayleypy.train_cayleypy import train_cayleypy
from hmc.agents.diffusion.train_diffusion import train_diffusion
from hmc.agents.effcube.train_effcube import train_effcube
import hmc.env
import argparse

import random
import numpy as np
import torch

parser = argparse.ArgumentParser()

# Agent
parser.add_argument("--agent", type=str, default="pqn", choices=[
    "pqn", "pqn_plus", "ppo", "rainbow", "cayleypy", "diffusion", "effcube"
], help="")

# Problem
parser.add_argument("--problem", type=str, default="cube", choices=["sliding", "cube", "lightsout"], help="name of the problem")
parser.add_argument("--env_size", type=int, default=3, help="size of the environment (if aplicable)")
parser.add_argument("--num_envs", type=int, default=256, help="number of parallel envs")
parser.add_argument("--scramble_len", type=int, default=5, help="length of puzzle scramble")
parser.add_argument("--ep_limit", type=int, default=10, help="maximum length of an episode")
parser.add_argument("--beam_size", type=int, default=256, help="size of beam for beam search")
parser.add_argument("--max_steps", type=int, default=20_000, help="max number of train steps")
parser.add_argument("--eval_each", type=int, default=1000, help="eval after each n steps")
parser.add_argument("--eval_num_envs", type=int, default=128, help="number of eval envs")
parser.add_argument("--eval_scramble_len", type=int, default=5, help="length of eval puzzle scramble")
parser.add_argument("--eval_ep_limit", type=int, default=10, help="maximum length of an eval episode")
parser.add_argument("--reward_type", type=str, default="punish", choices=["punish", "reward"], help="reward type")

# Network
parser.add_argument("--n1", type=int, default=1024, help="size of hidden layer")
parser.add_argument("--n2", type=int, default=0, help="size of hidden layer")
parser.add_argument("--nr", type=int, default=0, help="size of hidden layer")

# Common hyperparameters
parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
parser.add_argument("--l2", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
parser.add_argument("--min_train_size", type=int, default=256, help="batch_size")
parser.add_argument("--epochs", type=int, default=1, help="batch_size")
parser.add_argument("--clip_grad_norm", type=float, default=10.0, help="clip grad l2 norm")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--lambd", type=float, default=0.6, help="lambda-return coef")

# PQN
parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon-greedy exploration")
parser.add_argument("--epsilon_final", type=float, default=0.01, help="final epsilon")
parser.add_argument("--epsilon_final_at", type=int, default=20_000, help="steps to reach final eps.")

# PQN+
parser.add_argument("--atoms", type=int, default=51, help="number of atoms for dist. dqn")
parser.add_argument("--v_min", type=float, default=-26.0, help="min atom value")
parser.add_argument("--v_max", type=float, default=-1.0, help="max atom value")

# PPO
parser.add_argument("--clip_epsilon", type=float, default=0.2, help="ppo epsilon")
parser.add_argument("--entropy_reg", type=float, default=0.001, help="ppo entropy regularization")
parser.add_argument("--last_layer_init_scale", type=float, default=0.01, help="ppo entropy regularization")

# HER
parser.add_argument("--her_future", type=int, default=0, help="number of future HER episodes")
parser.add_argument("--her_final", type=int, default=0, help="number of final HER episodes")

# Reproducibility
parser.add_argument("--deterministic", action="store_true", help="sets everything to be deterministic")
parser.add_argument("--seed", type=int, default=0, help="random seed")


def train_agent(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    if args.agent in ("pqn", "pqn_plus"):
        train_pqn(args)
    elif args.agent == "ppo":
        train_ppo(args)
    elif args.agent == "rainbow":
        train_rainbow(args)
    elif args.agent == "cayleypy":
        train_cayleypy(args)
    elif args.agent == "diffusion":
        train_diffusion(args)
    elif args.agent == "effcube":
        train_effcube(args)

if __name__ == "__main__":
    args = parser.parse_args()
    train_agent(args)
