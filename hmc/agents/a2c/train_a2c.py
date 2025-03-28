from a2c import A2CAgent
import gymnasium as gym
import numpy as np


def evaluate(env: gym.vector.VectorEnv, agent: A2CAgent) -> float:
    states = env.reset()[0]
    dones = np.full(env.num_envs, False)
    rewards = 0
    while not np.all(dones):
        actions = np.argmax(agent.predict_action_probs(states))  # type: ignore
        states, rewards, terminated, truncated, _ = env.step(actions)
        rewards += rewards @ (~dones)
        dones |= terminated | truncated
    return rewards / env.num_envs


def train_a2c(
    env: gym.vector.VectorEnv,
    eval_env: gym.vector.VectorEnv,
    gamma: float,
    eval_each: int,
) -> A2CAgent:
    """
    def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # Set random seeds and the number of threads
        np.random.seed(args.seed)
        if args.seed is not None:
            torch.manual_seed(args.seed)
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

        # Construct the network
        network = Network(env, args)

        def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
            rewards, state, done = 0, env.reset(start_evaluation=start_evaluation, logging=logging)[0], False
            while not done:
                # TODO: Predict the action using the greedy policy.
                action_probabs = network.predict_actions(state[np.newaxis])[0]
                action = np.argmax(action_probabs)

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                rewards += reward
            return rewards

        # Create the vectorized environment
        vector_env = gym.make_vec(args.env, args.envs, gym.VectorizeMode.ASYNC)
        states = vector_env.reset(seed=args.seed)[0]

        training, autoreset = True, np.zeros(args.envs, dtype=bool)
        while training:
            for _ in range(args.evaluate_each):
                action_probabs = network.predict_actions(states)
                # select actions based on probabilities
                actions = np.array([np.random.choice(len(probabs), p=probabs) for probabs in action_probabs])

                # Perform steps in the vectorized environment
                next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
                dones = terminated | truncated

                estimates = network.predict_values(next_states).squeeze()
                not_reset = np.logical_not(autoreset)

                network.train(
                    (states),
                    (actions),
                    ((rewards / 500) + args.gamma * estimates.squeeze()) * not_reset,
                )

                states = next_states
                autoreset = dones
    """
    # TODO: polish this
    obs_space: gym.spaces.MultiDiscrete = env.single_action_space  # type: ignore
    action_space: gym.spaces.Discrete = env.single_action_space  # type: ignore

    assert isinstance(obs_space, gym.spaces.MultiDiscrete)
    assert isinstance(action_space, gym.spaces.Discrete)
    agent = A2CAgent(obs_space.shape[0], obs_space.nvec[0], action_space.n.item())

    training = True
    auto_reset = np.zeros(env.num_envs, dtype=bool)
    states = env.reset()[0]
    while training:
        for _ in range(eval_each):
            actions_probabs = agent.predict_action_probs()
            actions = np.array(
                [
                    np.random.choice(len(probabs), p=probabs)
                    for probabs in actions_probabs
                ]
            )
            next_states, rewards, terminated, truncated, _ = env.step(actions)
            dones = terminated | truncated

            estimates = agent.predict_values(next_states).squeeze()  # type: ignore
            not_reset = ~auto_reset
            agent.train(
                states, actions, (rewards + gamma * estimates) * not_reset
            )  # TODO: osetrit konce

            states = next_states
            auto_reset = dones
        # TODO: eval

    return agent
