import numpy as np
import tqdm
from typing import Any, Tuple, List
from hmc.nn.a2c import A2CCubeAgent
import hmc.kostka.kostka_vek as kv
import hmc.kostka.kostka as ko
from hmc.utils import data_gen
from hmc.utils import solver


def evaluate(
    agent: A2CCubeAgent, batch_size: int, sample_moves: int, limit: int, beam_size: int
) -> None:
    goals = kv.nova_kostka_vek(batch_size)
    kv.zamichej_nahodnymi_tahy_vek(goals, sample_moves)
    num_solved = solver.solve_beam_vek(goals, agent, beam_size, limit)
    # num_solved = solve_greedy_vek(goals, agent, limit)
    agent.info["solved"] = f"{100*num_solved/batch_size:.2f} %"


def format_float(x: float) -> str:
    return f"{x:.4f}" if x >= 1e-4 else f"{x:.4e}"


def format_info(info: dict[str, Any]) -> str:
    return " - ".join(
        map(
            lambda x: f"{x[0]} {format_float(x[1]) if isinstance(x[1], float) else x[1]}",
            info.items(),
        )
    )


def train_her_cube(
    steps: int,
    train_episodes: int,
    train_sample_moves: int,
    train_ep_lim: int,
    eval_each: int,
    eval_batch_size: int,
    eval_sample_moves: int,
    eval_ep_lim: int,
    eval_beam_size: int,
) -> A2CCubeAgent:

    def new_bar(steps: int) -> tqdm.tqdm:
        return tqdm.tqdm(total=steps, desc="Training", leave=True)

    agent = A2CCubeAgent()
    bar = new_bar(min(eval_each, steps))

    for step in range(1, steps + 1):
        states, actions, returns = data_gen.generate_batch(
            agent, train_episodes, train_sample_moves, train_ep_lim
        )
        agent.train(states, actions, returns)

        bar.update()
        if (step % eval_each) == 0 or step == steps:
            evaluate(agent, eval_batch_size, eval_sample_moves, eval_ep_lim, eval_beam_size)  # type: ignore

            bar.bar_format = f"{{desc}} {format_info(agent.info)} [{{elapsed}}, {{rate_fmt}}{{postfix}}]"
            bar.set_description(f"Evaluation after {step} steps", False)
            bar.close()
            if step < steps:
                bar = new_bar(min(eval_each, steps - step))

    return agent


if __name__ == "__main__":
    train_her_cube(
        steps=100_000,
        train_episodes=64,
        train_sample_moves=7,
        train_ep_lim=7,
        eval_each=100,
        eval_batch_size=100,
        eval_sample_moves=7,
        eval_ep_lim=7,
        eval_beam_size=1,
    )
