import numpy as np
from typing import Any
from nn.her_cube_agent import HERCubeAgent
import kostka.kostka_vek as kv
import kostka.kostka as ko
import tqdm

def generate_batch(agent: HERCubeAgent, episodes: int, sample_moves: int, move_limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    def compute_returns(rewards: np.ndarray) -> np.ndarray:
        return np.array([np.sum(rewards[i:]) for i in range(len(rewards))])
    
    def generate_episode(agent: HERCubeAgent, sample_moves: int, move_limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        kostka = ko.nova_kostka()
        goal = ko.nova_kostka()
        ko.zamichej(goal, sample_moves)
        state = np.stack([kostka, goal]) # state: [kostka, goal], shape: [2, kostka.shape] = [2, 6, 3, 3]
        
        states = np.empty([move_limit, *state.shape])# 2, *kostka.shape])
        actions = np.empty([move_limit])
        next_states = np.empty([move_limit, *state.shape])#2, *kostka.shape])
        rewards = np.empty(move_limit)

        assert not ko.je_stejna(kostka, goal), "Tak jsme to snad zamichali, ne?"

        # TODO: vektorova optimalizace
        assert move_limit > 0, "Move limit musí být větší než 0, protože se mi to nechce ošetřovat."
        done = False
        for i in range(move_limit):
            if done:
                break
            states[i] = state

            probs = agent.predict(state[np.newaxis])[0]
            action = np.random.choice(len(probs), size=1, p=probs)
            ko.tahni_tah(state[0], agent.indexy_na_tahy(action)[0])

            actions[i] = action[0]
            next_states[i] = state
            done = ko.je_stejna(state[0], state[1])
            rewards[i] = 0 if done else -1

        return states[:i], actions[:i], next_states[:i], state # type: ignore

    assert episodes > 0, "Počet episod musí být alespoň 1."
    all_states, all_actions, all_returns = [], [], []
    for _ in range(episodes):
        ep_states, ep_actions, ep_rewards, last_state = generate_episode(agent, sample_moves, move_limit)
        ep_returns = compute_returns(ep_rewards)
        all_states.append(ep_states)
        all_actions.append(ep_actions)
        all_returns.append(ep_returns)

        # --- HER ---
        ep_her_rewards = ep_rewards.copy()
        ep_her_rewards[-1] = 0
        ep_her_states = ep_states.copy()
        ep_her_states[:, 1] = last_state[0]
        ep_her_rewards = ep_rewards.copy()
        ep_her_rewards[-1] = 0
        ep_her_returns = compute_returns(ep_her_rewards)
        
        all_states.append(ep_her_states)
        all_actions.append(ep_actions)
        all_returns.append(ep_her_returns)

    all_states = np.stack(all_states).reshape(-1, 2*6*3*3)
    all_actions = np.stack(all_actions).reshape(-1)
    all_returns = np.stack(all_returns).reshape(-1)

    return all_states, all_actions, all_returns
    
def evaluate(agent: HERCubeAgent, batch_size: int, sample_moves: int, limit: int) -> None:
    # NOTE: oproti effc evaluate jsou jine stavy
    raise NotImplementedError

def format_float(x: float) -> str:
    return f"{x:.4f}" if x >= 1e-4 else f"{x:.4e}"

def format_info(info: dict[str, Any]) -> str:
    return " - ".join(map(lambda x: f"{x[0]} {format_float(x[1]) if isinstance(x[1], float) else x[1]}", info.items()))

def train_her_cube(steps: int,
                   train_episodes: int,
                   train_sample_moves: int,
                   train_ep_lim: int,
                   eval_each: int,
                   eval_batch_size: int,
                   eval_sample_moves: int,
                   eval_ep_lim: int) -> HERCubeAgent:
    def new_bar(steps: int) -> tqdm.tqdm:
        return tqdm.tqdm(total=steps, desc="Training", leave=True)
 

    agent = HERCubeAgent()
    bar = new_bar(min(eval_each, steps))

    for step in range(1, steps + 1):
        states, actions, returns = generate_batch(agent, train_episodes, train_sample_moves, train_ep_lim)
        agent.train(states, actions, returns)

        bar.update()
        if (step % eval_each) == 0 or step == steps: 
            evaluate(agent, eval_batch_size, eval_sample_moves, eval_ep_lim) # type: ignore
            
            bar.bar_format = f'{{desc}} {format_info(agent.info)} [{{elapsed}}, {{rate_fmt}}{{postfix}}]'
            bar.set_description(f"Evaluation after {step} steps", False)
            bar.close()
            if step < steps:
                bar = new_bar(min(eval_each, steps - step))

    return agent


if __name__ == "__main__":
    train_her_cube(
        steps=100_000,
        train_episodes=128,
        train_sample_moves=26,
        train_ep_lim=26,
        eval_each=100,
        eval_batch_size=100,
        eval_sample_moves=16,
        eval_ep_lim=16
    )
