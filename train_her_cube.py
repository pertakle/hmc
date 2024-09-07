import numpy as np
from typing import Any
from nn import her_cube_agent
from nn.her_cube_agent import HERCubeAgent
import kostka.kostka_vek as kv
import kostka.kostka as ko
import tqdm

def generate_batch(agent: HERCubeAgent, episodes: int, sample_moves: int, move_limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    def compute_returns(rewards: np.ndarray) -> np.ndarray:
        return np.cumsum(rewards[::-1])[::-1] # np.flip(a)
        #return np.array([np.sum(rewards[i:]) for i in range(len(rewards))])

    def generate_episodes_vec(
        agent: HERCubeAgent,
        sample_moves: int,
        move_limit: int,
        number_of_episodes: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates `number_of_episodes` vectorized episodes with `sample_moves` lengths.
        Goals are randomly scrambled cubes with `sample_moves` uniformly random moves.

        Returns:
            - States: ndarray, dtype=float, shape=[`move_limit`, `number_of_episodes`, 2 * cube features]
            - Actions: ndarray, dtype=float, shape=[`move_limit`, `number_of_episodes`]
            - Episode lengths: ndarray, dtype=int, shape=[`number_of_episodes`]
        """
        goals = kv.nova_kostka_vek(number_of_episodes)
        zamichani = kv.vygeneruj_nahodny_tah_vek([sample_moves, number_of_episodes])
        kv.tahni_tahy_vek(goals, zamichani)
        finished = np.full(number_of_episodes, False)

        states = kv.nova_kostka_vek(number_of_episodes)
        ep_states = np.zeros([move_limit, number_of_episodes, 2*np.prod(states.shape[1:])])
        ep_actions = np.zeros([move_limit, states.shape[0]])
        ep_lengths = np.zeros([number_of_episodes], dtype=int)
        #ep_states[:, :, 1] = goals # NOTE: tady bude chyba
        for i in range(move_limit):
            is_terminal = kv.je_stejna(states, goals)
            finished = np.logical_or(finished, is_terminal)
            not_finished = np.logical_not(finished)
            # TODO:
            # if np.all(finished):
            #     break

            states_goals = agent.merge_states_and_goals(states, goals)#np.stack([states, goals], axis=1) # NOTE: tady bude chyba
            probs = agent.predict(states_goals)
            # TODO: zrychlit (https://stackoverflow.com/questions/64673562/is-there-a-vectorized-way-to-sample-multiples-times-with-np-random-choice-with)
            actions = np.array([np.random.choice(len(p), p=p) for p in probs])
            moves = agent.indexy_na_tahy(actions)
            kv.tahni_tahy_vek(states, moves[None])

            print("states")
            kv.print_kostku_vek(states[:2])
            print("\ngoals")
            kv.print_kostku_vek(goals[:2])
            print("probabs", probs[:2])
            print("actions", actions[:2])
            print("moves", moves[:2])
            print("true", zamichani[:, :2])
            print()
            input()

            ep_states[i] = states_goals
            ep_actions[i] = actions
            ep_lengths += not_finished
        return ep_states, ep_actions, ep_lengths

    def make_her_episodes_vek():
        raise NotImplementedError

    assert episodes > 0, "Počet episod musí být alespoň 1."

    states, actions, ep_lengths = generate_episodes_vec(agent, sample_moves, move_limit, episodes)
    rewards = np.full(actions.shape, -1)
    returns = np.cumsum(rewards, axis=0)[::-1]

    num_transitions = ep_lengths.sum()
    all_states = np.empty([num_transitions, 2*6*3*3])
    all_actions = np.empty([num_transitions])
    all_returns = np.empty(all_actions.shape)

    i = 0
    for ep in range(episodes):
        for t in range(ep_lengths[ep]):
            all_states[i] = states[t, ep]#.reshape(-1)
            all_actions[i] = actions[t, ep]
            all_returns[i] = returns[t, ep]
            i += 1
    return all_states, all_actions, all_returns

    # --- HER ---
    _ = make_her_episodes_vek()
    her_goals = ...
    her_states = ...
    her_actions = actions.copy()
    her_returns = ...

    all_states = ...
    all_actions = ...
    all_returns = ...
    return all_states.reshape(-1, 2*6*3*3), all_actions.reshape(-1), all_returns.reshape(-1)


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

def solve_beam(goal: ko.Kostka, agent: HERCubeAgent, kandidatu: int, limit: int) -> int:
    #state = ko.nova_kostka()
    #kandidati = state.reshape([1, *state.shape])
    kandidati = kv.nova_kostka_vek(1)
    pr_kandidatu = np.ones([1, 1])

    for step in range(limit):
        if np.any(kv.je_stejna(kandidati, goal)):
            break

        kandidatu = len(kandidati)
        cube_shape = np.prod(kandidati.shape[1:])
        states_goals = np.empty([kandidatu, 2*cube_shape])
        states_goals[:, :cube_shape] = kandidati.reshape(kandidatu, -1)
        states_goals[:, cube_shape:] = goal.reshape(-1)

        predikce = agent.predict(states_goals)
        pr_nasledniku = pr_kandidatu * predikce # type: ignore
        pr_nasledniku_vektor = pr_nasledniku.reshape(-1)
        nej_indexy = np.argsort(pr_nasledniku_vektor)[-kandidatu:] # argsort setridi vzestupne

        indexy_otcu = nej_indexy // 12
        indexy_tahu = nej_indexy % 12

        kandidati = kandidati[indexy_otcu]
        tahy = agent.indexy_na_tahy(indexy_tahu)
        kv.tahni_tah_vek(kandidati, tahy)
        pr_kandidatu = pr_nasledniku[indexy_otcu, indexy_tahu].reshape(-1, 1)

        assert len(pr_kandidatu) == len(kandidati), f"Pocet kandidatu ({len(kandidati)}) a pravdepodobnosti ({len(pr_kandidatu)}) musi byt stejny."
        assert kandidati.shape[1:] == goal.shape, f"Nespravny tvar kandidatu {kandidati.shape}."
        assert len(kandidati) <= kandidatu, f"Prilis mnoho kandidatu {len(kandidati)}, limit je {kandidatu}."

    return 1 if np.any(kv.je_slozena(kandidati)) else 0

def solve_beam_vek(goal: ko.Kostka, agent: HERCubeAgent, kandidatu: int, limit: int) -> int:
    solved = 0
    for g in goal:
        solved += solve_beam(g, agent, kandidatu, limit)
    return solved

def solve_greedy_vek(cilova_kostka: kv.KostkaVek, agent: HERCubeAgent, limit: int) -> int:
    akt_kostka = kv.nova_kostka_vek(len(cilova_kostka))
    slozene = np.full(len(cilova_kostka), False)
    for _ in range(limit):
        slozene = np.logical_or(slozene, kv.je_stejna(akt_kostka, cilova_kostka))
        if np.all(slozene):
            break

        predikce = agent.predict(agent.merge_states_and_goals(akt_kostka, cilova_kostka))
        tahy = agent.indexy_na_tahy(np.argmax(predikce, axis=-1)) # type: ignore
        ko.print_kostku(akt_kostka[0])
        ko.print_kostku(cilova_kostka[0])
        print(predikce[0], tahy[0])
        input("------")
        kv.tahni_tah_vek(akt_kostka, tahy)
    input("KONEC")
    return np.count_nonzero(slozene)

def evaluate(agent: HERCubeAgent, batch_size: int, sample_moves: int, limit: int) -> None:
    goals = kv.nova_kostka_vek(batch_size)
    kv.tahni_tahy_vek(goals, kv.vygeneruj_nahodny_tah_vek([sample_moves, batch_size]))
    #num_solved = solve_beam_vek(goals, agent, 10, limit)
    num_solved = solve_greedy_vek(goals, agent, limit)
    agent.info["solved"] = f"{100*num_solved/batch_size:.2f} %"
    return
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
        #print(states.shape, states.dtype)
        #print(actions.shape, actions.dtype)
        #print(returns.shape, returns.dtype)
        #return
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
        train_sample_moves=1,
        train_ep_lim=2,
        eval_each=100,
        eval_batch_size=100,
        eval_sample_moves=1,
        eval_ep_lim=2
    )
