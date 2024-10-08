import numpy as np
from typing import Any, Tuple, List
from nn.her_cube_agent import HERCubeAgent
import kostka.kostka_vek as kv
import kostka.kostka as ko
import tqdm

def generate_batch(
        agent: HERCubeAgent, 
        episodes: int, 
        sample_moves: int, 
        move_limit: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    def generate_episodes_vec(
        agent: HERCubeAgent,
        sample_moves: int,
        move_limit: int,
        episodes: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates `episodes` vectorized episodes with `sample_moves` lengths.
        Goals are randomly scrambled cubes with `sample_moves` uniformly random moves.

        Returns:
            - States: ndarray, dtype=float, shape=[`move_limit + 1`, `episodes`, 2 * cube features]
            - Actions: ndarray, dtype=float, shape=[`move_limit`, `episodes`]
            - Returns: ndarray, dtype=float, shape=[`episodes`]
                reward is `-1` for each action, episode terminates when reaching goal
            - Episode lengths: ndarray, dtype=int, shape=[`episodes`]
        """
        
        STATE_GOAL_FEATURES = 2 * 6 * 3 * 3 # two flattened cubes
        goals = kv.nova_kostka_vek(episodes)
        kv.zamichej_nahodnymi_tahy_vek(goals, sample_moves)

        states = kv.nova_kostka_vek(episodes)
        dones = np.full(episodes, False)

        ep_state_goals = np.zeros([episodes, move_limit + 1, STATE_GOAL_FEATURES])
        ep_actions = np.zeros([episodes, move_limit])
        ep_returns = np.zeros([episodes, move_limit])
        ep_lengths = np.zeros(episodes, dtype=int)

        for step in range(move_limit + 1):
            solved = kv.je_stejna(states, goals)
            dones = np.logical_or(dones, solved)

            # NOTE: breaks last episode states
            # if np.all(dones):
            #     break

            # choose an action in the env
            state_goals = agent.merge_states_and_goals(states, goals) # TODO: reshape?
            probs: np.ndarray = agent.predict_action_probs(state_goals) #type: ignore
            assert probs.shape[1] == 12
            actions = np.array([np.random.choice(12, p=distribution) for distribution in probs])

            # store the transitions
            ep_state_goals[:, step] = state_goals.reshape([episodes, -1])
            if step < move_limit: # if not end of episode
                ep_actions[:, step] = actions
                ep_lengths += 1 - dones

            # make the action
            moves = agent.indexy_na_tahy(actions)
            kv.tahni_tah_vek(states, moves)

            if step < move_limit:
                # TODO: efektivneji next_stategoal
                next_stategoals = agent.merge_states_and_goals(states, goals)
                estimates = agent.predict_values(next_stategoals).squeeze()
                GAMMA = 1
                ep_returns[:, step] = -1 + GAMMA * estimates * (1 - kv.je_stejna(states, goals))

        return ep_state_goals, ep_actions, ep_returns, ep_lengths

    def stategoal_terminal(stategoals: np.ndarray) -> np.ndarray:
        CUBE_FEATURES = 6 * 3 * 3
        return kv.je_stejna(
                stategoals[:, :CUBE_FEATURES].reshape(-1, 6, 3, 3),
                stategoals[:, CUBE_FEATURES:].reshape(-1, 6, 3, 3)
        )


    def her_state_goals_last_state(
            agent: HERCubeAgent,
            episodes: np.ndarray, 
            actions: np.ndarray,
            returns: np.ndarray,
            ep_lengths: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Makes new transitions with goals reached at the end of episodes.
        NOTE: Keeps episodes.shape with the same episodes paddings.

        Params:
            - agent
            - episodes: 
                shape=[num episodes, padded ep lengths + 1, 2 * CUBE FEATURES],
                episodes contain the final state as the last state
                `state_goals.shape[1] > ep_lengths `
            - actions:
                shape=[num episodes, padded ep lengths]
            - returns:
                shape=[num episodes, padded ep lengths]
            - ep_lengths:
                vector of episodes lengths
                shape=[num episodes]

        Returns:
            - HER episodes with replaced goals
            - actions: just a copy
            - HER returns
            - ep lengths: jus a copy
        """
        
        CUBE_FEATURES = 6*3*3
        EPISODES = actions.shape[0]
        PADDED_EP_LEN = actions.shape[1]

        her_goals = np.array([episodes[ep, length, :CUBE_FEATURES] for ep, length in enumerate(ep_lengths)])

        def f(agent, episodes, actions, returns, ep_lengths):
            print(episodes.shape)
            t = agent.indexy_na_tahy(actions)
            for ep, epl in enumerate(ep_lengths):
                print(f"\n\nEpisoda {ep}, delka {epl}.")
                print(f"Returns: {returns[ep, :epl]}")
                print(f"Cil:")
                ko.print_kostku(episodes[ep, 0, CUBE_FEATURES:].astype(int).reshape(6, 3, 3))
                for step in range(epl):
                    print("Stav:")
                    ko.print_kostku(episodes[ep, step, :CUBE_FEATURES].astype(int).reshape(6, 3, 3))
                    print(f"return {returns[ep, step]:.3f}")
                    print(f"krok {step}, akce {int(actions[ep, step])} == tah {int(t[ep, step])}")
                print(f"terminal {epl}")
                ko.print_kostku(episodes[ep, epl, :CUBE_FEATURES].astype(int).reshape(6, 3, 3))
                
        #f(agent, episodes, actions, returns, ep_lengths)

        
        #print("\n\nher goals")
        #kv.print_kostku_vek(her_goals.astype(int).reshape(-1, 6, 3, 3))

        her_goals = her_goals[:, np.newaxis]
        her_episodes = episodes.copy()
        her_episodes[:, :, CUBE_FEATURES:] = her_goals

        #for ep, epl in enumerate(ep_lengths):
        #    print("\nEP", ep)
        #    for step in range(epl):
        #        ko.print_kostku(her_episodes[ep, step, CUBE_FEATURES:].astype(int).reshape(6, 3, 3))


        #print("=======")

        her_actions = actions.copy()

        her_ep_lengths = np.zeros(EPISODES, dtype=int)
        #not_dones = np.logical_not(stategoal_terminal(her_episodes[:, 0]))
        not_dones = np.full(EPISODES, True)
        for step in range(her_episodes.shape[1] - 1):
            her_ep_lengths += not_dones
            not_terminated = np.logical_not(stategoal_terminal(her_episodes[:, step+1]))
            not_dones = np.logical_and(not_dones, not_terminated)

        # TODO:
        assert np.all(her_ep_lengths > 0), "Zero lenght episode, panic!"

        # Compute returns
        her_returns = np.zeros(her_actions.shape)
        GAMMA = 1
        
        #NOTE: no need for V estimate - her episodes always end
        herret = -np.ones(EPISODES)
        for step in range(her_episodes.shape[1]-2, -1, -1):
            her_returns[:, step] = herret
            herret = -1 + GAMMA * herret

        #print("\n\n\n-----HER-----\n\n\n")
        #f(agent, her_episodes, her_actions, her_returns, her_ep_lengths)
        #exit()

        return her_episodes, her_actions, her_returns, her_ep_lengths

    def unroll_padded_episodes(
            episodes: np.ndarray, 
            actions: np.ndarray, 
            returns: np.ndarray,
            ep_lengths: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Unrolls all episodes and actions into batch of data.

        Params:
            - episodes:
            - actions
            - returns
            - ep_lengths

        Returns:
            - stategoals: batch of stategoals
            - actions: 1d vector of actions
            - returns: 1d vector of returns
        """
        CUBE_FEATURES = 6 * 3 * 3
        num_transitions = ep_lengths.sum()
        unrolled_stategoals = np.zeros([num_transitions, CUBE_FEATURES * 2], dtype=int)
        unrolled_actions = np.zeros(num_transitions, dtype=int)
        unrolled_returns = np.zeros(num_transitions)
        

        # TODO: vektorizace
        transition = 0
        for ep in range(len(ep_lengths)):
            for step in range(ep_lengths[ep]):
                unrolled_stategoals[transition] = episodes[ep, step]
                unrolled_actions[transition] = actions[ep, step]
                unrolled_returns[transition] = returns[ep, step]

                transition += 1
        return unrolled_stategoals, unrolled_actions, unrolled_returns

    def merge_data(data_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        stategoals = np.vstack([x[0] for x in data_list])
        actions = np.hstack([x[1] for x in data_list])
        returns = np.hstack([x[2] for x in data_list])
        return stategoals, actions, returns

    padded_ep = generate_episodes_vec(agent, sample_moves, move_limit, episodes)
    padded_her_ep = her_state_goals_last_state(agent, *padded_ep)

    data = unroll_padded_episodes(*padded_ep)
    her_data = unroll_padded_episodes(*padded_her_ep)
    
    return merge_data([data, her_data])

def solve_beam(goal: ko.Kostka, agent: HERCubeAgent, kandidatu: int, limit: int) -> int:
    # TODO: FIX
    kandidati = kv.nova_kostka_vek(1)
    pr_kandidatu = np.ones([1, 1])

    for step in range(limit):
        if np.any(kv.je_stejna(kandidati, goal)):
            break

        kandidatu = len(kandidati)
        CUBE_FEATURES = 6*3*3
        states_goals = np.empty([kandidatu, 2*CUBE_FEATURES])
        states_goals[:, :CUBE_FEATURES] = kandidati.reshape(kandidatu, -1)
        states_goals[:, CUBE_FEATURES:] = goal.reshape(-1)

        predikce = agent.predict_action_probs(states_goals)
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
    slozene = kv.je_stejna(akt_kostka, cilova_kostka)
    for _ in range(limit):
        predikce = agent.predict_action_probs(agent.merge_states_and_goals(akt_kostka, cilova_kostka))
        tahy = agent.indexy_na_tahy(np.argmax(predikce, axis=-1)) # type: ignore
        
        kv.tahni_tah_vek(akt_kostka, tahy)

        slozene = np.logical_or(slozene, kv.je_stejna(akt_kostka, cilova_kostka))
        if np.all(slozene):
            break

    return np.count_nonzero(slozene)

def evaluate(agent: HERCubeAgent, batch_size: int, sample_moves: int, limit: int) -> None:
    goals = kv.nova_kostka_vek(batch_size)
    kv.zamichej_nahodnymi_tahy_vek(goals, sample_moves)
    #num_solved = solve_beam_vek(goals, agent, 10, limit)
    num_solved = solve_greedy_vek(goals, agent, limit)
    agent.info["solved"] = f"{100*num_solved/batch_size:.2f} %"

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
        train_episodes=256,
        train_sample_moves=7,
        train_ep_lim=7,
        eval_each=100,
        eval_batch_size=10000,
        eval_sample_moves=7,
        eval_ep_lim=7
    )
