import numpy as np
from typing import Any, Tuple
from nn.her_cube_agent import HERCubeAgent
import kostka.kostka_vek as kv
import kostka.kostka as ko
import tqdm

def generate_batch(agent: HERCubeAgent, episodes: int, sample_moves: int, move_limit: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    def generate_episodes_vec2(
        agent: HERCubeAgent,
        sample_moves: int,
        move_limit: int,
        episodes: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates `episodes` vectorized episodes with `sample_moves` lengths.
        Goals are randomly scrambled cubes with `sample_moves` uniformly random moves.

        Returns:
            - States: ndarray, dtype=float, shape=[`move_limit + 1`, `number_of_episodes`, 2 * cube features]
            - Actions: ndarray, dtype=float, shape=[`move_limit`, `number_of_episodes`]
            - Episode lengths: ndarray, dtype=int, shape=[`number_of_episodes`]
        """
        
        STATE_GOAL_FEATURES = 2 * 6 * 3 * 3 # two flattened cubes
        goals = kv.nova_kostka_vek(episodes)
        kv.zamichej_nahodnymi_tahy_vek(goals, sample_moves)

        states = kv.nova_kostka_vek(episodes)
        dones = np.full(episodes, False)

        ep_state_goals = np.zeros([episodes, move_limit + 1, STATE_GOAL_FEATURES])
        ep_actions = np.zeros([episodes, move_limit])
        ep_lengths = np.zeros(episodes)

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

        return ep_state_goals, ep_actions, ep_lengths

    def her_state_goals_last_state(
            state_goals: np.ndarray, 
            ep_lengths: np.ndarray
        ) -> np.ndarray:
        """
        Makes new transitions with goals reached at the end of episodes.
        NOTE: Keeps state_goals.shape with the same episodes paddings.

        Returns:
            - state_goals with replaced goals
        """
        
        CUBE_FEATURES = 6*3*3

        her_goals = state_goals[:, ep_lengths, CUBE_FEATURES:]
        her_state_goals = state_goals.copy()
        her_state_goals[:, :, CUBE_FEATURES:] = her_goals

        return her_state_goals

    def prepare_padded_episodes(
            episodes: np.ndarray, 
            actions: np.ndarray, 
            ep_lengths: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            - state_goals
            - actions
            - returns
        """
        CUBE_FEATURES = 6 * 3 * 3
        num_transitions = ep_lengths.sum()
        state_goals = np.zeros([num_transitions, CUBE_FEATURES * 2], dtype=int)
        actions = np.zeros(num_transitions, dtype=int)
        returns = np.zeros(num_transitions)
        

        # TODO: vektorizace
        transition = 0
        for ep in range(len(ep_lengths)):
            for step in range(ep_lengths[ep]):
                state_goals[transition] = episodes[ep, step]
                actions[transition] = actions[ep, step]
                returns[transition] = -ep_lengths[ep] + step

                transition += 1
        return state_goals, actions, returns

    raise NotImplementedError("Implement `generate_batch` using new functions above.")


    def compute_returns(ep_lengths: np.ndarray, gamma: float = 1) -> np.ndarray:
        # TODO: optimalizace pokud vse skoncilo driv nez move limit
        # TODO: vektorizace
        returns = np.array([
            -ep_lengths + i*gamma
            for i in range(move_limit)
        ])
        return returns

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
        kv.zamichej_nahodnymi_tahy_vek(goals, sample_moves)
        finished = np.full(number_of_episodes, False)

        CUBE_LEN = 6*3*3
        states = kv.nova_kostka_vek(number_of_episodes)
        ep_states = np.zeros([move_limit, number_of_episodes, 2*CUBE_LEN])
        ep_actions = np.zeros([move_limit, number_of_episodes])
        ep_lengths = np.zeros([number_of_episodes], dtype=int)

        for i in range(move_limit):
            is_terminal = kv.je_stejna(states, goals)
            finished = np.logical_or(finished, is_terminal)
            not_finished = np.logical_not(finished)
            # TODO:
            # if np.all(finished):
            #     break

            states_goals = agent.merge_states_and_goals(states, goals)
            probs = agent.predict_action_probs(states_goals)
            # TODO: zrychlit (https://stackoverflow.com/questions/64673562/is-there-a-vectorized-way-to-sample-multiples-times-with-np-random-choice-with)
            actions = np.array([np.random.choice(len(p), p=p) for p in probs])
            moves = agent.indexy_na_tahy(actions)
            kv.tahni_tahy_vek(states, moves[None])

            assert ep_states[i].shape == states_goals.shape
            assert ep_actions[i].shape == actions.shape
            ep_states[i] = states_goals
            ep_actions[i] = actions
            ep_lengths += not_finished
        return ep_states, ep_actions, ep_lengths

    def make_her_episodes_vek(states: np.ndarray, actions: np.ndarray):
        her_goals = states[:, :6*3*3].reshape(-1, 6, 3, 3)
        her_moves = agent.indexy_na_tahy(actions)
        kv.tahni_tah_vek(her_goals, her_moves)
        her_state_goals = agent.merge_states_and_goals(states[:, :6*3*3], her_goals)
        returns = np.full(len(her_state_goals), -1)
        return her_state_goals, actions, returns

    assert episodes > 0, "Počet episod musí být alespoň 1."

    states, actions, ep_lengths = generate_episodes_vec(agent, sample_moves, move_limit, episodes)
    returns = compute_returns(ep_lengths)

    num_transitions = ep_lengths.sum()
    all_states = np.empty([num_transitions, 2*6*3*3], dtype=int)
    all_actions = np.empty([num_transitions], dtype=int)
    all_returns = np.empty(all_actions.shape)

    # TODO: vektorizace
    i = 0
    for ep in range(episodes):
        for t in range(ep_lengths[ep]):
            all_states[i] = states[t, ep].reshape(2*6*3*3)
            all_actions[i] = actions[t, ep]
            all_returns[i] = returns[t, ep]
            i += 1

    her_states, her_actions, her_returns = make_her_episodes_vek(all_states, all_actions)
    all_states = np.vstack((all_states, her_states))
    all_actions = np.hstack((all_actions, her_actions))
    all_returns = np.hstack((all_returns, her_returns))
    assert np.all(np.logical_and(-move_limit <= all_returns, all_returns < 0))
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
    slozene = np.full(len(cilova_kostka), False)
    for _ in range(limit):
        slozene = np.logical_or(slozene, kv.je_stejna(akt_kostka, cilova_kostka))
        if np.all(slozene):
            break

        predikce = agent.predict_action_probs(agent.merge_states_and_goals(akt_kostka, cilova_kostka))
        tahy = agent.indexy_na_tahy(np.argmax(predikce, axis=-1)) # type: ignore
        kv.tahni_tah_vek(akt_kostka, tahy)
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
        train_episodes=64,
        train_sample_moves=15,
        train_ep_lim=30,
        eval_each=100,
        eval_batch_size=100,
        eval_sample_moves=5,
        eval_ep_lim=30
    )
