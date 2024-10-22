import numpy as np
from deepercube.nn.her_cube_agent import A2CCubeAgent
from typing import Tuple, List
import deepercube.kostka.kostka as ko
import deepercube.kostka.kostka_vek as kv

def generate_batch(
        agent: A2CCubeAgent, 
        episodes: int, 
        sample_moves: int, 
        move_limit: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    def compute_ep_lengths(episodes: np.ndarray) -> np.ndarray:
        EPISODES = len(episodes)
        her_ep_lengths = np.zeros(EPISODES, dtype=int)
        not_dones = np.full(EPISODES, True)
        for step in range(episodes.shape[1] - 1):
            her_ep_lengths += not_dones
            not_terminated = np.logical_not(stategoal_terminal(episodes[:, step+1]))
            not_dones = np.logical_and(not_dones, not_terminated)
        return her_ep_lengths

    def get_last_next_stategoals(episodes: np.ndarray, ep_lengths: np.ndarray) -> np.ndarray:
        return np.array([episodes[i, l] for i, l in enumerate(ep_lengths)])

    def set_goals(episodes: np.ndarray, goals: np.ndarray) -> np.ndarray:
        CUBE_FEATURES = 6*3*3
        if len(goals.shape) != len(episodes.shape):
            goals = goals[:, np.newaxis]
        new_episodes = episodes.copy()
        new_episodes[:, :, CUBE_FEATURES:] = goals
        return new_episodes

    def compute_returns(agent: A2CCubeAgent, episodes: np.ndarray, ep_lengths: np.ndarray) -> np.ndarray:
        """
        Params:
            - agent: current agent
            - episodes: shape=[num episodes, max ep len, features], 
            all episodes are padded to the length of the longest episode
            - ep_lengths: lengths of episodes
        
        Returns:
            - returns: reward is -1 for each step, gamma is 1
        """
        EPISODES = episodes.shape[0]
        MOVE_LIMIT = episodes.shape[1] - 1
        GAMMA = 1
        last_next_stategoals = get_last_next_stategoals(episodes, ep_lengths)
        v_last_next_stategoals: np.ndarray = agent.predict_values(last_next_stategoals).squeeze() #type: ignore
        terminal = stategoal_terminal(last_next_stategoals)
        returns_t = -1 + GAMMA * v_last_next_stategoals * np.logical_not(terminal)
        returns = np.zeros([EPISODES, MOVE_LIMIT], dtype=np.float64)
        for t in range(MOVE_LIMIT - 1, -1, -1): # for steps backward
            returns[:, t] = returns_t
            returns_t = -1 + GAMMA * returns_t
        return returns

    def generate_episodes_vec(
        agent: A2CCubeAgent,
        sample_moves: int,
        move_limit: int,
        episodes: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates `episodes` vectorized episodes with `sample_moves` lengths.
        Goals are randomly scrambled cubes with `sample_moves` uniformly random moves.

        Returns:
            - State-Goals: ndarray, dtype=float, shape=[`episodes`, `move_limit + 1`, 2 * cube features]
            - Actions: ndarray, dtype=float, shape=[`episodes`, `move_limit`]
            - Returns: ndarray, dtype=float, shape=[`episodes`, `move_limit`]
                reward is `-1` for each action, episode terminates when reaching goal
            - Episode lengths: ndarray, dtype=int, shape=[`episodes`]
        """
        
        STATE_GOAL_FEATURES = 2 * 6 * 3 * 3 # two flattened cubes
        goals = kv.nova_kostka_vek(episodes)
        kv.zamichej_nahodnymi_tahy_vek(goals, sample_moves)

        states = kv.nova_kostka_vek(episodes)
        dones = np.full(episodes, False)

        ep_stategoals = np.zeros([episodes, move_limit + 1, STATE_GOAL_FEATURES])
        ep_actions = np.zeros([episodes, move_limit])

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
            ep_stategoals[:, step] = state_goals.reshape([episodes, -1])
            if step < move_limit: # if not end of episode
                ep_actions[:, step] = actions

            # make the action
            moves = agent.indexy_na_tahy(actions)
            kv.tahni_tah_vek(states, moves)

        ep_lengths = compute_ep_lengths(ep_stategoals)
        ep_returns = compute_returns(agent, ep_stategoals, ep_lengths)
        return ep_stategoals, ep_actions, ep_returns, ep_lengths

    def stategoal_terminal(stategoals: np.ndarray) -> np.ndarray:
        CUBE_FEATURES = 6 * 3 * 3
        return kv.je_stejna(
                stategoals[:, :CUBE_FEATURES].reshape(-1, 6, 3, 3),
                stategoals[:, CUBE_FEATURES:].reshape(-1, 6, 3, 3)
        )


    def make_her_last_state(
            agent: A2CCubeAgent,
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
        her_last_next_stategoals = get_last_next_stategoals(episodes, ep_lengths)
        
        her_goals = her_last_next_stategoals[:, :CUBE_FEATURES]
        her_episodes = set_goals(episodes, her_goals)
        her_actions = actions.copy()
        her_ep_lengths = compute_ep_lengths(her_episodes)

        #assert np.all(her_ep_lengths > 0), "Zero lenght episode, panic!"
        her_returns = compute_returns(agent, her_episodes, her_ep_lengths)

        return her_episodes, her_actions, her_returns, her_ep_lengths

    def unroll_padded_episodes(
            episodes: np.ndarray, 
            actions: np.ndarray, 
            returns: np.ndarray,
            ep_lengths: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unrolls all episodes and actions into batch of data.

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

    padded_her_ep = make_her_last_state(agent, *padded_ep)

    data = unroll_padded_episodes(*padded_ep)
    her_data = unroll_padded_episodes(*padded_her_ep)

    return merge_data([data, her_data])
