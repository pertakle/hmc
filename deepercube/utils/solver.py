from deepercube.nn.her_cube_agent import A2CCubeAgent
import numpy as np
import numpy.typing as npt
import deepercube.kostka.kostka as ko
import deepercube.kostka.kostka_vek as kv
from typing import TypeVar, Tuple

def solve_beam(goal: ko.Kostka, agent: A2CCubeAgent, beam_size: int, limit: int) -> int:
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
        states_goals[:, CUBE_FEATURES:] = goal.reshape(1, -1)

        predikce = agent.predict_action_probs(states_goals)
        pr_nasledniku = pr_kandidatu * predikce # type: ignore

        pr_nasledniku_vektor = pr_nasledniku.reshape(-1)
        nej_indexy = np.argsort(pr_nasledniku_vektor)[-beam_size:] # argsort setridi vzestupne

        indexy_otcu = nej_indexy // 12
        indexy_tahu = nej_indexy % 12

        kandidati = kandidati[indexy_otcu]
        tahy = agent.indexy_na_tahy(indexy_tahu)
        kv.tahni_tah_vek(kandidati, tahy)
        pr_kandidatu = pr_nasledniku[indexy_otcu, indexy_tahu].reshape(-1, 1)

        assert len(pr_kandidatu) == len(kandidati), f"Pocet kandidatu ({len(kandidati)}) a pravdepodobnosti ({len(pr_kandidatu)}) musi byt stejny."
        assert kandidati.shape[1:] == goal.shape, f"Nespravny tvar kandidatu {kandidati.shape}."
        assert len(kandidati) <= beam_size, f"Prilis mnoho kandidatu {len(kandidati)}, limit je {kandidatu}."

    return 1 if np.any(kv.je_stejna(kandidati, goal)) else 0

def solve_beam_vek(goal: kv.KostkaVek, agent: A2CCubeAgent, beam_size: int, limit: int) -> int:
    solved = 0
    for g in goal:
        solved += solve_beam(g, agent, beam_size, limit)
    return solved

B = TypeVar("B", bound=int)
S = TypeVar("S", bound=int)
KostkaVekBatch = np.ndarray[Tuple[B, S, 6, 3, 3], np.dtype[ko.ColorT]]
def any_solved(states: KostkaVekBatch, goals: kv.KostkaVek) -> np.ndarray[Tuple[B], np.dtype[np.bool_]]:
    solved = np.full(len(states), False)
    return solved
    ...

def solve_beam_vekevk(goals: kv.KostkaVek, agent: A2CCubeAgent, beam_size: int, limit: int) -> int:

    BATCH_SIZE = len(goals)
    kandidati = kv.nova_kostka_vek(BATCH_SIZE)[BATCH_SIZE, np.newaxis]
    pr_kandidatu = np.ones([BATCH_SIZE, 1, 1])
    solved = np.full(BATCH_SIZE, False)

    for step in range(limit):
        # check goals in kandidates
        any_solved = any_solved(kandidati, goals)
        solved = np.logical_or(solved, any_solved)
        # create stategoals
        # compute cumulative probs of successors
        # get moves of selected successors
        # move states
        # store new cumulative probabilities
        ...

    # return number of solved states
    return np.count_nonzero(solved)

def solve_greedy_vek(cilova_kostka: kv.KostkaVek, agent: A2CCubeAgent, limit: int) -> int:
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
