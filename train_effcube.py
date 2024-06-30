
from network import Agent
import numpy as np
import kostka_vek as kv
import kostka as ko
from copy import deepcopy
import tqdm
from typing import Any


def solve_beam(kostka: ko.Kostka, agent: Agent, kandidatu: int, limit: int) -> int:
    kandidati = kostka.copy()
    pravdepodobnosti = np.ones(kandidatu)

    for step in range(limit):
        probs = agent.predict(kandidati)
        raise NotImplementedError

def solve_greedy(kostka: kv.KostkaVek, agent: Agent, limit: int) -> int:
    slozene = np.full(len(kostka), False)
    for _ in range(limit):
        slozene = np.logical_or(slozene, kv.je_slozena(kostka))
        if np.all(slozene):
            break

        predikce = agent.predict(kostka)
        tahy = agent.indexy_na_tahy(np.argmax(predikce, axis=-1))
        kv.tahni_tah_vek(kostka, tahy)

    return np.count_nonzero(slozene)
        
def evaluate(agent: Agent, n: int, tahu: int, limit: int) -> None:
    kostka = kv.nova_kostka_vek(n)
    kv.tahni_tahy_vek(kostka, kv.vygeneruj_nahodny_tah_vek([tahu, n]))
    slozeno = solve_greedy(kostka, agent, limit)
    agent.info["solved"] = f"{100*slozeno/n:.2f} %"

       
def generate_batch(kostek: int, tahu: int) -> tuple[np.ndarray, np.ndarray]:
    FEATUR = 54 # np.prod(k[0].shape)
    k = kv.nova_kostka_vek(kostek)

    data = np.empty([tahu*kostek, FEATUR], dtype=np.uint8)
    target = np.zeros([tahu*kostek], dtype=np.int64)

    tahy = kv.vygeneruj_nahodny_tah_vek([tahu, kostek])

    for t, tah in enumerate(tahy):
        kv.tahni_tah_vek(k, tah)
        data[t*kostek:(t+1)*kostek] = k.reshape(-1, FEATUR)
        target[t*kostek:(t+1)*kostek] = -tah #inverzni tah

    return data, target


def format_float(x: float) -> str:
    return f"{x:.4f}" if x >= 1e-4 else f"{x:.4e}"

def format_info(info: dict[str, Any]) -> str:
    return " - ".join(map(lambda x: f"{x[0]} {format_float(x[1]) if isinstance(x[1], float) else x[1]}", info.items()))

def train_eff_cube(steps: int,
                   batch_size: int,
                   sample_moves: int,
                   eval_each: int,
                   eval_batch_size: int,
                   eval_sample_moves: int,
                   eval_lim: int) -> Agent:
    def new_bar(steps: int) -> tqdm.tqdm:
        return tqdm.tqdm(total=steps, desc="Training", leave=True)
 

    agent = Agent()
    bar = new_bar(min(eval_each, steps))

    for step in range(1, steps + 1):
        data, target = generate_batch(batch_size, sample_moves)
        agent.train(data, target)

        bar.update()
        if (step % eval_each) == 0 or step == steps: 
            evaluate(agent, eval_batch_size, eval_sample_moves, eval_lim)
            
            bar.bar_format = f'{{desc}} {format_info(agent.info)} [{{elapsed}}, {{rate_fmt}}{{postfix}}]'
            bar.set_description(f"Evaluation after {step} steps", False)
            bar.close()
            if step < steps:
                bar = new_bar(min(eval_each, steps - step))

    return agent


if __name__ == "__main__":
    train_eff_cube(
        steps=100_000,
        batch_size=128,
        sample_moves=26,
        eval_each=100,
        eval_batch_size=10000,
        eval_sample_moves=26,
        eval_lim=50
    )
