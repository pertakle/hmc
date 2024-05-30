from network import Agent
import numpy as np
import kostka_vek as kv
import kostka as ko
from copy import deepcopy
import tqdm
from typing import Any


def generate_batch(kostek: int, tahu: int) -> tuple[np.ndarray, np.ndarray]:
    FEATUR = 54
    k = kv.nova_kostka_vek(kostek)
    data = np.empty([tahu*kostek, FEATUR], dtype=np.uint8)
    target = np.zeros([tahu*kostek], dtype=np.uint8)
    tahy = np.random.randint(1, 7, [tahu, kostek]) * np.random.choice([-1, 1], [tahu, kostek])

    for t, tah in enumerate(tahy):
        data[t*kostek:(t+1)*kostek] = k.reshape(-1, FEATUR)
        target[t*kostek:(t+1)*kostek] = t
        kv.tahni_tah_vek(k, tah)

    return data, target


def evaluate_solve(agent: Agent, kostek: int, tahu: int, limit: int) -> int:
    k = kv.nova_kostka_vek(kostek)
    tahy = np.random.randint(1, 7, [tahu, kostek]) * np.random.choice([-1, 1], [tahu, kostek])
    kv.tahni_tahy_vek(k, tahy)
    slozeno = 0
    for kk in k:
        for _ in range(limit):
            if ko.je_slozena(kk):
                slozeno += 1
                break
            moznosti = np.stack([kk]*12)
            kv.tahni_tah_vek(moznosti, 
                             np.hstack([np.arange(1,7, dtype=np.int64), np.arange(1,7, dtype=np.int64)], dtype=np.int64) * \
                             np.hstack([np.ones(6, dtype=np.int64), np.full(6, -1, dtype=np.int64)], dtype=np.int64)
                             )
            pred = agent.predict(moznosti)
            kk = moznosti[np.argmin(pred)]
    agent.info["solved"] = f"{slozeno}/{kostek}"
    return slozeno


def train_agent(steps: int,
                batch_size: int, 
                sample_moves: int, 
                eval_each: int, 
                eval_batch_size: int, 
                eval_sample_moves: int,
                eval_lim: int) -> Agent:
    agent = Agent()
    for step in range(1, steps + 1):
        data, target = generate_batch(batch_size, sample_moves)
        agent.train(data, target)

        if (step % eval_each) == 0:
            slozenych = evaluate_solve(agent, eval_batch_size, eval_sample_moves, eval_lim)
            print(f"Evaluation after {step} steps:", " - ".join(map(lambda x: f"{x[0]} {x[1]:.4f}", agent.info.items())), f"- solved {slozenych}/{eval_batch_size}")
    return agent



def generate_val_iter_batch(agent_target: Agent, kostek: int, tahu: int) -> tuple[np.ndarray, np.ndarray]:
    VSECHNY_TAHY = np.hstack([np.arange(1, 7), -np.arange(1,7)]*kostek)
    k = kv.nova_kostka_vek(kostek)
    max_tahu = np.random.randint(0, tahu+1, [kostek])
    indexy = np.arange(kostek)
    data = k.copy()

    for i in range(np.max(max_tahu) + 1): 
        ulozit = indexy[max_tahu == i]
        data[ulozit] = k[ulozit]

        # nahodny tah
        # TODO: neefektivne tah vsemi
        tahy = kv.vygeneruj_nahodny_tah_vek(kostek)
        kv.tahni_tah_vek(k, tahy) 


    moznosti = np.repeat(data, 12, axis=0)
    kv.tahni_tah_vek(moznosti, VSECHNY_TAHY)

    predikce = agent_target.predict(moznosti)
    predikce= predikce * np.logical_not(kv.je_slozena(moznosti))
    predikce = predikce.reshape([kostek, 12])
    target = np.min(predikce, axis=1) + 1
    target *= max_tahu > 0 # pro nezamichane nas min synu nezajima

    return data, target
    
    
def format_float(x: float) -> str:
    return f"{x:.4f}" if x >= 1e-4 else f"{x:.4e}"

def format_info(info: dict[str, Any]) -> str:
    return " - ".join(map(lambda x: f"{x[0]} {format_float(x[1]) if isinstance(x[1], float) else x[1]}", info.items()))

def train_value_iteration(steps: int,
                          batch_size: int,
                          sample_moves: int,
                          copy_each: int,
                          epsilon: float,
                          eval_each: int,
                          eval_batch_size: int,
                          eval_sample_moves: int,
                          eval_lim: int) -> Agent:
    def new_bar(steps: int) -> tqdm.tqdm:
        return tqdm.tqdm(total=steps, desc="Training", leave=True)
 

    agent_behave = Agent()
    agent_target = deepcopy(agent_behave)
    bar = new_bar(min(eval_each, steps))

    for step in range(1, steps + 1):
        data, target = generate_val_iter_batch(agent_target, batch_size, sample_moves)
        agent_behave.train(data, target)

        if (step % copy_each) == 0 and agent_behave.info["mse_loss"] < epsilon:
            agent_target = deepcopy(agent_behave)

        bar.update()
        if (step % eval_each) == 0 or step == steps: 
            evaluate_solve(agent_behave, eval_batch_size, eval_sample_moves, eval_lim)
            
            bar.bar_format = '{desc} [{elapsed}, {rate_fmt}{postfix}]'
            bar.set_description(f"Evaluation after {step} steps: {format_info(agent_behave.info)}", False)
            bar.close()
            if step < steps:
                bar = new_bar(min(eval_each, steps - step))

    return agent_behave




if __name__ == "__main__":
    if True:
        train_value_iteration(
            steps=1_0,
            batch_size=1024,
            sample_moves=30,
            copy_each=100,
            epsilon=0.05,
            eval_each=100,
            eval_batch_size=100,
            eval_sample_moves= 13,
            eval_lim=30
        )
    else:
        train_agent(
            steps=1_000_000,
            batch_size=64,
            sample_moves=15,
            eval_each=100,
            eval_batch_size=10,
            eval_sample_moves=15,
            eval_lim=30
        )
