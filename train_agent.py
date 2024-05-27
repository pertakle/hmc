from network import Agent
import numpy as np
import kostka_vek as kv
import kostka as ko


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

# TODO: EvalSolveCallback
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
    return slozeno

def evaluate(agent: Agent, kostek: int, tahu: int) -> np.floating:
    d, t = generate_batch(kostek, tahu)
    pred = agent.predict(d)
    return np.mean((pred - t)**2)

def train_agent(batch_size: int, 
                sample_moves: int, 
                eval_each: int, 
                eval_batch_size: int, 
                eval_sample_moves: int,
                eval_lim: int) -> Agent:
    agent = Agent()
    for ep in range(100_000):
        for _ in range(eval_each):
            data, target = generate_batch(batch_size, sample_moves)
            agent.train(data, target)
        slozenych = evaluate_solve(agent, eval_batch_size, eval_sample_moves, eval_lim)
        print(f"Evaluation after {ep+1} epochs:", " - ".join(map(lambda x: f"{x[0]} {x[1]:.4f}", agent.info.items())), f"- solved {slozenych}/{eval_batch_size}")
    return agent

if __name__ == "__main__":
    train_agent(
        batch_size=10_000,
        sample_moves=27,
        eval_each=100,
        eval_batch_size=10,
        eval_sample_moves=15,
        eval_lim=30
    )
