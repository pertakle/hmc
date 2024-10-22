import numpy as np

class HERCubeAgent:

    def __init__(self) -> None:
        self.info = {}

    def predict_move(self, stategoals: np.ndarray, greedy: float) -> np.ndarray:
        raise NotImplementedError
    
    def train(self, episodes: np.ndarray) -> None:
        pass
