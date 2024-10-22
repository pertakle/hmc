from deepercube.nn.her_cube_agent import HERCubeAgent
import torch
import numpy as np

class DQNAgent(HERCubeAgent):

    def __init__(self) -> None:
        super().__init__()

    def predict_move(self, stategals: np.ndarray, greedy: float) -> np.ndarray:
        raise NotImplementedError
    
    def train(self, episodes: np.ndarray) -> None:
        pass
