import numpy as np

class HERCubeAgent:

    def __init__(self) -> None:
        self.info = {}

    def predict_action(self, stategoals: np.ndarray, greedy: float) -> np.ndarray:
        """
        Predicts action for each stategoal.

        Params:
            - `stategoals`: batch of stategoals
            - `greedy`: True when the agent should use best action possible

        Returns:
            - batch of selected actions for `stategoals`
        """
        raise NotImplementedError
    
    def train(self, episodes: np.ndarray) -> None:
        pass
