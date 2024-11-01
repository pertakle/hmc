import numpy as np

class Agent:
    """
    Abstract class defining an interface of an agent.
    """

    def __init__(self) -> None:
        self.info = {}

    def predict_actions(self, states: np.ndarray, greedy: float) -> np.ndarray:
        """
        Predicts action for each state.

        Params:
            - `states`: batch of states
            - `greedy`: True when the agent should use best action possible

        Returns:
            - batch of selected actions for `states`
        """
        raise NotImplementedError

    def predict_values(self, states: np.ndarray) -> np.ndarray:
        """
        Predicts value for each state.
        
        Params:
            - `states`: batch of states
        
        Returns:
            - value function of each state
        """
        raise NotImplementedError
    
    def train(self, states: np.ndarray, acitons: np.ndarray, returns: np.ndarray) -> None:
        raise NotImplementedError

