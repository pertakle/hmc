from deepercube.nn.her_cube_agent import HERCubeAgent
from deepercube.nn.network import Network
from deepercube.utils import wrappers
import numpy as np
import copy
import torch


class DQNAgent(HERCubeAgent):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self) -> None:
        super().__init__()
        self.model = Network(6*3*3*6, 12).to(self.device)
        self.model.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.target_network = copy.deepcopy(self.model).to(self.device)
        self.rng = torch.Generator()

        self.opt = torch.optim.Adam(self.model.parameters())
        self.loss = torch.nn.MSELoss()

    @wrappers.typed_torch_function(device, torch.float32, float)
    def predict_move(self, stategoals: torch.Tensor, greedy: float) -> torch.Tensor: # type: ignore
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(stategoals)
        if greedy:
            return torch.argmax(q_values, -1)
        return torch.randint(0, 13, size=stategoals.shape[:1], generator=self.rng)

    
    def train(self, episodes: np.ndarray) -> None:
        pass



    def copy_weights(self) -> None:
        self.model.load_state_dict(self.target_network.state_dict())
