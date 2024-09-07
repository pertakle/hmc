import torch
import numpy as np
from utils import wrappers
from nn.network import Network

class HERCubeAgent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        self.network = Network(9*6*6*2, 12).to(self.device)
        self.opt = torch.optim.Adam(self.network.parameters())
        self.loss = torch.nn.CrossEntropyLoss()
        self.info = {}

    def tahy_na_indexy(self, tahy: torch.Tensor) -> torch.Tensor:
        """Prevede tahy {-6,..,-1,1,..6} na indexy {0..12}"""
        minus_tahy = (tahy<0).type(torch.int64)
        indexy = tahy.type(torch.int64) - minus_tahy * 6
        indexy -= 2*indexy*minus_tahy
        indexy -= 1
        return indexy

    def indexy_na_tahy(self, indexy: np.ndarray) -> np.ndarray:
        """Prevede indexy {0..12} na tahy {-6,..,-1,1,..6}"""
        minus_tahy = (indexy > 5).astype(np.int64)
        tahy = indexy + 1
        tahy -= 2*indexy*minus_tahy
        tahy += 4*minus_tahy
        return tahy

    def merge_states_and_goals(self, states: np.ndarray, goals: np.ndarray) -> np.ndarray:
        batch_size = states.shape[0]
        CUBE_LEN = 6*3*3
        state_goal = np.empty([batch_size, 2*CUBE_LEN], dtype=states.dtype)
        state_goal[:, :CUBE_LEN] = states.reshape(batch_size, CUBE_LEN)
        state_goal[:, CUBE_LEN:] = goals.reshape(-1, CUBE_LEN)
        return state_goal

    @wrappers.typed_torch_function(device, torch.float32)
    def predict(self, x):
        self.network.eval()
        with torch.no_grad():
            logits = self.network(x)
            probs = torch.nn.functional.softmax(logits, -1)
            return probs

    @wrappers.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states, actions, returns):
        self.network.train()
        self.opt.zero_grad()

        logits = self.network(states)
        #l = self.loss(logits, actions)
        #print(f"{self.loss}({logits.shape, logits.dtype}, {actions.shape, actions.dtype}) = {l.shape, l.dtype}")
        #exit()
        loss = (self.loss(logits, actions) * returns).mean()

        loss.backward()
        with torch.no_grad():
            self.opt.step()

        self.info["loss"] = loss.item()
