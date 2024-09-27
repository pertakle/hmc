import torch
import numpy as np
from utils import wrappers
from nn.network import Network
import kostka.kostka_vek as kv
import kostka.kostka as ko

class HERCubeAgent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        self.policy_network = Network(2*6*3*3*6, 12).to(self.device)
        self.policy_network.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.policy_opt = torch.optim.Adam(self.policy_network.parameters(), lr=0.001)
        self.policy_loss = torch.nn.CrossEntropyLoss(reduction="none")

        self.val_network = Network(2*6*3*3*6, 1).to(self.device)
        self.val_network.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.val_opt = torch.optim.Adam(self.val_network.parameters(), lr=0.001)
        self.val_loss = torch.nn.MSELoss()

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

    def merge_states_and_goals(self, states: kv.KostkaVek, goals: kv.KostkaVek) -> np.ndarray:
        assert np.prod(states.shape) == np.prod(goals.shape)
        batch_size = states.shape[0]
        CUBE_LEN = 6*3*3
        state_goal = np.empty([batch_size, 2*CUBE_LEN], dtype=states.dtype)
        state_goal[:, :CUBE_LEN] = states.reshape(batch_size, CUBE_LEN)
        state_goal[:, CUBE_LEN:] = goals.reshape(-1, CUBE_LEN)
        return state_goal

    @wrappers.typed_torch_function(device, torch.float32)
    def predict_action_probs(self, x):
        self.policy_network.eval()
        with torch.no_grad():
            logits = self.policy_network(x)
            probs = torch.nn.functional.softmax(logits, -1)
            return probs

    @wrappers.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states, actions, returns):
        self.policy_network.train()
        self.policy_opt.zero_grad()

        predicted_baseline = self.val_network(states).squeeze()
        logits = self.policy_network(states)
        loss = (self.policy_loss(logits, actions) * (returns - predicted_baseline)).mean()

        loss.backward()
        self.policy_opt.step()

        self.val_network.train()
        self.val_opt.zero_grad()
        predicted_baseline = self.val_network(states).squeeze()
        val_loss = self.val_loss(predicted_baseline, returns)
        val_loss.backward()
        self.val_opt.step()

        self.info["loss"] = loss.item()

