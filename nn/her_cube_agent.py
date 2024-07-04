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
        minus_tahy = (tahy<0).type(torch.int64)
        indexy = tahy.type(torch.int64) - minus_tahy * 6
        indexy -= 2*indexy*minus_tahy
        indexy -= 1
        return indexy

    def indexy_na_tahy(self, indexy: np.ndarray) -> np.ndarray:
        minus_tahy = (indexy > 5).astype(np.int64)
        tahy = indexy + 1
        tahy -= 2*indexy*minus_tahy
        tahy += 4*minus_tahy
        return tahy

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

