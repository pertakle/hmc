import torch
import wrappers
import numpy as np

def res_block():
    return torch.nn.Sequential(
        torch.nn.Linear(1000, 1000, bias=False),
        torch.nn.BatchNorm1d(1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 1000),
        torch.nn.BatchNorm1d(1000)
    )
# https://github.com/forestagostinelli/DeepCubeA/blob/master/utils/pytorch_models.py

class Network(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.relu = torch.nn.ReLU()

        self.l1 = torch.nn.Linear(9*6*6, 5000, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(5000)

        self.l2 = torch.nn.Linear(5000, 1000, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(1000)

        self.res_blocks = torch.nn.ParameterList([res_block() for _ in range(4)])

        self.out = torch.nn.Linear(1000, 12)
        self.apply(wrappers.torch_init_with_xavier_and_zeros)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = torch.nn.functional.one_hot(x.reshape([B, -1]).type(torch.long), 6).reshape([B, -1]).type(torch.float32)
        hidden = self.l1(x)
        hidden = self.bn1(hidden)
        hidden = self.relu(hidden)

        hidden = self.l2(hidden)
        hidden = self.bn2(hidden)
        hidden = self.relu(hidden)

        for res_block in self.res_blocks:
            skip = hidden
            res = res_block(hidden)
            hidden = res + skip
            hidden = self.relu(hidden)

        out = self.out(hidden)
        return out


class Agent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        self.network = Network().to(self.device)
        self.opt = torch.optim.Adam(self.network.parameters())
        self.loss = torch.nn.CrossEntropyLoss()
        #self.values = torch.arange(27, dtype=torch.float32, device=self.device) # 0...26 = biží číslo
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

    @wrappers.typed_torch_function(device, torch.float32, torch.float32)
    def train(self, x, tahy):
        t = self.tahy_na_indexy(tahy)
        self.network.train()
        self.opt.zero_grad()

        y = self.network(x).squeeze()
        loss = self.loss(y, t)

        #entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
        #entropy_reg = 0.1 * entropy

        loss.backward()
        with torch.no_grad():
            self.opt.step()

        self.info["loss"] = loss.item()

