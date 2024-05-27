import torch
import wrappers

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

        self.out = torch.nn.Linear(1000, 27)
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
        

    @wrappers.typed_torch_function(device, torch.float32)
    def predict(self, x):
        self.network.eval()
        with torch.no_grad():
            logits = self.network(x)
        return torch.argmax(logits, 1)

    @wrappers.typed_torch_function(device, torch.float32, torch.long)
    def train(self, x, t):
        self.network.train()
        self.opt.zero_grad()
        y = self.network(x)
        loss = self.loss(y, t)
        loss.backward()
        with torch.no_grad():
            self.opt.step()

