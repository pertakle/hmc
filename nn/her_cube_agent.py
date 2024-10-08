import torch
import numpy as np
from utils import wrappers
from nn.network import Network
import kostka.kostka_vek as kv
import kostka.kostka as ko

class HERCubeAgent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        lr = 0.0001
        self.actor = Network(2*6*3*3*6, 12).to(self.device)
        self.actor.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.actor_loss = torch.nn.CrossEntropyLoss(reduction="none")

        self.critic = Network(2*6*3*3*6, 1).to(self.device)
        self.critic.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_loss = torch.nn.MSELoss()

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
        self.actor.eval()
        with torch.no_grad():
            logits = self.actor(x)
        probs = torch.nn.functional.softmax(logits, -1)
        return probs

    @wrappers.typed_torch_function(device, torch.float32)
    def predict_values(self, x):
        self.critic.eval()
        with torch.no_grad():
            predictions = self.critic(x)
        return predictions

    @wrappers.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states, actions, returns):
        self.actor.train()
        self.actor_opt.zero_grad()

        logits = self.actor(states)
        v = self.critic(states).squeeze()

        beta = 5
        actor_loss = torch.mean((returns - v.detach()) * self.actor_loss(logits, actions) - \
                                beta * torch.distributions.Categorical(torch.softmax(logits, 1)).entropy())


        actor_loss.backward()
        with torch.no_grad():
            self.actor_opt.step()

        self.critic.train()
        self.critic_opt.zero_grad()
        critic_loss = self.critic_loss(v, returns)
        critic_loss.backward()
        with torch.no_grad():
            self.critic_opt.step()

        self.info["actor_loss"] = actor_loss.item()
        self.info["critic_loss"] = critic_loss.item()


