import torch
from hmc.utils import wrappers
from hmc.agents.nn.network import Network


class A2CAgent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, features: int, features_bound: int, actions: int):
        super().__init__()

        lr = 0.00005
        self.beta = 27
        self.actor = Network(features_bound, features, actions).to(self.device)
        self.actor.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.actor_loss = torch.nn.CrossEntropyLoss(reduction="none")

        self.critic = Network(features_bound, features, actions).to(self.device)
        self.critic.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_loss = torch.nn.MSELoss()

        self.info = {}

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

        actor_loss = torch.mean(
            (returns - v.detach()) * self.actor_loss(logits, actions)
            - self.beta
            * torch.distributions.Categorical(torch.softmax(logits, 1)).entropy()
        )

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

        entropy = (
            torch.distributions.Categorical(torch.softmax(logits, 1))
            .entropy()
            .mean()
            .item()
        )
        max_entropy = (
            torch.distributions.Categorical(torch.ones(12) / 12).entropy().item()
        )
        self.info["entropy rate"] = entropy / max_entropy
