import torch
from hmc.agents.nn.network import ResMLP
import argparse
from hmc.utils.wrappers import torch_init_with_orthogonal_and_zeros
import hmc.agents.rl_utils.torch_buffers as tbuf

class PPO:
    def __init__(
        self,
        args: argparse.Namespace,
        obs_bound: int,
        obs_size: int,
        actions: int,
        device: torch.device,
    ) -> None:
        self.device = device

        self.epochs = 1
        self.batch_size = args.batch_size

        self.actions = actions
        self.clip_grad_norm = 0.5#args.clip_grad_norm
        self.clip_epsilon = 0.25#args.clip_epsilon
        self.entropy_reg = 0.1#args.entropy_reg

        self.actor = ResMLP(
            obs_size, obs_bound, 1024, 1024, 1, actions,
            noisy=False, norm=None, norm_last_only=False
        ).apply(torch_init_with_orthogonal_and_zeros).to(self.device)
        self.critic = ResMLP(
            obs_size, obs_bound, 1024, 1024, 1, 1,
            noisy=False, norm=None, norm_last_only=False
        ).apply(torch_init_with_orthogonal_and_zeros).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), args.learning_rate, eps=1e-7)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), args.learning_rate, eps=1e-7)
        self.critic_loss = torch.nn.MSELoss()

    def predict_action_probs(self, states: torch.Tensor) -> torch.Tensor:
        self.actor.eval()
        with torch.no_grad():
            logits = self.actor(states)
            return logits.softmax(-1)

    def predict_values(self, states: torch.Tensor) -> torch.Tensor:
        self.critic.eval()
        with torch.no_grad():
            return self.critic(states)

    def train(self, episodes: tbuf.TorchReplayEpData) -> None:
        for epoch in range(self.train_epochs):
            self.train_epoch(episodes)

    def train_epoch(self, train_data: tbuf.TorchReplayEpData) -> None:
        data_length = ...
        advantages = ...
        rhos = ...
        indices = torch.randperm(...)
        for i in range(0, data_length - self.batch_size + 1, self.batch_size):
            batch_data = ... # train_data[i : i + self.batch_size]
            self.train_actor_batch(...)
            self.train_critic_batch(...)

    def train_critic_batch(self, states: torch.Tensor, returns: torch.Tensor) -> None:
        self.critic.train()
        predictions = self.critic(states)
        loss = self.critic_loss(predictions, returns)
        self.critic_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad_norm)
        with torch.no_grad():
            self.critic_opt.step()


    def train_actor_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_apr: torch.Tensor,
        advantages: torch.Tensor
    ) -> None:
        self.actor.train()
        logits = self.actor(states)
        probs = logits.softmax(-1)
        rho = probs[range(len(states)), actions] / old_apr
        ppo_loss = -torch.minimum(
            rho * advantages,
            torch.clip(rho, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        ).mean()

        entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
        loss = ppo_loss - entropy * self.entropy_reg

        self.actor_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
        with torch.no_grad():
            self.actor_opt.step()

