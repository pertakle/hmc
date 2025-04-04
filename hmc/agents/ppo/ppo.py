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
        num_actions: int,
        device: torch.device,
    ) -> None:
        self.device = device

        self.num_actions = num_actions
        self.batch_size = args.batch_size
        self.gamma = args.gamma

        self.clip_grad_norm = 0.5#args.clip_grad_norm
        self.clip_epsilon = 0.25#args.clip_epsilon
        self.entropy_reg = 0.01#args.entropy_reg
        self.trace_lambda = 0.95
        self.train_epochs = 1

        self.actor = ResMLP(
            obs_size, obs_bound, args.n1, args.n2, args.nr, num_actions,
            noisy=False, norm=None, norm_last_only=False
        ).apply(torch_init_with_orthogonal_and_zeros).to(self.device)
        # TODO: init out *0.01
        with torch.no_grad():
            self.actor.out.weight.mul_(0.01)

        self.critic = ResMLP(
            obs_size, obs_bound, args.n1, args.n2, args.nr, 1,
            noisy=False, norm=None, norm_last_only=False
        ).apply(torch_init_with_orthogonal_and_zeros).to(self.device)

        # self.actor_opt = torch.optim.Adam(self.actor.parameters(), args.learning_rate, eps=1e-7)
        # self.critic_opt = torch.optim.Adam(self.critic.parameters(), args.learning_rate, eps=1e-7)
        self._params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.opt = torch.optim.Adam(self._params, args.learning_rate, eps=1e-7)
        self.critic_loss = torch.nn.MSELoss()

    def predict_action_probs(self, states: torch.Tensor) -> torch.Tensor:
        self.actor.eval()
        with torch.no_grad():
            logits = self.actor(states)
            probs = logits.softmax(-1)
            return probs

    def predict_values(self, states: torch.Tensor) -> torch.Tensor:
        self.critic.eval()
        with torch.no_grad():
            return self.critic(states)

    def train(self, episodes: tbuf.TorchReplayEpData) -> None:
        old_apr = self._action_probs(episodes)
        # print("train size", episodes.lengths.sum().item())
        for epoch in range(self.train_epochs):
            self.train_epoch(episodes, old_apr)

    def _action_probs(self, train_data: tbuf.TorchReplayEpData) -> torch.Tensor:
        probs = self.predict_action_probs(train_data.states.reshape(-1, train_data.states.shape[-1]))
        probs = probs[range(len(probs)), train_data.actions.flatten()]
        probs = probs.reshape(train_data.actions.shape)

        unroll_mask = (
            torch.arange(train_data.states.shape[1], device=train_data.get_device())[None, :]
            < train_data.lengths[:, None]
        )
        return probs[unroll_mask]

    
    def _prep_data(self, train_data: tbuf.TorchReplayEpData):
        # TODO: single predict
        state_shape = train_data.states.shape[-1]
        values = self.predict_values(train_data.states.reshape(-1, state_shape)).reshape(train_data.rewards.shape)
        next_values = self.predict_values(train_data.next_states.reshape(-1, state_shape)).reshape(train_data.rewards.shape)

        # GAE
        E, T, SG = train_data.states.shape
        sg = train_data.next_states.reshape(E, T, 2, -1)
        s = sg[:, :, 0]
        g = sg[:, :, 1]
        not_terminations = torch.all(s == g, dim=-1).logical_not()

        gae = torch.zeros_like(train_data.rewards)
        gae[:, -1] = train_data.rewards[:, -1] + self.gamma * not_terminations[:, -1] * next_values[:, -1] - values[:, -1]
        for t in range(train_data.lengths.max() - 2, -1, -1):
            not_dones_t = train_data.lengths - 1 != t
            gae[:, t] = (
                train_data.rewards[:, t] + self.gamma * not_terminations[:, t] + next_values[:, t]
                - values[:, t] + 
                self.gamma * self.trace_lambda * not_dones_t * gae[:, t+1]
            )

        # TODO: normalize here or below returns
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        returns = gae + values

        # Unroll states, action, a_probs, gae, returns
        unroll_mask = (
            torch.arange(train_data.states.shape[1], device=train_data.get_device())[None, :]
            < train_data.lengths[:, None]
        )
        return (
            train_data.states[unroll_mask],
            train_data.actions[unroll_mask],
            gae[unroll_mask],
            returns[unroll_mask]
        )

    def train_epoch(self, train_data: tbuf.TorchReplayEpData, old_actions_pr: torch.Tensor) -> None:
        states, actions, advantages, returns = self._prep_data(train_data)

        num_transitions = len(states)
        # indices = torch.randperm(num_transitions)
        indices = torch.randint(0, num_transitions, [num_transitions])
        for i in range(0, num_transitions - self.batch_size + 1, self.batch_size):
            b_indices = indices[i : i + self.batch_size]

            b_states = states[b_indices]
            b_actions = actions[b_indices]
            b_old_apr = old_actions_pr[b_indices]
            b_advantages = advantages[b_indices]
            b_returns = returns[b_indices]

            self.train_batch(b_states, b_actions, b_old_apr, b_advantages, b_returns)

    def train_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_apr: torch.Tensor,
        advantages: torch.Tensor,
        returns
    ) -> None:
        self.actor.train()
        self.critic.train()

        logits = self.actor(states)
        # print(logits.exp().max())
        probs = logits.softmax(-1)
        rho = probs[range(len(states)), actions] / (old_apr + 1e-8)
        # rho = rho / rho.max()
        # rho = torch.clip(rho, 0, 1.5)
        ppo_loss = -torch.minimum(
            rho * advantages,
            torch.clip(rho, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        ).mean()

        entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
        # print(entropy.item())
        critic_loss = self.critic_loss(self.critic(states).squeeze(), returns)
        
        loss = ppo_loss - entropy * self.entropy_reg + critic_loss
        # print(dict(dict(dict(self.critic.named_modules())[""].named_modules())["l1"].named_modules())["0"].weight.norm())

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._params, self.clip_grad_norm)
        with torch.no_grad():
            self.opt.step()

