import copy
import torch
import numpy as np
from deepercube.utils import wrappers
from deepercube.nn.network import Network
import deepercube.kostka.kostka_vek as kv
from typing import Dict

class HERCubeAgent:

    def __init__(self) -> None:
        self.info = {}

    def predict_move(self, stategals: np.ndarray, greedy: float) -> np.ndarray:
        raise NotImplementedError
    
    def train(self, episodes: np.ndarray) -> None:
        pass


class A2CCubeAgent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        lr = 0.00005
        self.beta = 27
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

        actor_loss = torch.mean((returns - v.detach()) * self.actor_loss(logits, actions) - \
                                self.beta * torch.distributions.Categorical(torch.softmax(logits, 1)).entropy())

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

        entropy = torch.distributions.Categorical(torch.softmax(logits, 1)).entropy().mean().item()
        max_entropy = torch.distributions.Categorical(torch.ones(12)/12).entropy().item()
        self.info["entropy rate"] = entropy / max_entropy


class SACHERCubeAgent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):

        class Actor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.network = Network(2*6*3*3*6, 12)
                self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
            
            def forward(self, inputs: torch.Tensor, sample: bool):
                actions_logprobs = self.network(inputs)
                actions_distributions = torch.distributions.Categorical(logits=actions_logprobs)
                if sample:
                    actions = actions_distributions.rsample()
                else:
                    actions = torch.argmax(actions_logprobs, -1)
                log_prob = actions_distributions.log_prob(actions).mean()
                alpha = torch.exp(self.log_alpha)
                return actions, log_prob, alpha
        
        def new_critic() -> Network:
            return Network(2*6*3*3*6, 12)

        lr = 0.00005
        self.target_tau = 0.005
        self.target_entropy = 2

        self.actor = Actor().to(self.device)
        self.critic1 = new_critic().to(self.device)
        self.critic2 = new_critic().to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr)

        self.mse_loss = torch.nn.MSELoss()

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

    # Method for performing exponential moving average of weights of the given two modules.
    def update_parameters_by_ema(self, source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
        with torch.no_grad():
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1 - tau)
                target_param.data.add_(tau * param.data)

    @wrappers.typed_torch_function(device, torch.float32)
    def predict_action_probs(self, x):
        self.actor.eval()
        with torch.no_grad():
            logits = self.actor(x)
        probs = torch.nn.functional.softmax(logits, -1)
        return probs

    @wrappers.typed_torch_function(device, torch.float32)
    def predict_values(self, x):
        return torch.ones([x.shape[0], 1])
        self.critic.eval()
        with torch.no_grad():
            predictions = self.critic(x)
        return predictions

    @wrappers.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states, actions, returns):
        self.actor.train()
        self.actor_opt.zero_grad()
        actor_actions, log_prob, alpha = self.actor(states, sample=True)
        print(actor_actions.shape)
        exit()

        crit1 = self.critic1(states, dim=-1)
        crit2 = self.critic2(states, dim=-1)

        actor_loss = ((alpha.detach() * log_prob) - torch.min(crit1, crit2)).mean()
        actor_loss.backward()

        alpha_loss = alpha * (-log_prob.detach() - self.target_entropy).detach()
        alpha_loss.backward()

        with torch.no_grad():
            self._actor_optimizer.step()
        
        # - the critics using MSE loss.
        self._critic1.train()
        self._critic1_optimizer.zero_grad()
        values1 = self._critic1(torch.cat([states, actions], dim=-1)).squeeze()
        loss1 = self._mse_loss(values1, returns)
        loss1.backward()
        with torch.no_grad():
            self._critic1_optimizer.step()
            
        self._critic2.train()
        self._critic2_optimizer.zero_grad()
        values2 = self._critic2(torch.cat([states, actions], dim=-1)).squeeze()
        loss2 = self._mse_loss(values2, returns)
        loss2.backward()
        with torch.no_grad():
            self._critic2_optimizer.step()
        #
        # Finally, update the two target critic networks exponential moving
        # average with weight `args.target_tau`, using `self.update_parameters_by_ema`.
        self.update_parameters_by_ema(self._critic1, self._target_critic1, self.target_tau)
        self.update_parameters_by_ema(self._critic2, self._target_critic2, self.target_tau)


        self.info["actor_loss"] = actor_loss.item()
        self.info["critic_loss"] = critic_loss.item()

        entropy = torch.distributions.Categorical(torch.softmax(logits, 1)).entropy().mean().item()
        max_entropy = torch.distributions.Categorical(torch.ones(12)/12).entropy().item()
        self.info["entropy rate"] = entropy / max_entropy

