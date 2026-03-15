import torch
from hmc.agents.nn.network import ResMLP
from hmc.utils.wrappers import torch_init_with_orthogonal_and_zeros
import argparse


class CRL:

    def __init__(
        self,
        args: argparse.Namespace,
        obs_bound: int,
        obs_size: int,
        embedding_size: int,
        device: torch.device,
    ) -> None:
        self.embedding_model = ResMLP(
            obs_size, obs_bound, args.n1, args.n2, args.nr, embedding_size,
            noisy=False, norm="layer", norm_last_only=False, activation="silu"
        ).apply(torch_init_with_orthogonal_and_zeros).to(device)

        self.opt = torch.optim.Adam(self.embedding_model.parameters(), args.learning_rate)
        # self.opt = torch.optim.AdamW(self.model.parameters(), args.learning_rate, weight_decay=0.001)

    def predict(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        self.embedding_model.eval()
        with torch.no_grad():
            s = self.embedding_model(state)
            g = self.embedding_model(goal)
            return torch.norm(s - g)

    def train(self, true_states: torch.Tensor, true_goals: torch.Tensor, false_goals: torch.Tensor) -> None:
        self.embedding_model.train()

        true_s_emb = self.embedding_model(true_states).squeeze()
        true_g_emb = self.embedding_model(true_goals).squeeze()
        false_g_emb = self.embedding_model(false_goals).squeeze()

        true_l2 = torch.norm(true_s_emb - true_g_emb)
        false_l2 = torch.norm(true_s_emb[..., None] - false_g_emb)

        # loss = -log(e^ltrue / sum_j(e^lfalse))
        #      = -(log(e^ltrue) - log(sum_j(e^lfalse)))
        #      = -log(e^ltrue) + log(sum_j(e^lfalse))
        #      = -ltrue + log sum_j e^lfalse
        #      = log(sum_j(e^lfalse)) - ltrue
        loss = torch.logsumexp(false_l2, -1) - true_l2
        loss = loss.mean()

        self.opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            self.opt.step()
            
