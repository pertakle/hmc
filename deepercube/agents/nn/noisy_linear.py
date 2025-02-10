import torch
import torch.nn.functional


class NoisyLinear(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.device = device
        self.sigma_init = sigma_init

        self.mu_w = torch.nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
        )
        self.sigma_w = torch.nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
        )
        self.register_buffer(
            "epsilon_w",
            torch.empty(
                out_features,
                in_features,
                dtype=dtype,
                device=device,
                requires_grad=False,
            ),
        )

        if bias:
            self.mu_b = torch.nn.Parameter(
                torch.empty(
                    out_features, dtype=dtype, device=device, requires_grad=True
                )
            )
            self.sigma_b = torch.nn.Parameter(
                torch.empty(
                    out_features, dtype=dtype, device=device, requires_grad=True
                )
            )
            self.register_buffer(
                "epsilon_b",
                torch.empty(
                    out_features, dtype=dtype, device=device, requires_grad=False
                ),
            )
        else:
            self.register_buffer("mu_b", None)
            self.register_buffer("sigma_b", None)
            self.register_buffer("epsilon_b", None)

        self.reset_parameters()
        self.reset_noise()

    def _get_weights(self) -> torch.Tensor:
        weights = self.mu_w
        if self.training:
            weights = weights + self.sigma_w * self.epsilon_w
        return weights

    def _get_bias(self) -> torch.Tensor | None:
        if not self.has_bias:
            return None
        bias = self.mu_b
        if self.training:
            bias = bias + self.sigma_b * self.epsilon_b
        return bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.reset_noise()
        return torch.nn.functional.linear(x, self._get_weights(), self._get_bias())

    def reset_parameters(self) -> None:
        mu_range = 1 / self.in_features**0.5
        sigma_val = self.sigma_init / self.in_features**0.5
        self.mu_w.data.uniform_(-mu_range, mu_range)
        self.sigma_w.data.fill_(sigma_val)
        if self.has_bias:
            self.mu_b.data.uniform_(-mu_range, mu_range)
            self.sigma_b.data.fill_(sigma_val)

    def _get_noise(self, size: int) -> torch.Tensor:
        noise = torch.randn(size, device=self.mu_w.device)
        return torch.sign(noise).mul_(torch.sqrt(torch.abs(noise)))

    def reset_noise(self) -> None:
        noise_in = self._get_noise(self.in_features)
        noise_out = self._get_noise(self.out_features)
        self.epsilon_w.copy_(torch.outer(noise_out, noise_in))
        if self.has_bias:
            self.epsilon_b.copy_(noise_out)

