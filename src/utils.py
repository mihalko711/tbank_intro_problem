import os
import random

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def sequential_model_1d(
    input_size,
    hidden_sizes,
    output_size,
    activation_function="Tanh",
    finish_with_activation=False,
):
    activation = getattr(nn, activation_function)()
    layers = []
    current_input_size = input_size

    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(current_input_size, hidden_size))
        layers.append(activation)
        current_input_size = hidden_size

    layers.append(nn.Linear(current_input_size, output_size))
    if finish_with_activation:
        layers.append(activation)

    return nn.Sequential(*layers)


def ensure_parent_folders(*paths):
    for path in paths:
        parent_folder = os.path.dirname(path)
        if parent_folder and not os.path.exists(parent_folder):
            os.makedirs(parent_folder, exist_ok=True)


class Moments(nn.Module):
    def __init__(
        self, device, decay=0.99, min_=1, percentile_low=0.05, percentile_high=0.95
    ):
        super().__init__()
        self._decay = decay
        self._min = torch.tensor(min_)
        self._percentile_low = percentile_low
        self._percentile_high = percentile_high
        self.register_buffer("low", torch.zeros((), dtype=torch.float32, device=device))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.detach()
        low = torch.quantile(x, self._percentile_low)
        high = torch.quantile(x, self._percentile_high)
        self.low = self._decay * self.low + (1 - self._decay) * low
        self.high = self._decay * self.high + (1 - self._decay) * high
        inverse_scale = torch.max(self._min, self.high - self.low)
        return self.low.detach(), inverse_scale.detach()
