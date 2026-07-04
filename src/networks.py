import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Independent, Normal, OneHotCategoricalStraightThrough
from torch.distributions.utils import probs_to_logits

from .utils import sequential_model_1d


class RecurrentModel(nn.Module):
    def __init__(self, recurrent_size, latent_size, action_size, config):
        super().__init__()
        self.config = config
        self.activation = getattr(nn, self.config["activation"])()

        self.linear = nn.Linear(latent_size + action_size, self.config["hidden_size"])
        self.recurrent = nn.GRUCell(self.config["hidden_size"], recurrent_size)

    def forward(self, recurrent_state, latent_state, action):
        return self.recurrent(
            self.activation(self.linear(torch.cat((latent_state, action), -1))),
            recurrent_state,
        )


class PriorNet(nn.Module):
    def __init__(self, input_size, latent_length, latent_classes, config):
        super().__init__()
        self.config = config
        self.latent_length = latent_length
        self.latent_classes = latent_classes
        self.latent_size = latent_length * latent_classes
        self.network = sequential_model_1d(
            input_size,
            [self.config["hidden_size"]] * self.config["num_layers"],
            self.latent_size,
            self.config["activation"],
        )

    def forward(self, x):
        raw_logits = self.network(x)

        probabilities = raw_logits.view(-1, self.latent_length, self.latent_classes).softmax(-1)
        uniform = torch.ones_like(probabilities) / self.latent_classes
        final_probabilities = (
            1 - self.config["uniform_mix"]
        ) * probabilities + self.config["uniform_mix"] * uniform
        logits = probs_to_logits(final_probabilities)

        sample = Independent(
            OneHotCategoricalStraightThrough(logits=logits), 1
        ).rsample()
        return sample.view(-1, self.latent_size), logits


class PosteriorNet(nn.Module):
    def __init__(self, input_size, latent_length, latent_classes, config):
        super().__init__()
        self.config = config
        self.latent_length = latent_length
        self.latent_classes = latent_classes
        self.latent_size = latent_length * latent_classes
        self.network = sequential_model_1d(
            input_size,
            [self.config["hidden_size"]] * self.config["num_layers"],
            self.latent_size,
            self.config["activation"],
        )

    def forward(self, x):
        raw_logits = self.network(x)

        probabilities = raw_logits.view(-1, self.latent_length, self.latent_classes).softmax(-1)
        uniform = torch.ones_like(probabilities) / self.latent_classes
        final_probabilities = (
            1 - self.config["uniform_mix"]
        ) * probabilities + self.config["uniform_mix"] * uniform
        logits = probs_to_logits(final_probabilities)

        sample = Independent(
            OneHotCategoricalStraightThrough(logits=logits), 1
        ).rsample()
        return sample.view(-1, self.latent_size), logits


class RewardModel(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.config = config
        self.network = sequential_model_1d(
            input_size,
            [self.config["hidden_size"]] * self.config["num_layers"],
            2,
            self.config["activation"],
        )

    def forward(self, x):
        mean, log_std = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(log_std).squeeze(-1))


class ContinueModel(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.config = config
        self.network = sequential_model_1d(
            input_size,
            [self.config["hidden_size"]] * self.config["num_layers"],
            1,
            self.config["activation"],
        )

    def forward(self, x):
        return Bernoulli(logits=self.network(x).squeeze(-1))


class EncoderConv(nn.Module):
    def __init__(self, input_shape, output_size, config):
        super().__init__()
        self.config = config
        activation = getattr(nn, self.config["activation"])()
        channels, height, width = input_shape
        self.output_size = output_size

        depth = self.config["depth"]
        kernel = self.config["kernel_size"]
        stride = self.config["stride"]
        padding = 1

        layers = []
        in_c = channels
        h, w = height, width
        num_layers = 0
        while h >= kernel and w >= kernel:
            out_c = depth * (2 ** num_layers)
            layers.append(nn.Conv2d(in_c, out_c, kernel, stride, padding=padding))
            layers.append(activation)
            in_c = out_c
            h = (h + 2 * padding - kernel) // stride + 1
            w = (w + 2 * padding - kernel) // stride + 1
            num_layers += 1

        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_c * h * w, output_size))
        layers.append(activation)

        self.convolutional_net = nn.Sequential(*layers)
        self._num_layers = num_layers

    def forward(self, x):
        return self.convolutional_net(x).view(-1, self.output_size)


class DecoderConv(nn.Module):
    def __init__(self, input_size, output_shape, config, encoder_num_layers=None):
        super().__init__()
        self.config = config
        self.channels, self.height, self.width = output_shape
        activation = getattr(nn, self.config["activation"])()

        depth = self.config["depth"]
        kernel = self.config["kernel_size"]
        stride = self.config["stride"]
        padding = 1

        # Determine number of decoder layers
        if encoder_num_layers is not None:
            num_layers = encoder_num_layers
        else:
            num_layers = config.get("num_conv_layers", 4)

        # Compute encoder output spatial size
        h_enc, w_enc = self.height, self.width
        for _ in range(num_layers):
            h_enc = (h_enc + 2 * padding - kernel) // stride + 1
            w_enc = (w_enc + 2 * padding - kernel) // stride + 1

        # Project to bottleneck
        final_channels = depth * (2 ** (num_layers - 1)) if num_layers > 0 else depth
        layers = [
            nn.Linear(input_size, final_channels * h_enc * w_enc),
            nn.Unflatten(1, (final_channels, h_enc, w_enc)),
        ]

        # Compute target spatial sizes at each decoder stage (reverse of encoder)
        h_sizes, w_sizes = [self.height], [self.width]
        for _ in range(num_layers):
            h_sizes.append((h_sizes[-1] + 2 * padding - kernel) // stride + 1)
            w_sizes.append((w_sizes[-1] + 2 * padding - kernel) // stride + 1)

        in_c = final_channels
        for i in range(num_layers):
            h_in = h_sizes[num_layers - i]
            h_targ = h_sizes[num_layers - i - 1]
            w_in = w_sizes[num_layers - i]
            w_targ = w_sizes[num_layers - i - 1]
            k_h = h_targ - (h_in - 1) * stride
            k_w = w_targ - (w_in - 1) * stride
            k_use = max(k_h, k_w)
            if k_h != k_w:
                # asymmetric kernel; use larger and add output_padding
                op_h = h_targ - ((h_in - 1) * stride + k_use)
                op_w = w_targ - ((w_in - 1) * stride + k_use)
            else:
                op_h = op_w = 0

            if i == num_layers - 1:
                out_c = self.channels
            else:
                out_c = depth * (2 ** (num_layers - i - 2))

            layers.append(nn.ConvTranspose2d(in_c, out_c, k_use, stride, output_padding=(op_h, op_w)))
            if i < num_layers - 1:
                layers.append(activation)
            in_c = out_c

        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
