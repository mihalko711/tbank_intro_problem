import os

import torch
import torch.nn as nn
from torch.distributions import Independent, OneHotCategoricalStraightThrough, kl_divergence

from .buffer import ReplayBuffer
from .networks import (
    ContinueModel,
    DecoderConv,
    EncoderConv,
    PosteriorNet,
    PriorNet,
    RecurrentModel,
    RewardModel,
)
from .utils import symlog, two_hot_encode


class RSSMWorldModel:
    def __init__(self, observation_shape, action_size, config, device):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.config = config
        self.device = device

        self.recurrent_size = config["recurrent_size"]
        self.latent_size = config["latent_length"] * config["latent_classes"]
        self.full_state_size = self.recurrent_size + self.latent_size

        self.encoder = EncoderConv(
            observation_shape, config["encoded_obs_size"], config["encoder"]
        ).to(device)
        self.decoder = DecoderConv(
            self.full_state_size, observation_shape, config["decoder"],
            encoder_num_layers=self.encoder._num_layers
        ).to(device)
        self.recurrent_model = RecurrentModel(
            self.recurrent_size, self.latent_size, action_size, config["recurrent_model"]
        ).to(device)
        self.prior_net = PriorNet(
            self.recurrent_size, config["latent_length"], config["latent_classes"], config["prior_net"]
        ).to(device)
        self.posterior_net = PosteriorNet(
            self.recurrent_size + config["encoded_obs_size"],
            config["latent_length"],
            config["latent_classes"],
            config["posterior_net"],
        ).to(device)
        self.reward_predictor = RewardModel(self.full_state_size, config["reward"]).to(device)
        if config.get("use_continuation_prediction", False):
            self.continue_predictor = ContinueModel(self.full_state_size, config["continuation"]).to(device)

        self.buffer = ReplayBuffer(observation_shape, action_size, config["buffer"]["capacity"], device)

        num_reward_bins = config["reward"].get("num_bins", 21)
        self.reward_bins = torch.linspace(-0.5, 1.5, num_reward_bins, device=device)

        self.wm_parameters = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.recurrent_model.parameters())
            + list(self.prior_net.parameters())
            + list(self.posterior_net.parameters())
            + list(self.reward_predictor.parameters())
        )
        if config.get("use_continuation_prediction", False):
            self.wm_parameters += list(self.continue_predictor.parameters())

        self.wm_optimizer = torch.optim.Adam(self.wm_parameters, lr=config["world_model_lr"])

        self.total_episodes = 0
        self.total_env_steps = 0
        self.total_gradient_steps = 0

    # ── Training ──────────────────────────────────────────────────────────

    def train_step(self, data):
        observations = data["observations"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]

        batch_size = observations.shape[0]
        batch_length = observations.shape[1]

        is_first = data.get("is_first")

        encoded_observations = (
            self.encoder(observations.view(-1, *self.observation_shape))
            .view(batch_size, batch_length, -1)
        )

        prev_recurrent = torch.zeros(batch_size, self.recurrent_size, device=self.device)
        prev_latent = torch.zeros(batch_size, self.latent_size, device=self.device)

        recurrent_states, prior_logits, posteriors, posterior_logits = [], [], [], []
        for t in range(1, batch_length):
            if is_first is not None:
                reset = is_first[:, t]
                prev_recurrent = torch.where(reset, torch.zeros_like(prev_recurrent), prev_recurrent)
                prev_latent = torch.where(reset, torch.zeros_like(prev_latent), prev_latent)
            recurrent = self.recurrent_model(prev_recurrent, prev_latent, actions[:, t - 1])
            _, prior_logit = self.prior_net(recurrent)
            posterior, posterior_logit = self.posterior_net(
                torch.cat((recurrent, encoded_observations[:, t]), -1)
            )

            recurrent_states.append(recurrent)
            prior_logits.append(prior_logit)
            posteriors.append(posterior)
            posterior_logits.append(posterior_logit)

            prev_recurrent = recurrent
            prev_latent = posterior

        recurrent_states = torch.stack(recurrent_states, dim=1)
        prior_logits = torch.stack(prior_logits, dim=1)
        posteriors = torch.stack(posteriors, dim=1)
        posterior_logits = torch.stack(posterior_logits, dim=1)
        full_states = torch.cat((recurrent_states, posteriors), dim=-1)

        # ── reconstruction loss (MSE on pixel values) ──
        recon_means = (
            self.decoder(full_states.view(-1, self.full_state_size))
            .view(batch_size, batch_length - 1, *self.observation_shape)
        )
        recon_loss = nn.functional.mse_loss(recon_means, observations[:, 1:])

        with torch.no_grad():
            prior_latent, _ = self.prior_net(recurrent_states.view(-1, self.recurrent_size))
            prior_full = torch.cat((recurrent_states.view(-1, self.recurrent_size), prior_latent), -1)
            prior_recon = self.decoder(prior_full).view(batch_size, batch_length - 1, *self.observation_shape)
            prior_recon_loss = nn.functional.mse_loss(prior_recon, observations[:, 1:])

        # ── reward loss (symlog + two-hot) ──
        reward_logits = self.reward_predictor(full_states)
        reward_sym = symlog(rewards[:, 1:])
        reward_target = two_hot_encode(reward_sym, self.reward_bins)
        reward_loss = -(reward_target * nn.functional.log_softmax(reward_logits, -1)).sum(-1).mean()

        # ── KL loss ──
        prior_dist = Independent(OneHotCategoricalStraightThrough(logits=prior_logits), 1)
        prior_dist_sg = Independent(OneHotCategoricalStraightThrough(logits=prior_logits.detach()), 1)
        posterior_dist = Independent(OneHotCategoricalStraightThrough(logits=posterior_logits), 1)
        posterior_dist_sg = Independent(OneHotCategoricalStraightThrough(logits=posterior_logits.detach()), 1)

        prior_loss = kl_divergence(posterior_dist_sg, prior_dist)
        posterior_loss = kl_divergence(posterior_dist, prior_dist_sg)
        free_nats = torch.full_like(prior_loss, self.config["free_nats"])

        kl_active = (prior_loss > free_nats).float().mean()

        beta_prior = self.config["beta_prior"]
        beta_posterior = self.config["beta_posterior"]

        kl_raw = (beta_prior * prior_loss + beta_posterior * posterior_loss).mean()

        prior_loss = beta_prior * torch.maximum(prior_loss, free_nats)
        posterior_loss = beta_posterior * torch.maximum(posterior_loss, free_nats)
        kl_loss = (prior_loss + posterior_loss).mean()

        wm_loss = recon_loss + reward_loss + kl_loss

        if self.config.get("use_continuation_prediction", False):
            continue_dist = self.continue_predictor(full_states)
            continue_loss = nn.functional.binary_cross_entropy(
                continue_dist.probs, (1 - dones[:, 1:]).reshape(-1)
            )
            wm_loss += continue_loss

        self.wm_optimizer.zero_grad()
        wm_loss.backward()
        nn.utils.clip_grad_norm_(
            self.wm_parameters,
            self.config["gradient_clip"],
            norm_type=self.config.get("gradient_norm_type", 2),
        )
        self.wm_optimizer.step()

        kl_shift = (beta_prior + beta_posterior) * self.config["free_nats"]
        metrics = {
            "wm_loss": wm_loss.item() - kl_shift,
            "recon_loss": recon_loss.item(),
            "prior_recon_loss": prior_recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "kl_loss": kl_loss.item() - kl_shift,
            "kl_raw": kl_raw.item(),
            "kl_active": kl_active.item(),
        }
        return full_states.view(-1, self.full_state_size).detach(), metrics

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def reset_state(self):
        return (
            torch.zeros(1, self.recurrent_size, device=self.device),
            torch.zeros(1, self.latent_size, device=self.device),
        )

    @torch.no_grad()
    def encode_step(self, recurrent_state, latent_state, action, observation):
        encoded_obs = self.encoder(
            torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        )
        recurrent_state = self.recurrent_model(recurrent_state, latent_state, action)
        latent_state, _ = self.posterior_net(
            torch.cat((recurrent_state, encoded_obs.view(1, -1)), -1)
        )
        return recurrent_state, latent_state

    @torch.no_grad()
    def imagine_rollouts(self, start_recurrent, start_latent, candidate_actions):
        num_candidates, horizon = candidate_actions.shape[:2]

        recurrent = start_recurrent.repeat(num_candidates, 1)
        latent = start_latent.repeat(num_candidates, 1)
        full_states = []

        for t in range(horizon):
            recurrent = self.recurrent_model(recurrent, latent, candidate_actions[:, t])
            latent, _ = self.prior_net(recurrent)
            full_states.append(torch.cat((recurrent, latent), -1))

        return torch.stack(full_states, dim=1)  # (num_candidates, horizon, full_state_size)

    @torch.no_grad()
    def rollout_prior(self, start_recurrent, start_latent, action_fn, horizon):
        recurrent, latent = start_recurrent, start_latent
        full_states = []

        for _ in range(horizon):
            action = action_fn(torch.cat((recurrent, latent), -1))
            recurrent = self.recurrent_model(recurrent, latent, action)
            latent, _ = self.prior_net(recurrent)
            full_states.append(torch.cat((recurrent, latent), -1))

        return torch.stack(full_states, dim=1)

    # ── Checkpoint ────────────────────────────────────────────────────────

    def save_checkpoint(self, path):
        if not path.endswith(".pth"):
            path += ".pth"

        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "recurrent_model": self.recurrent_model.state_dict(),
            "prior_net": self.prior_net.state_dict(),
            "posterior_net": self.posterior_net.state_dict(),
            "reward_predictor": self.reward_predictor.state_dict(),
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "total_gradient_steps": self.total_gradient_steps,
        }
        if hasattr(self, "continue_predictor"):
            checkpoint["continue_predictor"] = self.continue_predictor.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        if not path.endswith(".pth"):
            path += ".pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.recurrent_model.load_state_dict(checkpoint["recurrent_model"])
        self.prior_net.load_state_dict(checkpoint["prior_net"])
        self.posterior_net.load_state_dict(checkpoint["posterior_net"])
        self.reward_predictor.load_state_dict(checkpoint["reward_predictor"])
        try:
            self.wm_optimizer.load_state_dict(checkpoint["wm_optimizer"])
        except (ValueError, KeyError):
            pass
        self.total_gradient_steps = checkpoint["total_gradient_steps"]
        if hasattr(self, "continue_predictor") and "continue_predictor" in checkpoint:
            self.continue_predictor.load_state_dict(checkpoint["continue_predictor"])
