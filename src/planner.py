import warnings

import torch
import torch.nn.functional as F
import open_clip

from .utils import symexp, decode_two_hot


class UniformCandidates:
    def __init__(self, num_candidates, horizon, action_size, device):
        self.num_candidates = num_candidates
        self.horizon = horizon
        self.action_size = action_size
        self.device = device

    @torch.no_grad()
    def sample(self):
        actions = torch.randint(
            0, self.action_size, (self.num_candidates, self.horizon), device=self.device
        )
        return F.one_hot(actions, self.action_size).float()


class HeuristicCandidates:
    def __init__(self, num_candidates, horizon, action_size, device,
                 forward_action=2, forward_prob=0.8, turn_prob=0.1):
        assert forward_prob + 2 * turn_prob <= 1.0
        self.num_candidates = num_candidates
        self.horizon = horizon
        self.action_size = action_size
        self.device = device
        self.forward_action = forward_action
        self.forward_prob = forward_prob
        self.turn_prob = turn_prob

    @torch.no_grad()
    def sample(self):
        probs = torch.rand(self.num_candidates, self.horizon, device=self.device)
        actions = torch.where(
            probs < self.turn_prob,
            torch.zeros_like(probs, dtype=torch.long),
            torch.where(
                probs < 2 * self.turn_prob,
                torch.ones_like(probs, dtype=torch.long),
                torch.full_like(probs, self.forward_action, dtype=torch.long),
            ),
        )
        return F.one_hot(actions, self.action_size).float()


class CLIPScorer:
    def __init__(self, device, rssm, goal_text="a green goal square",
                 model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.device = device
        self.rssm = rssm
        self.model_name = model_name
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.model.eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)

        self.normalize_mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=device
        ).view(1, 3, 1, 1)
        self.normalize_std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=device
        ).view(1, 3, 1, 1)

        self.set_goal(goal_text)

    def set_goal(self, goal_text):
        text = self._tokenizer([goal_text])
        with torch.no_grad():
            text_emb = self.model.encode_text(text.to(self.device))
        self.text_embedding = F.normalize(text_emb, dim=-1).squeeze(0)
        self.goal_text = goal_text

    @torch.no_grad()
    def score_rollouts(self, trajectories, gamma=0.99, mode="discounted_sum"):
        num_candidates, horizon = trajectories.shape[:2]
        flat = trajectories.view(-1, self.rssm.full_state_size)
        decoded = self.rssm.decoder(flat)
        decoded = symexp(decoded).clamp(0, 1)
        decoded = decoded.view(num_candidates, horizon, *self.rssm.observation_shape)

        B = num_candidates * horizon
        images = decoded.view(B, *self.rssm.observation_shape)
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        images = (images - self.normalize_mean) / self.normalize_std

        image_embs = self.model.encode_image(images)
        image_embs = F.normalize(image_embs, dim=-1)
        image_embs = image_embs.view(num_candidates, horizon, -1)

        sims = (image_embs * self.text_embedding.unsqueeze(0).unsqueeze(0)).sum(-1)

        if mode == "discounted_sum":
            discounts = gamma ** torch.arange(horizon, device=self.device)
            return (sims * discounts.unsqueeze(0)).sum(-1)
        elif mode == "max":
            return sims.max(dim=-1).values
        elif mode == "max_discounted":
            discounts = gamma ** torch.arange(horizon, device=self.device)
            return (sims * discounts.unsqueeze(0)).max(dim=-1).values
        else:
            raise ValueError(f"unknown scoring mode: {mode}")


class CEMCandidates:
    def __init__(self, num_candidates, horizon, action_size, device,
                 num_iters=3, elite_frac=0.1, alpha=0.5, init_probs=None):
        self.num_candidates = num_candidates
        self.horizon = horizon
        self.action_size = action_size
        self.device = device
        self.num_iters = num_iters
        self.num_elite = max(1, int(num_candidates * elite_frac))
        self.alpha = alpha
        if init_probs is None:
            init_probs = torch.ones(action_size, device=device) / action_size
        self.init_probs = init_probs

    @torch.no_grad()
    def plan(self, score_fn):
        probs = self.init_probs.unsqueeze(0).repeat(self.horizon, 1).clone()

        best_actions, best_score = None, -float("inf")
        for _ in range(self.num_iters):
            dist = torch.distributions.Categorical(probs=probs)
            samples = dist.sample((self.num_candidates,))
            actions_one_hot = F.one_hot(samples, self.action_size).float()

            scores = score_fn(actions_one_hot)

            elite_idx = scores.topk(self.num_elite).indices
            elite = samples[elite_idx]

            new_probs = torch.zeros_like(probs)
            for t in range(self.horizon):
                counts = torch.bincount(elite[:, t], minlength=self.action_size).float()
                new_probs[t] = (counts + 1e-3) / (counts.sum() + self.action_size * 1e-3)

            probs = self.alpha * new_probs + (1 - self.alpha) * probs

            top_score, top_i = scores.max(0).values, scores.argmax()
            if top_score > best_score:
                best_score = top_score
                best_actions = actions_one_hot[top_i]

        return best_actions.unsqueeze(0)


class Planner:
    def __init__(self, rssm, clip_scorer, num_candidates=64, horizon=15,
                 gamma=0.99, score_mode="discounted_sum",
                 cem_num_iters=3, cem_elite_frac=0.1, cem_alpha=0.5):
        self.rssm = rssm
        self.clip_scorer = clip_scorer
        self.num_candidates = num_candidates
        self.horizon = horizon
        self.gamma = gamma
        self.score_mode = score_mode

        self._uniform_sampler = UniformCandidates(
            num_candidates, horizon, rssm.action_size, rssm.device
        )
        self._heuristic_sampler = HeuristicCandidates(
            num_candidates, horizon, rssm.action_size, rssm.device
        )
        self._cem_sampler = CEMCandidates(
            num_candidates, horizon, rssm.action_size, rssm.device,
            num_iters=cem_num_iters, elite_frac=cem_elite_frac, alpha=cem_alpha
        )

    def _clip_score_fn(self, recurrent_state, latent_state, score_mode=None):
        mode = score_mode if score_mode is not None else self.score_mode
        @torch.no_grad()
        def score_fn(actions_one_hot):
            trajectories = self.rssm.imagine_rollouts(
                recurrent_state, latent_state, actions_one_hot
            )
            return self.clip_scorer.score_rollouts(
                trajectories, self.gamma, mode=mode
            )
        return score_fn

    @torch.no_grad()
    def plan_action_cem(self, recurrent_state, latent_state, score_mode=None):
        score_fn = self._clip_score_fn(recurrent_state, latent_state, score_mode=score_mode)
        best = self._cem_sampler.plan(score_fn)
        return best[:, 0]

    @torch.no_grad()
    def plan_action_random_shooting(self, recurrent_state, latent_state,
                                     use_heuristic=False, use_argmax=False, score_mode=None):
        mode = score_mode if score_mode is not None else self.score_mode
        sampler = self._heuristic_sampler if use_heuristic else self._uniform_sampler
        candidates = sampler.sample()
        trajectories = self.rssm.imagine_rollouts(
            recurrent_state, latent_state, candidates
        )
        scores = self.clip_scorer.score_rollouts(
            trajectories, self.gamma, mode=mode
        )

        if use_argmax:
            best_idx = scores.argmax()
            return candidates[best_idx, 0].unsqueeze(0)

        first_actions = candidates[:, 0].argmax(dim=-1)
        agg = []
        for a in range(self.rssm.action_size):
            mask = first_actions == a
            agg.append(scores[mask].mean().item() if mask.any() else -float("inf"))
        best = max(range(len(agg)), key=lambda i: agg[i])
        return F.one_hot(
            torch.tensor(best, device=self.rssm.device).unsqueeze(0),
            self.rssm.action_size,
        ).float()

    @torch.no_grad()
    def plan_action(self, recurrent_state, latent_state):
        warnings.warn(
            "plan_action is deprecated, use plan_action_random_shooting(..., use_heuristic=True)",
            DeprecationWarning, stacklevel=2,
        )
        return self.plan_action_random_shooting(recurrent_state, latent_state, use_heuristic=True)

    @torch.no_grad()
    def plan_action_aggregated(self, recurrent_state, latent_state):
        warnings.warn(
            "plan_action_aggregated is deprecated, use plan_action_random_shooting(..., use_heuristic=True)",
            DeprecationWarning, stacklevel=2,
        )
        return self.plan_action_random_shooting(recurrent_state, latent_state, use_heuristic=True)

    @torch.no_grad()
    def _score_rollouts_reward(self, trajectories, gamma=0.99, mode="discounted_sum"):
        reward_logits = self.rssm.reward_predictor(trajectories)
        rewards = decode_two_hot(reward_logits, self.rssm.reward_bins)
        discounts = gamma ** torch.arange(
            trajectories.shape[1], device=rewards.device
        )
        if mode == "discounted_sum":
            return (rewards * discounts.unsqueeze(0)).sum(-1)
        elif mode == "max":
            return rewards.max(dim=-1).values
        elif mode == "max_discounted":
            return (rewards * discounts.unsqueeze(0)).max(dim=-1).values
        else:
            raise ValueError(f"unknown scoring mode: {mode}")

    @torch.no_grad()
    def plan_action_reward(self, recurrent_state, latent_state, score_mode=None):
        mode = score_mode if score_mode is not None else self.score_mode
        candidates = self._heuristic_sampler.sample()
        trajectories = self.rssm.imagine_rollouts(
            recurrent_state, latent_state, candidates
        )
        scores = self._score_rollouts_reward(trajectories, self.gamma, mode=mode)
        best_idx = scores.argmax()
        return candidates[best_idx, 0].unsqueeze(0)

    @torch.no_grad()
    def plan_action_reward_aggregated(self, recurrent_state, latent_state, score_mode=None):
        mode = score_mode if score_mode is not None else self.score_mode
        candidates = self._heuristic_sampler.sample()
        trajectories = self.rssm.imagine_rollouts(
            recurrent_state, latent_state, candidates
        )
        scores = self._score_rollouts_reward(trajectories, self.gamma, mode=mode)

        first_actions = candidates[:, 0].argmax(dim=-1)
        agg = []
        for a in range(self.rssm.action_size):
            mask = first_actions == a
            agg.append(scores[mask].mean().item() if mask.any() else -float("inf"))
        best = max(range(len(agg)), key=lambda i: agg[i])
        return F.one_hot(
            torch.tensor(best, device=self.rssm.device).unsqueeze(0),
            self.rssm.action_size,
        ).float()
