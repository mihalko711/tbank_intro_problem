import torch
import torch.nn.functional as F
import open_clip

from .utils import symexp


class HeuristicCandidates:
    def __init__(self, num_candidates, horizon, action_size, device):
        self.num_candidates = num_candidates
        self.horizon = horizon
        self.action_size = action_size
        self.device = device

    @torch.no_grad()
    def sample(self):
        probs = torch.rand(self.num_candidates, self.horizon, device=self.device)
        actions = torch.where(
            probs < 0.1,
            torch.zeros_like(probs, dtype=torch.long),
            torch.where(
                probs < 0.2,
                torch.ones_like(probs, dtype=torch.long),
                torch.full_like(probs, 2, dtype=torch.long),
            ),
        )
        return F.one_hot(actions, self.action_size).float()


class CLIPScorer:
    def __init__(self, device, rssm, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.device = device
        self.rssm = rssm
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.model.eval()
        self._compute_text_embedding()

        self.normalize_mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=device
        ).view(1, 3, 1, 1)
        self.normalize_std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=device
        ).view(1, 3, 1, 1)

    def _compute_text_embedding(self):
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        text = tokenizer(["a green goal square"])
        text_emb = self.model.encode_text(text.to(self.device))
        self.text_embedding = F.normalize(text_emb, dim=-1).squeeze(0)

    @torch.no_grad()
    def score_rollouts(self, trajectories, gamma=0.99):
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
        discounts = gamma ** torch.arange(horizon, device=self.device)
        return (sims * discounts.unsqueeze(0)).sum(-1)


class Planner:
    def __init__(self, rssm, clip_scorer, num_candidates=64, horizon=15, gamma=0.99):
        self.rssm = rssm
        self.clip_scorer = clip_scorer
        self.num_candidates = num_candidates
        self.horizon = horizon
        self.gamma = gamma
        self._candidate_sampler = HeuristicCandidates(
            num_candidates, horizon, rssm.action_size, rssm.device
        )

    @torch.no_grad()
    def plan_action(self, recurrent_state, latent_state):
        candidates = self._candidate_sampler.sample()
        trajectories = self.rssm.imagine_rollouts(
            recurrent_state, latent_state, candidates
        )
        scores = self.clip_scorer.score_rollouts(trajectories, self.gamma)
        best_idx = scores.argmax()
        return candidates[best_idx, 0].unsqueeze(0)

    @torch.no_grad()
    def plan_action_aggregated(self, recurrent_state, latent_state):
        candidates = self._candidate_sampler.sample()
        trajectories = self.rssm.imagine_rollouts(
            recurrent_state, latent_state, candidates
        )
        scores = self.clip_scorer.score_rollouts(trajectories, self.gamma)

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
