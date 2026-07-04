import torch


class Planner:
    def __init__(self, rssm, vlm_scorer, num_candidates=64, horizon=15):
        self.rssm = rssm
        self.vlm_scorer = vlm_scorer
        self.num_candidates = num_candidates
        self.horizon = horizon

    @torch.no_grad()
    def plan_action(self, recurrent_state, latent_state):
        candidate_actions = self._sample_candidates()
        trajectories = self.rssm.imagine_rollouts(
            recurrent_state, latent_state, candidate_actions
        )
        scores = self.vlm_scorer(trajectories)
        best_idx = scores.argmax()
        return candidate_actions[best_idx, 0]

    def _sample_candidates(self):
        actions = torch.randint(
            0, self.rssm.action_size,
            (self.num_candidates, self.horizon, 1),
            device=self.rssm.device,
        )
        return torch.nn.functional.one_hot(actions, self.rssm.action_size).squeeze(-2).float()
