import random

import torch
import torch.nn.functional as F


def goal_visible(obs):
    green = obs[1]
    red = obs[0]
    blue = obs[2]
    mask = (green > 0.8) & (red < 0.3) & (blue < 0.3)
    return mask.sum() > 20


def find_goal(env):
    grid = env.unwrapped.grid
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                return x, y
    return None


def wall_ahead(env):
    front = env.unwrapped.grid.get(*env.unwrapped.front_pos)
    return front is not None


def navigate_to_goal(env):
    ax, ay = env.unwrapped.agent_pos
    agent_dir = env.unwrapped.agent_dir
    gx, gy = find_goal(env)

    if gx is None:
        return random.choice([0, 1, 2])

    dx = gx - ax
    dy = gy - ay

    if abs(dx) >= abs(dy):
        target_dir = 0 if dx > 0 else 2
    else:
        target_dir = 1 if dy > 0 else 3

    if agent_dir == target_dir:
        return 2
    left = (agent_dir - 1) % 4
    return 0 if left == target_dir else 1


class ScriptedPolicy:
    def __init__(self, env, action_size, epsilon=0.05, device="cpu"):
        self.env = env
        self.action_size = action_size
        self.epsilon = epsilon
        self.device = device
        self.turn_dir = random.choice([0, 1])

    @torch.no_grad()
    def __call__(self, full_state, observation):
        if random.random() < self.epsilon:
            idx = random.choice([0, 1, 2])
        elif goal_visible(observation):
            idx = navigate_to_goal(self.env)
        elif wall_ahead(self.env):
            idx = self.turn_dir
        else:
            idx = 2
        return F.one_hot(
            torch.tensor(idx, device=self.device), self.action_size
        ).float().unsqueeze(0)
