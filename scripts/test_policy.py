#!/usr/bin/env python3
import argparse
import random

import numpy as np

from src.environment import make_minigrid_env


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="MiniGrid-Empty-6x6-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = make_minigrid_env(args.env_id, seed=args.seed, render_mode="human")
    print(f"Env: {args.env_id}, actions={env.action_space.n}")

    total_reward = 0.0
    for ep in range(args.episodes):
        obs = env.reset()
        score = 0.0
        done = False
        step = 0
        turn_dir = random.choice([0, 1])

        while not done and step < args.max_steps:
            if random.random() < args.epsilon:
                action = random.choice([0, 1, 2])
            elif goal_visible(obs):
                action = navigate_to_goal(env)
            elif wall_ahead(env):
                action = turn_dir
            else:
                action = 2

            obs, reward, done = env.step(action)
            score += reward
            step += 1

        total_reward += score
        print(f"Episode {ep + 1}: {step} steps, reward={score:.1f}")

    env.close()
    print(f"Average reward: {total_reward / args.episodes:.2f}")


if __name__ == "__main__":
    main()
