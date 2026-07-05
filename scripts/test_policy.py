#!/usr/bin/env python3
import argparse
import random

import numpy as np

from src.environment import make_minigrid_env


def goal_direction(obs):
    """Detect goal in partial view observation.
    Returns 'left', 'right', 'forward', or None if not visible."""
    green = obs[1]
    red = obs[0]
    blue = obs[2]
    mask = (green > 0.8) & (red < 0.3) & (blue < 0.3)
    if mask.sum() < 5:
        return None
    ys, xs = np.where(mask)
    cx = xs.mean()
    w = obs.shape[2]
    if cx < w * 0.4:
        return "left"
    elif cx > w * 0.6:
        return "right"
    return "forward"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="MiniGrid-Empty-6x6-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = make_minigrid_env(args.env_id, seed=args.seed, render_mode="human")
    obs_shape, action_size = env.observation_space.shape, env.action_space.n
    print(f"Env: {args.env_id}, obs={obs_shape}, actions={action_size}")

    total_reward = 0
    for episode in range(args.episodes):
        obs = env.reset()
        score = 0
        step = 0
        done = False

        while not done and step < args.max_steps:
            if random.random() < args.epsilon:
                valid = env.valid_actions()
                action_idx = random.choice(valid)
            else:
                direction = goal_direction(obs)
                if direction is not None:
                    if direction == "forward":
                        action_idx = 2
                    elif direction == "left":
                        action_idx = 0
                    else:
                        action_idx = 1
                else:
                    valid = env.valid_actions()
                    action_idx = random.choice(valid)

            obs, reward, done = env.step(action_idx)
            score += reward
            step += 1

        total_reward += score
        print(f"Episode {episode + 1}: {step} steps, reward={score:.1f}")

    env.close()
    avg = total_reward / args.episodes
    print(f"Average reward: {avg:.2f}")
    print("Done.")


if __name__ == "__main__":
    main()
