#!/usr/bin/env python3
import argparse
import random

import numpy as np

from src.environment import make_minigrid_env, action_to_env


def find_goal(env):
    grid = env.unwrapped.grid
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                return np.array([x, y], dtype=float)
    return None


def goal_seeking_action(env, epsilon):
    if random.random() < epsilon:
        return random.choice([0, 1, 2])

    agent_pos = np.array(env.unwrapped.agent_pos, dtype=float)
    agent_dir = env.unwrapped.agent_dir
    goal_pos = find_goal(env)

    if goal_pos is None:
        return random.choice([0, 1, 2])

    diff = goal_pos - agent_pos
    abs_diff = np.abs(diff)

    if abs_diff[0] > abs_diff[1]:
        target_dir = 0 if diff[0] > 0 else 2
    else:
        target_dir = 1 if diff[1] > 0 else 3

    if agent_dir == target_dir:
        return 2
    else:
        left_turn = (agent_dir - 1) % 4
        right_turn = (agent_dir + 1) % 4
        if left_turn == target_dir or right_turn != target_dir:
            return 0
        else:
            return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="MiniGrid-Empty-6x6-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = make_minigrid_env(args.env_id, seed=args.seed, render_mode="human")
    obs_shape, action_size = env.observation_space.shape, env.action_space.n
    print(f"Env: {args.env_id}, obs={obs_shape}, actions={action_size}")

    for episode in range(args.episodes):
        obs = env.reset()
        score = 0
        step = 0
        done = False

        while not done and step < args.max_steps:
            action_idx = goal_seeking_action(env, args.epsilon)
            obs, reward, done = env.step(action_idx)
            score += reward
            step += 1

        print(f"Episode {episode + 1}: {step} steps, reward={score:.1f}")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
