#!/usr/bin/env python3
import argparse
import datetime
import os

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, ViewSizeWrapper
from PIL import Image

matplotlib.use("Agg")


def visualize_trajectory(env_id, seed, output_dir, max_steps, make_gif):
    env = gym.make(env_id, render_mode="rgb_array", agent_view_size=3)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    frames = []
    obs, info = env.reset(seed=seed)
    done = False
    step = 0

    while not done and step < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        full_map = env.unwrapped.render()

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(full_map)
        axes[0].set_title("Full map")
        axes[0].axis("off")
        axes[1].imshow(obs)
        axes[1].set_title(f"Partial view (step {step})")
        axes[1].axis("off")
        plt.tight_layout()

        path = os.path.join(output_dir, f"step_{step:03d}.png")
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)

        frames.append(Image.open(path))
        step += 1

    if make_gif and frames:
        gif_path = os.path.join(output_dir, "trajectory.gif")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=120,
        )
        print(f"Saved GIF: {gif_path}")

    env.close()
    return step


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MiniGrid trajectories — save frames as PNG + GIF"
    )
    parser.add_argument(
        "--env-id",
        default="MiniGrid-Empty-16x16-v0",
        help="MiniGrid environment ID (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-steps", type=int, default=200, help="Max steps per episode"
    )
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF creation")
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("visualizations", f"trajectory_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    steps = visualize_trajectory(
        args.env_id, args.seed, out_dir, args.max_steps, not args.no_gif
    )
    print(f"Done: {steps} steps saved to {out_dir}/")


if __name__ == "__main__":
    main()
