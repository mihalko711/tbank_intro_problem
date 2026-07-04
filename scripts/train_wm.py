import argparse
import os

import torch
import yaml

from src import (
    RSSMWorldModel,
    collect_episode,
    evaluate,
    get_env_properties,
    make_minigrid_env,
    seed_everything,
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path):
    config = load_config(config_path)
    rssm_cfg = config["rssm"]
    seed_everything(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = make_minigrid_env(config["environment_name"], seed=config["seed"])
    env_eval = make_minigrid_env(config["environment_name"], seed=config["seed"] + 999)
    obs_shape, action_size = get_env_properties(env)
    print(f"Obs shape: {obs_shape}, action size: {action_size}")

    rssm = RSSMWorldModel(obs_shape, action_size, rssm_cfg, device)

    # ── Prepare directories ──
    run_name = f"{config['environment_name']}_{config['run_name']}"
    checkpoint_dir = config["folder_names"]["checkpoints_folder"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Initial data collection ──
    print(f"Collecting {config['episodes_before_start']} initial episodes...")
    for _ in range(config["episodes_before_start"]):
        collect_episode(env, rssm, rssm.buffer)
    print(f"Buffer size: {len(rssm.buffer)}")

    # ── Training loop ──
    iterations = config["gradient_steps"] // config["replay_ratio"]
    for iteration in range(iterations):
        for _ in range(config["replay_ratio"]):
            data = rssm.buffer.sample(rssm_cfg["batch_size"], rssm_cfg["batch_length"])
            _, metrics = rssm.train_step(data)
            rssm.total_gradient_steps += 1

            gs = rssm.total_gradient_steps
            if config["save_checkpoints"] and gs % config["checkpoint_interval"] == 0:
                path = os.path.join(checkpoint_dir, f"{run_name}_{gs // 1000}k")
                rssm.save_checkpoint(path)
                avg, std = evaluate(
                    env_eval,
                    rssm,
                    lambda s: torch.nn.functional.one_hot(
                        torch.randint(0, action_size, (1,), device=device), action_size
                    ).float(),
                )
                print(
                    f"Step {gs:>6d} | WM loss: {metrics['wm_loss']:.2f} | "
                    f"Eval reward: {avg:.2f} ± {std:.2f}"
                )

        for _ in range(config["num_interaction_episodes"]):
            collect_episode(env, rssm, rssm.buffer)

        if iteration % 10 == 0:
            wm = metrics["wm_loss"]
            recon = metrics["recon_loss"]
            rw = metrics["reward_loss"]
            kl = metrics["kl_loss"]
            print(
                f"[{rssm.total_gradient_steps:>6d} grad steps] "
                f"wm={wm:.1f} recon={recon:.1f} reward={rw:.1f} kl={kl:.1f} "
                f"buffer={len(rssm.buffer)}"
            )

    env.close()
    env_eval.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minigrid_default.yml")
    main(parser.parse_args().config)
