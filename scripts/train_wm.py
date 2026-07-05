import argparse
import os

import numpy as np
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
    # Fixed-layout eval env to isolate environment variance from model variance
    fixed_env_name = config["environment_name"].replace("-Random-", "-")
    if fixed_env_name != config["environment_name"]:
        env_eval_fixed = make_minigrid_env(fixed_env_name, seed=config["seed"] + 1000)
    else:
        env_eval_fixed = None
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

                def random_policy(state, env=env_eval):
                    valid = getattr(env, "valid_actions", lambda: list(range(action_size)))()
                    idx = np.random.choice(valid)
                    return torch.nn.functional.one_hot(
                        torch.tensor(idx, device=device).unsqueeze(0), action_size
                    ).float()

                num_ep = config["num_evaluation_episodes"]
                avg, std = evaluate(env_eval, rssm, random_policy, num_episodes=num_ep)

                if env_eval_fixed is not None:
                    def random_policy_fixed(state):
                        valid = getattr(env_eval_fixed, "valid_actions", lambda: list(range(action_size)))()
                        idx = np.random.choice(valid)
                        return torch.nn.functional.one_hot(
                            torch.tensor(idx, device=device).unsqueeze(0), action_size
                        ).float()
                    avg_fixed, std_fixed = evaluate(env_eval_fixed, rssm, random_policy_fixed, num_episodes=num_ep)
                    print(
                        f"Step {gs:>6d} | KL(raw)={metrics['kl_raw']:.4f} KL={metrics['kl_loss']:.2f} | "
                        f"Eval(random)={avg:.2f}±{std:.2f} Eval(fixed)={avg_fixed:.2f}±{std_fixed:.2f}"
                    )
                else:
                    print(
                        f"Step {gs:>6d} | KL(raw)={metrics['kl_raw']:.4f} KL={metrics['kl_loss']:.2f} | "
                        f"Eval={avg:.2f}±{std:.2f}"
                    )

        for _ in range(config["num_interaction_episodes"]):
            collect_episode(env, rssm, rssm.buffer)

        if iteration % 10 == 0:
            print(
                f"[{rssm.total_gradient_steps:>6d} grad steps] "
                f"wm={metrics['wm_loss']:.1f} recon={metrics['recon_loss']:.1f} "
                f"prior_recon={metrics['prior_recon_loss']:.4f} "
                f"reward={metrics['reward_loss']:.1f} nonzero={metrics['reward_nonzero']:.3f} "
                f"kl={metrics['kl_loss']:.1f} kl_raw={metrics['kl_raw']:.4f} "
                f"buffer={len(rssm.buffer)}"
            )

    env.close()
    env_eval.close()
    if env_eval_fixed is not None:
        env_eval_fixed.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minigrid_default.yml")
    main(parser.parse_args().config)
