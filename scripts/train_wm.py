import argparse
import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src import (
    RSSMWorldModel,
    collect_episode,
    evaluate,
    get_env_properties,
    make_minigrid_env,
    seed_everything,
)
from src.policy import ScriptedPolicy
from src.planner import CLIPScorer, Planner


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
    fixed_env_name = config["environment_name"].replace("-Random-", "-")
    if fixed_env_name != config["environment_name"]:
        env_eval_fixed = make_minigrid_env(fixed_env_name, seed=config["seed"] + 1000)
    else:
        env_eval_fixed = None
    obs_shape, action_size = get_env_properties(env)
    print(f"Obs shape: {obs_shape}, action size: {action_size}")

    rssm = RSSMWorldModel(obs_shape, action_size, rssm_cfg, device)
    scripted_policy = ScriptedPolicy(env, action_size, epsilon=0.05, device=device)

    try:
        clip_scorer = CLIPScorer(device, rssm)
        planner = Planner(rssm, clip_scorer, num_candidates=64, horizon=rssm_cfg["imagination_horizon"], gamma=0.99)
        print("Planner (CLIP-ViT) initialized")
    except Exception as e:
        print(f"Planner init failed ({e}), falling back to random policy for evaluation")
        planner = None

    run_name = f"{config['environment_name']}_{config['run_name']}"
    checkpoint_dir = config["folder_names"]["checkpoints_folder"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Collecting {config['episodes_before_start']} initial episodes...")
    n_init = config["episodes_before_start"]
    for _ in range(n_init // 2):
        collect_episode(env, rssm, rssm.buffer, action_fn=scripted_policy)
        collect_episode(env, rssm, rssm.buffer)
    if n_init % 2:
        collect_episode(env, rssm, rssm.buffer, action_fn=scripted_policy)
    print(f"Buffer size: {len(rssm.buffer)}")

    iterations = config["gradient_steps"] // config["replay_ratio"]
    pbar = tqdm(total=config["gradient_steps"], desc="Training")

    for iteration in range(iterations):
        for _ in range(config["replay_ratio"]):
            data = rssm.buffer.sample(rssm_cfg["batch_size"], rssm_cfg["batch_length"])
            _, metrics = rssm.train_step(data)
            rssm.total_gradient_steps += 1

            pbar.set_postfix(
                wm=f"{metrics['wm_loss']:.2f}",
                recon=f"{metrics['recon_loss']:.2f}",
                prior=f"{metrics['prior_recon_loss']:.3f}",
                reward=f"{metrics['reward_loss']:.2f}",
                nonzero=f"{metrics['reward_nonzero']:.3f}",
                kl_raw=f"{metrics['kl_raw']:.4f}",
                kl_act=f"{metrics['kl_active']:.2f}",
                buf=f"{len(rssm.buffer)}",
            )
            pbar.update(1)

            gs = rssm.total_gradient_steps
            if config["save_checkpoints"] and gs % config["checkpoint_interval"] == 0:
                path = os.path.join(checkpoint_dir, f"{run_name}_{gs // 1000}k")
                rssm.save_checkpoint(path)

                def planner_policy(state, obs=None):
                    if planner is not None:
                        rec = state[:, : rssm.recurrent_size]
                        lat = state[:, rssm.recurrent_size :]
                        return planner.plan_action_random_shooting(rec, lat, use_heuristic=True)
                    valid = getattr(env_eval, "valid_actions", lambda: list(range(action_size)))()
                    idx = np.random.choice(valid)
                    return torch.nn.functional.one_hot(
                        torch.tensor(idx, device=device).unsqueeze(0), action_size
                    ).float()

                num_ep = config["num_evaluation_episodes"]
                avg, std = evaluate(env_eval, rssm, planner_policy, num_episodes=num_ep)

                if env_eval_fixed is not None:
                    def planner_policy_fixed(state, obs=None):
                        if planner is not None:
                            rec = state[:, : rssm.recurrent_size]
                            lat = state[:, rssm.recurrent_size :]
                            return planner.plan_action_random_shooting(rec, lat, use_heuristic=True)
                        valid = getattr(env_eval_fixed, "valid_actions", lambda: list(range(action_size)))()
                        idx = np.random.choice(valid)
                        return torch.nn.functional.one_hot(
                            torch.tensor(idx, device=device).unsqueeze(0), action_size
                        ).float()
                    avg_fixed, std_fixed = evaluate(env_eval_fixed, rssm, planner_policy_fixed, num_episodes=num_ep)
                    tag = "Planner" if planner is not None else "random"
                    pbar.write(
                        f"Step {gs:>6d} | Eval({tag})={avg:.2f}±{std:.2f} Eval(fixed)={avg_fixed:.2f}±{std_fixed:.2f}"
                    )
                else:
                    tag = "Planner" if planner is not None else "random"
                    pbar.write(
                        f"Step {gs:>6d} | Eval({tag})={avg:.2f}±{std:.2f}"
                    )

        n_interact = config["num_interaction_episodes"]
        for _ in range(n_interact // 2):
            collect_episode(env, rssm, rssm.buffer, action_fn=scripted_policy)
            collect_episode(env, rssm, rssm.buffer)
        if n_interact % 2:
            collect_episode(env, rssm, rssm.buffer, action_fn=scripted_policy)

    pbar.close()
    env.close()
    env_eval.close()
    if env_eval_fixed is not None:
        env_eval_fixed.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minigrid_default.yml")
    main(parser.parse_args().config)
