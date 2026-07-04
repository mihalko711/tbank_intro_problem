import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


class PixelsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = self.observation_space
        new_shape = (obs_space.shape[-1],) + obs_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=new_shape, dtype=np.float32
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1)) / 255.0


class DoneWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated

    def reset(self, seed=None, **kwargs):
        obs, info = self.env.reset(seed=seed)
        return obs


class ActionMaskWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)

    def action(self, action):
        return action


def make_minigrid_env(env_id, seed=None, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = DoneWrapper(PixelsWrapper(env))
    env = ActionMaskWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def get_env_properties(env):
    obs_shape = env.observation_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_size = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        action_size = env.action_space.shape[0]
    else:
        raise ValueError(f"Unsupported action space: {type(env.action_space)}")
    return obs_shape, action_size


def one_hot(action, num_actions):
    oh = np.zeros(num_actions, dtype=np.float32)
    oh[action] = 1.0
    return oh


def action_to_env(action_tensor):
    if action_tensor.ndim == 0 or action_tensor.shape[-1] == 1:
        return int(action_tensor.item())
    return int(np.argmax(action_tensor))


@torch.no_grad()
def collect_episode(env, rssm, buffer, action_fn=None):
    recurrent_state, latent_state = rssm.reset_state()
    action = torch.zeros(1, rssm.action_size, device=rssm.device)

    observation = env.reset()
    score = 0
    step_count = 0
    done = False

    while not done:
        recurrent_state = rssm.recurrent_model(recurrent_state, latent_state, action)
        encoded_obs = rssm.encoder(
            torch.from_numpy(observation).float().unsqueeze(0).to(rssm.device)
        )
        latent_state, _ = rssm.posterior_net(
            torch.cat((recurrent_state, encoded_obs.view(1, -1)), -1)
        )

        full_state = torch.cat((recurrent_state, latent_state), -1)
        if action_fn is not None:
            action = action_fn(full_state)
        else:
            action = torch.nn.functional.one_hot(
                torch.randint(0, rssm.action_size, (1,), device=rssm.device),
                num_classes=rssm.action_size,
            ).float()

        action_numpy = action.cpu().numpy().reshape(-1)
        env_action = action_to_env(action_numpy)

        next_observation, reward, done = env.step(env_action)
        buffer.add(observation, action_numpy, reward, done)

        observation = next_observation
        score += reward
        step_count += 1

    return score, step_count


@torch.no_grad()
def evaluate(env, rssm, action_fn, num_episodes=10):
    scores = []
    for _ in range(num_episodes):
        recurrent_state, latent_state = rssm.reset_state()
        action = torch.zeros(1, rssm.action_size, device=rssm.device)
        observation = env.reset()
        score = 0
        done = False

        while not done:
            recurrent_state = rssm.recurrent_model(
                recurrent_state, latent_state, action
            )
            encoded_obs = rssm.encoder(
                torch.from_numpy(observation).float().unsqueeze(0).to(rssm.device)
            )
            latent_state, _ = rssm.posterior_net(
                torch.cat((recurrent_state, encoded_obs.view(1, -1)), -1)
            )
            full_state = torch.cat((recurrent_state, latent_state), -1)
            action = action_fn(full_state)

            action_numpy = action.cpu().numpy().reshape(-1)
            next_observation, reward, done = env.step(action_to_env(action_numpy))
            observation = next_observation
            score += reward

        scores.append(score)
    return np.mean(scores), np.std(scores)
