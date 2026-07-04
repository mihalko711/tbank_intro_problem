import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
env = RGBImgPartialObsWrapper(env)  # obs теперь = partial view агента
env = ImgObsWrapper(env)  # убираем mission-поле

obs, info = env.reset()

plt.ion()
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    full_view = env.unwrapped.render()  # полная карта
    partial_view = obs  # то, что видит агент

    axes[0].imshow(full_view)
    axes[0].set_title("Полная карта")
    axes[0].axis("off")

    axes[1].imshow(partial_view)
    axes[1].set_title("Вид агента (partial)")
    axes[1].axis("off")

    plt.pause(0.05)

    if terminated or truncated:
        obs, info = env.reset()

plt.ioff()
plt.show()
env.close()
