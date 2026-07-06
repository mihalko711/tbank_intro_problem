# `src/` — исходный код

Python-пакет с реализацией world model (DreamerV3-style RSSM), CLIP-планировщика и инфраструктуры для MiniGrid.

Основа кода — [NaturalDreamer](https://github.com/InexperiencedMe/NaturalDreamer).

## Модули

### `rssm.py` — RSSMWorldModel

Главный класс. Оркестрирует encoder, recurrent model, prior/posterior, decoder, reward predictor.

**Методы:**
- `train_step(data)` — forward pass по последовательности, KL + reconstruction + reward loss, backward
- `encode_step(recurrent, latent, action, obs)` — single-step posterior
- `imagine_rollouts(recurrent, latent, candidate_actions)` — open-loop prior rollouts
- `rollout_prior(recurrent, latent, action_fn, horizon)` — closed-loop роллауты
- `save_checkpoint / load_checkpoint`

### `networks.py` — PyTorch-модули

| Класс | Назначение |
|---|---|
| `EncoderConv` | CNN 3×56×56 → 1024 |
| `DecoderConv` | transposed-CNN 1024 → 3×56×56 |
| `RecurrentModel` | GRUCell(h=200) |
| `PriorNet` | MLP → categorical prior |
| `PosteriorNet` | MLP → categorical posterior |
| `RewardModel` | MLP → two-hot reward |
| `ContinueModel` | MLP → продолжение эпизода |

### `planner.py` — Планировщик

| Класс | Назначение |
|---|---|
| `UniformCandidates` | honest random shooting (uniform) |
| `HeuristicCandidates` | biased (80% forward, 10% left, 10% right) |
| `CLIPScorer` | ViT-B-32 scorer + `set_goal()` |
| `CEMCandidates` | дискретный CEM (3 итерации) |
| `Planner` | планирование через VLM / reward scorer |

**Методы Planner:**
- `plan_action_cem` — CEM + VLM
- `plan_action_random_shooting` — RS + VLM (uniform или heuristic)
- `plan_action_reward` — RS + learned reward model
- `plan_action_reward_aggregated` — RS + reward + agg

### `environment.py` — MiniGrid обёртки

| Функция | Назначение |
|---|---|
| `make_minigrid_env` | фабрика env (PixelsWrapper + DoneWrapper + partial obs) |
| `get_env_properties` | obs_shape, action_size |
| `collect_episode` | сбор одного эпизода в buffer |
| `evaluate` | eval N эпизодов, возврат mean/std return |
| `action_to_env` | one-hot → env action |

### `buffer.py` — ReplayBuffer

Круговой numpy-буфер фиксированной ёмкости. `sample(batch, seq_len)` возвращает последовательности с `is_first`-флагами.

### `policy.py` — ScriptedPolicy

Эвристика: детект зелёной цели по пикселям → навигация к ней; поворот при стене; иначе forward. Используется для сбора данных.

### `utils.py` — Утилиты

- `seed_everything` — воспроизводимость
- `sequential_model_1d` — MLP фабрика (Linear + LayerNorm + activation)
- `symlog / symexp` — DreamerV3 transforms
- `two_hot_encode / decode_two_hot` — классификационная награда
- `Moments` — бегущие квантили
