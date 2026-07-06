# `scripts/` — исполняемые скрипты

| Скрипт | Назначение |
|---|---|
| `train_wm.py` | Основной скрипт обучения. Парсит YAML-конфиг, создаёт RSSM + planner, заполняет buffer, цикл тренировки с evaluation. |

```bash
python scripts/train_wm.py --config configs/minigrid_default.yml
```

| `test_policy.py` | Тест скриптованной политики в human-render окне. Проверяет детект цели, навигацию, повороты. |

```bash
python scripts/test_policy.py --env-id MiniGrid-Empty-6x6-v0 --episodes 5
```

| `env_exploration.py` | Визуализация MiniGrid: full map vs partial view агента side-by-side. Случайные действия. |

```bash
python scripts/env_exploration.py
```

| `visualize_trajectories.py` | Генерация PNG-кадров и GIF случайных траекторий. |

```bash
python scripts/visualize_trajectories.py --env-id MiniGrid-Empty-6x6-v0 --seed 42
```

## Зависимости

Скрипты используют `src/` пакет и конфиги из `configs/`. Некоторые скрипты (`env_exploration.py`, `visualize_trajectories.py`) могут работать напрямую с Gymnasium/MiniGrid без `src/`.
