# tbank_intro_problem

Решение вступительного задания для магистратуры Т-Банк + ФПМИ МФТИ.

World model-based RL агент для навигации в MiniGrid с планированием через CLIP-ViT.

Основа кода — [NaturalDreamer](https://github.com/InexperiencedMe/NaturalDreamer) (DreamerV3). Доработки:
- Система типов: symlog observation targets + two-hot reward prediction
- LayerNorm + SiLU активации во всех сетях
- Кастомные конфиги (уменьшенная размерность латента)
- Интеграция с MiniGrid-Empty-Random-6x6-v0

## Архитектура

```
Agent
 ├── RSSMWorldModel (DreamerV3)
 │    ├── EncoderConv       CNN 3×56×56 → 1024
 │    ├── RecurrentModel    GRUCell, h=200
 │    ├── PriorNet          латентный prior (16×16=256)
 │    ├── PosteriorNet      латентный posterior
 │    ├── DecoderConv       transposed-CNN → реконструкция
 │    ├── RewardModel       two-hot reward predictor (21 bins)
 │    └── ReplayBuffer      круговой буфер на 500k
 │
 ├── Planner
 │    ├── CLIPScorer        ViT-B-32, cosine similarity к тексту
 │    ├── HeuristicCandidates   80/10/10 biased sampling
 │    ├── UniformCandidates     uniform random shooting
 │    └── CEMCandidates        дискретный CEM
 │
 └── ScriptedPolicy        эвристика для сбора данных
```

## Требования

- Python 3.12
- PyTorch, Gymnasium, MiniGrid, open-clip-torch, Matplotlib, NumPy, PyYAML, tqdm

## Быстрый старт

```bash
# установка
uv sync

# обучение world model
python scripts/train_wm.py --config configs/minigrid_default.yml

# планирование + eval (ноутбук)
jupyter notebook notebooks/plan_with_clip.ipynb
```

## Структура

```
├── configs/             YAML-конфиги
├── src/                 исходный код (Python-пакет)
│   ├── buffer.py        replay buffer
│   ├── environment.py   обёртки MiniGrid + сбор эпизодов
│   ├── networks.py      PyTorch-модули (encoder, decoder, RSSM)
│   ├── planner.py       CLIP-scorer, candidate samplers, планировщик
│   ├── policy.py        скриптованная политика
│   ├── rssm.py          RSSMWorldModel — главный класс
│   └── utils.py         утилиты (symlog, two-hot, seed)
├── scripts/             исполняемые скрипты
│   ├── train_wm.py      обучение world model
│   ├── test_policy.py   тест скриптованной политики
│   ├── env_exploration.py     визуализация env
│   └── visualize_trajectories.py   PNG+GIF траекторий
├── notebooks/           Jupyter ноутбуки
│   ├── train_world_model.ipynb        обучение RSSM
│   ├── evaluate_world_model.ipynb     загрузка + роллауты
│   ├── fine_tune_world_model.ipynb    дообучение
│   └── plan_with_clip.ipynb          CLIP-планирование + eval 15 стратегий
├── checkpoints/         сохранённые модели (gitignored)
└── visualizations/      сгенерированные GIF и PNG (gitignored)
```

## Публичный API

```python
from src import (
    RSSMWorldModel,                # мировая модель
    Planner, CLIPScorer,           # планировщик + VLM-scorer
    HeuristicCandidates,           # biased random shooting
    UniformCandidates,             # uniform random shooting
    CEMCandidates,                 # дискретный CEM
    make_minigrid_env,             # фабрика env
    collect_episode, evaluate,     # сбор данных и eval
    seed_everything,               # воспроизводимость
)
```

## Kaggle

- **Model:** [minigridrssm](https://www.kaggle.com/models/mihailchirkov/minigridrssm)
- **Notebooks:**
  - [dreamer-rssm-on-minigrid](https://www.kaggle.com/code/mihailchirkov/dreamer-rssm-on-minigrid) — обучение RSSM
  - [dreamer-rssm-evaluation](https://www.kaggle.com/code/mihailchirkov/dreamer-rssm-evaluation) — eval мира
  - [dreamer-rssm-posttrain](https://www.kaggle.com/code/mihailchirkov/dreamer-rssm-posttrain) — дообучение
  - [dreamer-rssm-mpc](https://www.kaggle.com/code/mihailchirkov/dreamer-rssm-mpc) — MPC planning
  - [dreamer-rssm-mpc-v2](https://www.kaggle.com/code/mihailchirkov/dreamer-rssm-mpc-v2) — итоговый planning + eval
