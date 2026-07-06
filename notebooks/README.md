# `notebooks/` — Jupyter ноутбуки

| Notebook | Описание |
|---|---|
| `train_world_model.ipynb` | Обучение RSSM с нуля: создание env, инициализация модели, цикл gradient steps, live loss curves, периодическая реконструкция. |
| `evaluate_world_model.ipynb` | Загрузка чекпоинта, сбор эпизодов, сравнение real vs posterior vs prior rollout'ов. |
| `fine_tune_world_model.ipynb` | Загрузка чекпоинта и дообучение с переопределёнными гиперпараметрами. |
| `plan_with_clip.ipynb` | CLIP-планировщик: визуализация кандидатов, eval 15 стратегий, score distributions, decision GIFs. |

## Типичный workflow

1. `train_world_model.ipynb` — обучить модель
2. `evaluate_world_model.ipynb` — оценить качество реконструкции
3. `fine_tune_world_model.ipynb` — дообучить при необходимости
4. `plan_with_clip.ipynb` — запустить планировщик, собрать метрики, сгенерировать GIF

## `plan_with_clip.ipynb` — детали

Содержит полную матрицу из 15 стратегий:

| Категория | Варианты |
|---|---|
| Sampler | Heuristic RS, Uniform RS, CEM |
| Score mode | discounted_sum, max |
| Selection | argmax, agg |
| Reward-only | argmax, agg |
| Baseline | Random |

Генерирует:
- Таблицу return/std для всех 15 стратегий (`results`)
- Bar chart `plots/plan_15_strategies.png`
- 15 decision GIF `visualizations/decision_*.gif`
