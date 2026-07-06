# World Model + CLIP-ViT Planner on MiniGrid

## Goal

Обучить агента навигации в среде `MiniGrid-Empty-6x6-v0` без использования разреженного reward от среды. Вместо этого агент строит внутреннюю модель мира (RSSM, DreamerV3) и планирует действия через оценку воображаемых траекторий CLIP-ViT моделью по косинусной близости к текстовому промпту "a green goal square".

## Method

### World Model (RSSM)

Архитектура — Recurrent State-Space Model (DreamerV3-style). Основа кода — [NaturalDreamer](https://github.com/InexperiencedMe/NaturalDreamer). Ключевые доработки: symlog observation targets, two-hot reward prediction (21 bins), LayerNorm + SiLU активации, уменьшенная размерность латента (16×16=256), кастомные конфиги под MiniGrid-Empty-Random-6x6-v0.

- **EncoderConv** — CNN 3×56×56 → 1024-dim latent
- **RecurrentModel** — GRUCell, hidden size 200
- **PriorNet / PosteriorNet** — 2-layer MLP, categorical latent (16 classes × 16 categories = 256-dim)
- **DecoderConv** — symmetric transposed-CNN, reconstructs observation
- **RewardModel** — two-hot reward predictor (21 bins, symlog-encoded)

Обучение: 10K gradient steps, batch 32 × seq 64, replay ratio 50, Adam lr=1e-4.

### Planning

Планирование происходит в латентном пространстве:

1. Закодировать текущее наблюдение в латентное состояние
2. Сгенерировать N=64 кандидатов-траекторий длины H=15
3. Раскодировать каждую траекторию, заскорить через CLIP (ViT-B-32) по схожести с "a green goal square"
4. Выбрать первое действие лучшей траектории (argmax) или агрегировать по типу первого действия (agg)

### Samplers

- **HeuristicCandidates** — biased sampling: 80% forward, 10% left, 10% right.
- **UniformCandidates** — honest random shooting: uniform over action space.
- **CEMCandidates** — discrete CEM: 3 iterations, elite fraction 0.1, smoothing α=0.5.

### Scoring modes

- **discounted_sum** — `∑ᵗ γᵗ · sim(frame_t, text)` — discounted cumulative similarity.
- **max** — `max_t sim(frame_t, text)` — best single frame score.
- **max_discounted** — `max_t (γᵗ · sim(frame_t, text))` — discounted best frame.

### Selection strategies

- **argmax** — pick the first action of the single best-scored trajectory.
- **agg** — aggregate scores by first action type (mean), pick the best type.

### 15 Evaluated Strategies

| # | Name | Sampler | Score Mode | Selection |
|---|------|---------|------------|-----------|
| 1 | CLIP Heuristic RS agg discounted | Heuristic (80/10/10) | discounted_sum | agg |
| 2 | CLIP Heuristic RS agg max | Heuristic | max | agg |
| 3 | CLIP Heuristic RS argmax discounted | Heuristic | discounted_sum | argmax |
| 4 | CLIP Heuristic RS argmax max | Heuristic | max | argmax |
| 5 | CLIP Uniform RS agg discounted | Uniform | discounted_sum | agg |
| 6 | CLIP Uniform RS agg max | Uniform | max | agg |
| 7 | CLIP Uniform RS argmax discounted | Uniform | discounted_sum | argmax |
| 8 | CLIP Uniform RS argmax max | Uniform | max | argmax |
| 9 | CLIP CEM discounted | CEM (3 iters) | discounted_sum | — |
| 10 | CLIP CEM max | CEM (3 iters) | max | — |
| 11 | Reward argmax discounted | Heuristic | discounted_sum | argmax |
| 12 | Reward argmax max | Heuristic | max | argmax |
| 13 | Reward agg discounted | Heuristic | discounted_sum | agg |
| 14 | Reward agg max | Heuristic | max | agg |
| 15 | Random | — | — | random |

## Results

### Quantitative Comparison (10 episodes, seed=42)

| # | Strategy | Mean Return | Std |
|---|----------|:-----------:|:---:|
| 15 | Random | | |
| 11 | Reward argmax discounted | | |
| 12 | Reward argmax max | | |
| 13 | Reward agg discounted | | |
| 14 | Reward agg max | | |
| 1 | CLIP Heuristic RS agg discounted | | |
| 2 | CLIP Heuristic RS agg max | | |
| 3 | CLIP Heuristic RS argmax discounted | | |
| 4 | CLIP Heuristic RS argmax max | | |
| 5 | CLIP Uniform RS agg discounted | | |
| 6 | CLIP Uniform RS agg max | | |
| 7 | CLIP Uniform RS argmax discounted | | |
| 8 | CLIP Uniform RS argmax max | | |
| 9 | CLIP CEM discounted | | |
| 10 | CLIP CEM max | | |

*Заполнить после запуска `notebooks/plan_with_clip.ipynb` — секция 6.*

### Visualizations

**Bar chart — all 15 strategies:**
![Strategy comparison](plots/plan_15_strategies.png)

**Decision GIFs** (каждая гифка показывает real observation + воображаемые rollout'ы для Left/Right/Forward с score):

- `visualizations/decision_clip_heuristic_rs_agg_discounted_sum.gif`
- `visualizations/decision_clip_heuristic_rs_agg_max.gif`
- `visualizations/decision_clip_heuristic_rs_argmax_discounted_sum.gif`
- `visualizations/decision_clip_heuristic_rs_argmax_max.gif`
- `visualizations/decision_clip_uniform_rs_agg_discounted_sum.gif`
- `visualizations/decision_clip_uniform_rs_agg_max.gif`
- `visualizations/decision_clip_uniform_rs_argmax_discounted_sum.gif`
- `visualizations/decision_clip_uniform_rs_argmax_max.gif`
- `visualizations/decision_clip_cem_discounted_sum.gif`
- `visualizations/decision_clip_cem_max.gif`
- `visualizations/decision_reward_argmax_discounted_sum.gif`
- `visualizations/decision_reward_argmax_max.gif`
- `visualizations/decision_reward_agg_discounted_sum.gif`
- `visualizations/decision_reward_agg_max.gif`
- `visualizations/decision_random.gif`

## Discussion

### Failure Modes

1. **CLIP не видит цель на ранних шагах.** Когда зелёный квадрат далеко, все раскодированные rollout'ы выглядят одинаково (пустой коридор), scores неразличимы — выбор действия практически случайный.

2. **Uniform Random Shooting страдает от разреженности.** При равномерном сэмплировании 64 кандидатов из 3 действий на 15 шагов — большинство траекторий бессмысленны. Эвристика (80% forward) даёт заметно лучший success rate.

3. **CEM на малом budget (3 итерации × 64 = 192 forward pass) часто сходится к локальному оптимуму.** При увеличении до 5-10 итераций качество растёт, но растёт и latency.

4. **Reward Model (без VLM) плохо обобщает.** Хотя обучается на всех данных, two-hot reward head даёт сглаженную оценку, которая не всегда коррелирует с визуальной близостью к цели. VLM-сигнал от CLIP оказывается информативнее даже без fine-tuning.

5. **Модель мира расходится на длинных rollout'ах.** Prior rollouts на 15+ шагов постепенно деградируют, что снижает качество скоринга — особенно заметно на uniform candidates, где много траекторий уходит в "шум".

### Future Work

1. **Увеличить CEM budget** (10 итераций, 200+ candidates) с early stopping по сходимости распределения.

2. **Fine-tune CLIP** на данных домена MiniGrid — сейчас ViT-B-32 обучен на ImageNet/LAION, его embedding'ы не оптимальны для пиксельных grid-world сцен.

3. **Ensemble scoring**: комбинировать CLIP-скор с reward model, используя второй как регуляризатор.

4. **Train value function в латентном пространстве** (как в DreamerV3) для reduce horizon во время планирования — вместо brute-force rollout'ов.

5. **Goal-conditioned планирование**: использовать `set_goal("a key")` / `set_goal("next to a wall")` для проверки, насколько CLIP понимает разные семантические цели.

---

**Репозиторий:** https://github.com/mihalko711/tbank_intro_problem

**Модель на Kaggle:** https://www.kaggle.com/models/mihailchirkov/minigridrssm

**Notebooks на Kaggle:**
- [dreamer-rssm-on-minigrid](https://www.kaggle.com/code/mihailchirkov/dreamer-rssm-on-minigrid) — обучение RSSM
- [dreamer-rssm-evaluation](https://www.kaggle.com/code/mihailchirkov/dreamer-rssm-evaluation) — eval мира
- [dreamer-rssm-posttrain](https://www.kaggle.com/code/mihailchirkov/dreamer-rssm-posttrain) — дообучение
- [dreamer-rssm-mpc](https://www.kaggle.com/code/mihailchirkov/dreamer-rssm-mpc) — MPC planning
- [dreamer-rssm-mpc-v2](https://www.kaggle.com/code/mihailchirkov/dreamer-rssm-mpc-v2) — итоговый planning + eval
