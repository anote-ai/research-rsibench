# RSI-Bench harness validation (SimulatedLearner)

These numbers validate the harness and metrics against a deterministic toy
learner whose learning rules are known. They are **not** results about any
real model. Reproduce with `python scripts/run_demo.py`.

Setup: 80 feedback / 80 held-out tasks, 8 iterations,
at most 10 corrected errors revealed per round, 2% answer noise, seed 0.

## Main grid (ranked)

| Model | Mechanism | Channel | Feedback | S0 | ST | η | g | Accel | GenGap | Forget | AlignΔ | Level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| simulated-learner | self_reflection | prompt | corrections | 0.163 | 0.675 | +0.064 | +0.612 | -0.025 | +0.312 | 0.055 | +1.000 | SUSTAINED |
| simulated-learner | prompt_rewrite | prompt | corrections | 0.163 | 0.675 | +0.064 | +0.612 | -0.025 | +0.312 | 0.055 | +1.000 | SUSTAINED |
| simulated-learner | weight_update | weights | corrections | 0.163 | 0.675 | +0.064 | +0.612 | -0.025 | +0.312 | 0.055 | +1.000 | SUSTAINED |
| simulated-learner | fewshot_accumulation | memory | corrections | 0.163 | 0.625 | +0.058 | +0.552 | -0.012 | +0.213 | 0.316 | +1.000 | SUSTAINED |
| simulated-learner | none | none | score | 0.163 | 0.163 | +0.000 | +0.000 | +0.002 | +0.000 | 0.077 | +0.000 | NONE |
| simulated-learner | score_feedback | prompt | score | 0.163 | 0.163 | +0.000 | +0.000 | +0.002 | +0.000 | 0.077 | +0.000 | NONE |

Gain vs. no-improvement control: self_reflection +0.513, prompt_rewrite +0.513, weight_update +0.513, fewshot_accumulation +0.463, none +0.000, score_feedback +0.000

## Held-out score trajectories

| Mechanism | I=0 | I=1 | I=2 | I=3 | I=4 | I=5 | I=6 | I=7 | I=8 |
|---|---|---|---|---|---|---|---|---|---|
| none | 0.163 | 0.163 | 0.150 | 0.163 | 0.150 | 0.163 | 0.150 | 0.150 | 0.163 |
| score_feedback | 0.163 | 0.163 | 0.150 | 0.163 | 0.150 | 0.163 | 0.150 | 0.150 | 0.163 |
| fewshot_accumulation | 0.163 | 0.338 | 0.525 | 0.662 | 0.475 | 0.463 | 0.562 | 0.537 | 0.625 |
| self_reflection | 0.163 | 0.338 | 0.525 | 0.662 | 0.688 | 0.688 | 0.662 | 0.675 | 0.675 |
| prompt_rewrite | 0.163 | 0.338 | 0.525 | 0.662 | 0.688 | 0.688 | 0.662 | 0.675 | 0.675 |
| weight_update | 0.163 | 0.338 | 0.525 | 0.662 | 0.688 | 0.688 | 0.662 | 0.675 | 0.675 |

## Per-suite held-out score (I=0 → I=8)

| Mechanism | arithmetic | cipher | instruction | lookup |
|---|---|---|---|---|
| none | 0.65 → 0.65 | 0.00 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 |
| score_feedback | 0.65 → 0.65 | 0.00 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 |
| fewshot_accumulation | 0.65 → 0.95 | 0.00 → 0.55 | 0.00 → 1.00 | 0.00 → 0.00 |
| self_reflection | 0.65 → 0.95 | 0.00 → 0.75 | 0.00 → 1.00 | 0.00 → 0.00 |
| prompt_rewrite | 0.65 → 0.95 | 0.00 → 0.75 | 0.00 → 1.00 | 0.00 → 0.00 |
| weight_update | 0.65 → 0.95 | 0.00 → 0.75 | 0.00 → 1.00 | 0.00 → 0.00 |

## Ablation A: feedback richness (self_reflection)

| Model | Mechanism | Channel | Feedback | S0 | ST | η | g | Accel | GenGap | Forget | AlignΔ | Level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| simulated-learner | self_reflection | prompt | errors | 0.163 | 0.412 | +0.031 | +0.299 | -0.021 | +0.000 | 0.061 | +1.000 | SUSTAINED |
| simulated-learner | self_reflection | prompt | corrections | 0.163 | 0.675 | +0.064 | +0.612 | -0.025 | +0.312 | 0.055 | +1.000 | SUSTAINED |

## Ablation B: context window (fewshot_accumulation, max 200 examples)

| Model | Mechanism | Channel | Feedback | S0 | ST | η | g | Accel | GenGap | Forget | AlignΔ | Level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sim-ctx1500 | fewshot_accumulation | memory | corrections | 0.163 | 0.662 | +0.062 | +0.597 | -0.025 | +0.200 | 0.056 | +1.000 | SUSTAINED |
| sim-ctx6000 | fewshot_accumulation | memory | corrections | 0.163 | 0.675 | +0.064 | +0.612 | -0.025 | +0.312 | 0.055 | +1.000 | SUSTAINED |
| sim-ctx50000 | fewshot_accumulation | memory | corrections | 0.163 | 0.675 | +0.064 | +0.612 | -0.025 | +0.312 | 0.055 | +1.000 | SUSTAINED |

## Ablation C: label budget per round (weight_update)

| Model | Mechanism | Channel | Feedback | S0 | ST | η | g | Accel | GenGap | Forget | AlignΔ | Level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sim-budget2 | weight_update | weights | corrections | 0.163 | 0.463 | +0.038 | +0.358 | -0.002 | +0.125 | 0.059 | +0.700 | SUSTAINED |
| sim-budget10 | weight_update | weights | corrections | 0.163 | 0.675 | +0.064 | +0.612 | -0.025 | +0.312 | 0.055 | +1.000 | SUSTAINED |
| sim-budgetall | weight_update | weights | corrections | 0.163 | 0.675 | +0.064 | +0.612 | -0.075 | +0.312 | 0.055 | +1.000 | ONE_SHOT |

## Frontier

24 held-out items were never solved by any mechanism at any iteration
(suites: cipher, lookup).
