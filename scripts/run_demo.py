#!/usr/bin/env python3
"""Reproduce the harness-validation results in the paper (no API keys needed).

Runs every improvement mechanism against the deterministic SimulatedLearner on
the four core suites and writes:

  results/sim_trajectories.json   full per-iteration, per-item trajectories
  results/sim_results.md          summary tables (main grid + ablations)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rsibench import (  # noqa: E402
    FeedbackLevel,
    SimulatedLearner,
    frontier_items,
    get_mechanism,
    make_benchmark,
    markdown_table,
    rank,
    run_rsi,
    summarize,
)

ITERATIONS = 8
BUDGET = 10
SEED = 0
ROOT = os.path.join(os.path.dirname(__file__), "..")


def main() -> None:
    bench = make_benchmark(n_per_suite=40, seed=SEED)
    print(f"benchmark {bench.name}: {len(bench.feedback)} feedback / {len(bench.heldout)} held-out tasks")

    grid = [
        ("none", FeedbackLevel.SCORE),
        ("score_feedback", FeedbackLevel.SCORE),
        ("fewshot_accumulation", FeedbackLevel.CORRECTIONS),
        ("self_reflection", FeedbackLevel.CORRECTIONS),
        ("prompt_rewrite", FeedbackLevel.CORRECTIONS),
        ("weight_update", FeedbackLevel.CORRECTIONS),
    ]
    trajs = {}
    for name, level in grid:
        model = SimulatedLearner(noise=0.02, seed=SEED)  # fresh model: weights are per-run
        trajs[name] = run_rsi(model, bench, get_mechanism(name), iterations=ITERATIONS,
                              feedback_level=level, feedback_budget=BUDGET, seed=SEED)
    control = trajs["none"]
    main_rows = rank(summarize(t, control) for t in trajs.values())

    # Ablation A: feedback richness for self-reflection
    abl_fb = []
    for level in (FeedbackLevel.ERRORS, FeedbackLevel.CORRECTIONS):
        t = run_rsi(SimulatedLearner(noise=0.02, seed=SEED), bench, get_mechanism("self_reflection"),
                    iterations=ITERATIONS, feedback_level=level, feedback_budget=BUDGET, seed=SEED)
        abl_fb.append(summarize(t, control))

    # Ablation B: context window for few-shot accumulation
    abl_ctx = []
    for chars in (1500, 6000, 50000):
        t = run_rsi(SimulatedLearner(name=f"sim-ctx{chars}", context_chars=chars, noise=0.02, seed=SEED),
                    bench, get_mechanism("fewshot_accumulation", max_examples=200), iterations=ITERATIONS,
                    feedback_level=FeedbackLevel.CORRECTIONS, feedback_budget=BUDGET, seed=SEED)
        abl_ctx.append(summarize(t, control))

    # Ablation C: feedback budget for weight updates
    abl_budget = []
    for budget in (2, 10, None):
        t = run_rsi(SimulatedLearner(name=f"sim-budget{budget or 'all'}", noise=0.02, seed=SEED), bench,
                    get_mechanism("weight_update"), iterations=ITERATIONS,
                    feedback_level=FeedbackLevel.CORRECTIONS, feedback_budget=budget, seed=SEED)
        abl_budget.append(summarize(t, control))

    curves = "\n".join(
        f"| {n} | " + " | ".join(f"{s:.3f}" for s in t.heldout_scores) + " |" for n, t in trajs.items()
    )
    header = "| Mechanism | " + " | ".join(f"I={i}" for i in range(ITERATIONS + 1)) + " |"
    sep = "|---|" + "---|" * (ITERATIONS + 1)
    suites = sorted(trajs["none"].iterations[0].heldout_by_suite)
    suite_rows = "\n".join(
        f"| {n} | " + " | ".join(
            f"{t.iterations[0].heldout_by_suite[s]:.2f} → {t.iterations[-1].heldout_by_suite[s]:.2f}" for s in suites
        ) + " |"
        for n, t in trajs.items()
    )
    frontier = frontier_items(list(trajs.values()))

    md = f"""# RSI-Bench harness validation (SimulatedLearner)

These numbers validate the harness and metrics against a deterministic toy
learner whose learning rules are known. They are **not** results about any
real model. Reproduce with `python scripts/run_demo.py`.

Setup: {len(bench.feedback)} feedback / {len(bench.heldout)} held-out tasks, {ITERATIONS} iterations,
at most {BUDGET} corrected errors revealed per round, 2% answer noise, seed {SEED}.

## Main grid (ranked)

{markdown_table(main_rows)}

Gain vs. no-improvement control: """ + ", ".join(
        f"{s.mechanism} {s.gain_vs_control:+.3f}" for s in main_rows) + f"""

## Held-out score trajectories

{header}
{sep}
{curves}

## Per-suite held-out score (I=0 → I={ITERATIONS})

| Mechanism | {" | ".join(suites)} |
|---|{"---|" * len(suites)}
{suite_rows}

## Ablation A: feedback richness (self_reflection)

{markdown_table(abl_fb)}

## Ablation B: context window (fewshot_accumulation, max 200 examples)

{markdown_table(abl_ctx)}

## Ablation C: label budget per round (weight_update)

{markdown_table(abl_budget)}

## Frontier

{len(frontier)} held-out items were never solved by any mechanism at any iteration
(suites: {", ".join(sorted({f.split("-")[0] for f in frontier})) or "none"}).
"""
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "sim_results.md"), "w") as f:
        f.write(md)
    with open(os.path.join(ROOT, "results", "sim_trajectories.json"), "w") as f:
        json.dump({n: t.to_dict() for n, t in trajs.items()}, f, indent=1)
    print(md)


if __name__ == "__main__":
    main()
