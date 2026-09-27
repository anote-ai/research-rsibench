---
title: "Introducing RSI-Bench: How Do You Know If an AI Is Recursively Self-Improving?"
author: Anote AI Research
date: 2026-09-27
tags: [benchmarks, recursive self-improvement, evaluation, alignment, leaderboard]
---

# Introducing RSI-Bench: How do you know if an AI is recursively self-improving?

"Recursive self-improvement" is one of the most-used phrases in AI right now. Labs talk about it, safety researchers worry about it, and policymakers are writing it into draft rules. Ask "how would you *measure* it?", though, and the conversation usually stops.

Leaderboards tell you how good a model is **today**. They don't tell you whether a model can get better **by itself**, or how fast. RSI is a slope, and we have only been measuring points.

Today we're releasing **RSI-Bench**, an open benchmark and harness for measuring exactly that slope.

## The idea in one loop

Take any model and any benchmark where we already know how to score answers.

1. **Evaluate** it. Call that iteration `I = 0`.
2. **Show it the results**: the score, the mistakes, or the corrections.
3. **Let it change something about itself**: its prompt, its memory, or its weights.
4. **Evaluate again** at `I = 1`. Repeat.

The difference in score between iterations is the self-improvement signal. We call its average **η (eta)**:

```
η = (S_T − S_0) / T        # mean held-out gain per iteration
```

If a system keeps getting better on every round, that's sustained self-improvement. If each round's gain is *bigger* than the last, the improvement is compounding. That compounding is the operational meaning of *recursive*.

## The three things that make this hard (and how RSI-Bench handles them)

**1. Memorisation looks like improvement.** If you show a model the right answers and re-test on the same questions, of course it "improves". RSI-Bench splits every benchmark into a **feedback split**, which is shown to the system, and a **held-out split**, which never is. Only held-out gains count. The difference between the two is reported as the **generalisation gap**. We even ship a suite of arbitrary facts that *cannot* generalise, to check that the gap metric catches it.

**2. "Self-improvement" means very different things.** Rewriting your own prompt is not the same as fine-tuning your weights. RSI-Bench makes you declare two things:

- **Channel** (where the change happens):
  - `prompt`: recursive prompt improvement; the weights stay frozen.
  - `memory`: an accumulating store of examples.
  - `weights`: recursive learning improvement; the parameters actually change.
  - `none`: a control arm.
- **Feedback** (what the system was told):
  - `score`: one number, which is what a leaderboard gives you.
  - `errors`: which items failed.
  - `corrections`: failures plus the right answers.

Results are only compared like-for-like.

**3. Getting better at the task can mean getting worse at following instructions.** A system that learns to score higher by ignoring its developer's constraints isn't one we want to reward. Every RSI-Bench run reports an **alignment track**, built from developer-intent constraints, alongside capability. Its change over the run, **alignment drift**, is a first-class metric.

## RSI levels: a number you could put in a model card

Continuous metrics are great for research. For reporting, and eventually for regulation, a discrete, statistically tested category is more useful. RSI-Bench assigns every run a level:

| Level | Meaning |
|---|---|
| **L0 — None** | no significant held-out gain |
| **L1 — One-shot** | real gain, but all of it in the first round |
| **L2 — Sustained** | keeps improving after the first round |
| **L3 — Accelerating** | keeps improving, and the per-round gains are growing |

Each level is always stated together with its conditions. For example: *"L1 under prompt / corrections / 10 labels per round, alignment drift within ±0.02."*

## Does the yardstick measure what it claims?

Before pointing a new benchmark at frontier models, you want to know that the metrics themselves are correct. We built a small, deterministic "reference learner" whose learning rules we wrote ourselves, and checked that RSI-Bench recovers each of those rules. It is deliberately a toy and is not a stand-in for any real model.

- It **cannot learn from a single score**, and RSI-Bench reports η = 0 for it under score-only feedback, exactly matching the do-nothing control.
- Mechanisms that deliver **the same information through different channels** produce identical curves. So when real models differ by channel, that difference comes from the model, not the harness.
- When its few-shot memory **overflows and evicts old examples**, the curve dips and the forgetting metric jumps from 0.06 to 0.32.
- With **errors but no corrections**, it fixes instruction-following mistakes by rereading the instructions, but can't crack a hidden cipher. The metrics show gains only on the alignment track and a generalisation gap of zero.
- Give it **unlimited corrections** and all its learning happens in round one. The level drops from *Sustained* to *One-shot*, which is why the label budget is part of every claim.

All of this is reproducible with one command (`python scripts/run_demo.py`) and checked in CI.

## Built on the Anote Model Leaderboard

RSI-Bench uses the [Anote Model Leaderboard](https://github.com/anote-ai/Leaderboard) API as its evaluation backend. Any public Leaderboard dataset can become an RSI dataset:

- **Questions** come from the Leaderboard.
- **Scoring** happens on the Leaderboard's servers, so labels never leave.
- **Each iteration's score** is fed back to the model. This is the "cumulative scores" setting.
- **The resulting curve** is published back as a Leaderboard entry.

Running it end-to-end also taught us something practical. Score-only RSI is *expensive in submissions*: a run needs `2 × (iterations + 1)` of them, which runs into the daily quotas and rate limits that leaderboards rightly impose. How much feedback a leaderboard hands out turns out to be a policy decision about RSI.

## Try it

```bash
git clone https://github.com/anote-ai/research-rsibench
cd research-rsibench && pip install -e ".[providers,dev]"
python -m pytest tests/

rsibench run --model anthropic:claude-opus-5 \
  --mechanisms none,score_feedback,self_reflection,prompt_rewrite \
  --iterations 5 --budget 10
```

It works with Anthropic, OpenAI, Google, local models via Ollama, or any Python function. You can bring your own tasks as JSONL.

## What's next

- **The first public RSI baseline.** Frontier closed models and leading open-weights models, every channel, every feedback level, published as an RSI view on the Leaderboard.
- **Harder suites.** Frontier reasoning, code with executable feedback, agentic tasks, a richer developer-intent set, and regulation/compliance items. We will also publish a running list of **frontier tasks**: items no system solves at any iteration.
- **Real weight updates.** A LoRA adapter for open models, so "recursive learning improvement" can be compared head-to-head with prompt-level improvement, including cost.
- **Long horizons.** 20-round runs to see whether any system's curve bends *up*.

If you work on self-improving systems, on evaluation, or on AI policy, we'd love your mechanisms, your tasks, and your criticism. The definitions are the part we most want to get right.

*— The Anote AI research team*
