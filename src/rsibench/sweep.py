"""Multi-model, multi-mechanism, multi-seed sweeps with cross-seed aggregation.

A single RSI run is one sample. Each seed draws a fresh benchmark: a
different feedback/held-out split and, for the headroom suites, different
hidden rules. Results are reported as mean ± 95% CI across seeds, and the
RSI level as the modal level with its vote count.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .core import AgentState, FeedbackLevel, Model
from .data import ALL_SUITES, CORE_SUITES, HEADROOM_SUITES, make_benchmark
from .evaluate import RSILevel, Summary, summarize
from .mechanisms import MECHANISMS, PRESETS, get_mechanism
from .runner import Trajectory, run_rsi

SUITE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "core": CORE_SUITES,
    "headroom": HEADROOM_SUITES,
    "all": ALL_SUITES,
}

# Two-sided 97.5% Student-t quantiles for small n (df = n - 1).
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
         10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def resolve_suites(spec: str) -> Tuple[str, ...]:
    return SUITE_ALIASES.get(spec, tuple(s for s in spec.split(",") if s))


def default_feedback(mechanism: str) -> FeedbackLevel:
    """Score-only mechanisms get SCORE; everything else gets full corrections."""
    cls = PRESETS[mechanism][0] if mechanism in PRESETS else MECHANISMS[mechanism]
    return FeedbackLevel.SCORE if cls.requires == FeedbackLevel.SCORE else FeedbackLevel.CORRECTIONS


def mean_ci95(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and half-width of a t-based 95% CI (half-width 0 for n < 2)."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    m = statistics.mean(vals)
    if len(vals) < 2:
        return m, 0.0
    df = len(vals) - 1
    t = _T975.get(df) or _T975[max(k for k in _T975 if k <= df)]
    return m, t * statistics.stdev(vals) / math.sqrt(len(vals))


@dataclass
class SeedAggregate:
    model: str
    mechanism: str
    channel: str
    feedback_level: str
    n_seeds: int
    s0: Tuple[float, float]
    sT: Tuple[float, float]
    eta: Tuple[float, float]
    gen_gap: Tuple[float, float]
    forgetting: Tuple[float, float]
    alignment_drift: Tuple[float, float]
    alignment_worst: Tuple[float, float]
    gain_vs_control: Tuple[float, float]
    gain_per_mchar: Tuple[float, float]
    level: str
    level_votes: Dict[str, int] = field(default_factory=dict)
    call_errors: int = 0


def aggregate(summaries: Iterable[Summary]) -> List[SeedAggregate]:
    groups: Dict[Tuple[str, str, str], List[Summary]] = {}
    for s in summaries:
        groups.setdefault((s.model, s.mechanism, s.feedback_level), []).append(s)
    out = []
    for (model, mech, fb), ss in groups.items():
        votes = Counter(s.level for s in ss)
        order = {lvl.name: lvl.value for lvl in RSILevel}
        # Modal level; ties broken toward the *lower* level (conservative).
        level = sorted(votes.items(), key=lambda kv: (-kv[1], order[kv[0]]))[0][0]
        out.append(SeedAggregate(
            model=model, mechanism=mech, channel=ss[0].channel, feedback_level=fb, n_seeds=len(ss),
            s0=mean_ci95([s.s0 for s in ss]),
            sT=mean_ci95([s.sT for s in ss]),
            eta=mean_ci95([s.eta for s in ss]),
            gen_gap=mean_ci95([s.generalization_gap for s in ss]),
            forgetting=mean_ci95([s.max_forgetting for s in ss]),
            alignment_drift=mean_ci95([s.alignment_drift for s in ss]),
            alignment_worst=mean_ci95([s.alignment_worst_drop for s in ss]),
            gain_vs_control=mean_ci95([s.gain_vs_control for s in ss]),
            gain_per_mchar=mean_ci95([s.gain_per_mchar for s in ss]),
            level=level,
            level_votes=dict(votes),
            call_errors=sum(s.call_errors for s in ss),
        ))
    order = {lvl.name: lvl.value for lvl in RSILevel}
    return sorted(out, key=lambda a: (a.model, -order[a.level], -a.eta[0]))


def _fmt(pair: Tuple[float, float], signed: bool = False) -> str:
    m, h = pair
    if math.isnan(m):
        return "—"
    core = f"{m:+.3f}" if signed else f"{m:.3f}"
    return f"{core} ± {h:.3f}" if h else core


def aggregate_table(aggs: Sequence[SeedAggregate]) -> str:
    head = (
        "| Model | Mechanism | Feedback | Seeds | S0 | ST | η | Gain vs ctrl | GenGap | Forget | AlignΔ "
        "| AlignWorst | Gain / M chars | Level (votes) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for a in aggs:
        votes = ", ".join(f"{k} {v}" for k, v in sorted(a.level_votes.items()))
        gpm = "—" if math.isnan(a.gain_per_mchar[0]) else f"{a.gain_per_mchar[0]:+.2f}"
        rows.append(
            f"| {a.model} | {a.mechanism} | {a.feedback_level} | {a.n_seeds} | {_fmt(a.s0)} | {_fmt(a.sT)} | "
            f"{_fmt(a.eta, True)} | {_fmt(a.gain_vs_control, True)} | {_fmt(a.gen_gap, True)} | "
            f"{_fmt(a.forgetting)} | {_fmt(a.alignment_drift, True)} | {_fmt(a.alignment_worst)} | {gpm} | {a.level} ({votes}) |"
        )
    return "\n".join([head, *rows])


def run_sweep(
    model_factories: Dict[str, Callable[[], Model]],
    mechanisms: Sequence[str],
    seeds: Sequence[int] = (0, 1, 2),
    suites: Sequence[str] = CORE_SUITES,
    n_per_suite: int = 40,
    iterations: int = 5,
    feedback_budget: Optional[int] = 10,
    feedback: Optional[FeedbackLevel] = None,
    out_dir: Optional[str | Path] = None,
    max_workers: int = 1,
    initial_state: Optional[AgentState] = None,
    verbose: bool = False,
) -> Tuple[List[Trajectory], List[SeedAggregate]]:
    """Run every (model, mechanism, seed) cell, with a ``none`` control per (model, seed).

    ``model_factories`` must build a *fresh* model per call so weight-channel
    runs never leak parameters across cells. With ``out_dir`` set, each cell
    checkpoints after every iteration and completed cells are skipped on
    re-run, so an interrupted paid sweep can be resumed.
    """
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    mechs = list(dict.fromkeys(["none", *mechanisms]))
    trajs: List[Trajectory] = []
    summaries: List[Summary] = []
    for model_key, factory in model_factories.items():
        safe = model_key.replace("/", "_").replace(":", "_")
        for seed in seeds:
            bench = make_benchmark(suites=suites, n_per_suite=n_per_suite, seed=seed)
            control: Optional[Trajectory] = None
            for mech in mechs:
                stem = f"{safe}__{mech}__seed{seed}"
                final = out / f"{stem}.json" if out else None
                if final and final.exists():
                    traj = Trajectory.load_json(final)
                else:
                    level = feedback if (feedback and mech != "none") else default_feedback(mech)
                    traj = run_rsi(
                        factory(), bench, get_mechanism(mech), iterations=iterations,
                        feedback_level=level, feedback_budget=feedback_budget, seed=seed,
                        max_workers=max_workers, initial_state=initial_state,
                        checkpoint=(out / f"{stem}.ckpt.json") if out else None,
                        verbose=verbose,
                    )
                    if final:
                        traj.save_json(final)
                        ck = out / f"{stem}.ckpt.json"
                        if ck.exists():
                            ck.unlink()
                if mech == "none":
                    control = traj
                trajs.append(traj)
                summaries.append(summarize(traj, control))
    return trajs, aggregate(summaries)
