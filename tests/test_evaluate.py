from __future__ import annotations

import math

import pytest

from rsibench import (
    FeedbackLevel,
    FewShotAccumulation,
    NoImprovement,
    RSILevel,
    SimulatedLearner,
    WeightUpdate,
    make_benchmark,
    run_rsi,
)
from rsibench.evaluate import (
    acceleration,
    auic,
    eta,
    fit_saturating,
    forgetting_rate,
    frontier_items,
    generalization_gap,
    markdown_table,
    monotonicity,
    newly_solved,
    normalized_gain,
    paired_bootstrap_ci,
    plateau_iteration,
    rank,
    rsi_level,
    sign_test_p,
    step_gains,
    summarize,
)
from rsibench.runner import IterationResult, Trajectory


def test_curve_metrics() -> None:
    s = [0.2, 0.4, 0.5, 0.55]
    assert step_gains(s) == pytest.approx([0.2, 0.1, 0.05])
    assert eta(s) == pytest.approx(0.35 / 3)
    assert normalized_gain(s) == pytest.approx(0.35 / 0.8)
    assert monotonicity(s) == 1.0
    assert acceleration(s) < 0  # diminishing returns
    assert acceleration([0.1, 0.15, 0.25, 0.45]) > 0
    assert auic(s) == pytest.approx((0.2 + 0.3 + 0.35) / 3)


def test_edge_cases() -> None:
    assert eta([0.5]) == 0.0
    assert normalized_gain([1.0, 1.0]) == 0.0
    with pytest.raises(ValueError):
        eta([0.5, 1.5])


def test_plateau() -> None:
    assert plateau_iteration([0.1, 0.5, 0.5, 0.5]) == 1
    assert plateau_iteration([0.1, 0.2, 0.3]) is None


def test_fit_recovers_known_curve() -> None:
    s = [0.9 - 0.7 * math.exp(-0.5 * i) for i in range(8)]
    s_inf, k, rmse = fit_saturating(s)
    assert s_inf == pytest.approx(0.9, abs=1e-3)
    assert k == pytest.approx(0.5, abs=0.01)
    assert rmse < 1e-3


def test_item_metrics() -> None:
    before = {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0}
    after = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 1.0}
    assert forgetting_rate(before, after) == 0.5
    assert newly_solved(before, after) == ["c", "d"]
    mean, lo, hi = paired_bootstrap_ci(before, after)
    assert mean == pytest.approx(0.25) and lo <= mean <= hi
    assert sign_test_p(before, before) == 1.0
    big_b = {str(i): 0.0 for i in range(30)}
    big_a = {str(i): 1.0 for i in range(30)}
    assert sign_test_p(big_b, big_a) < 1e-6


def _traj(scores, per_item=None) -> Trajectory:
    its = [IterationResult(iteration=i, feedback_score=s, heldout_score=s,
                           heldout_per_task=None if per_item is None else per_item[i]) for i, s in enumerate(scores)]
    return Trajectory(model="m", benchmark="b", mechanism={"name": "x", "channel": "prompt"},
                      feedback_level="corrections", iterations=its)


def test_levels_from_aggregate_curves() -> None:
    assert rsi_level(_traj([0.3, 0.3, 0.3])) == RSILevel.NONE
    assert rsi_level(_traj([0.3, 0.6, 0.6, 0.6])) == RSILevel.ONE_SHOT
    assert rsi_level(_traj([0.3, 0.5, 0.6, 0.65])) == RSILevel.SUSTAINED
    assert rsi_level(_traj([0.1, 0.15, 0.25, 0.45])) == RSILevel.ACCELERATING


def test_end_to_end_summary_and_ranking() -> None:
    b = make_benchmark(n_per_suite=20)
    ctrl = run_rsi(SimulatedLearner(), b, NoImprovement(), iterations=3, feedback_level=FeedbackLevel.SCORE)
    few = run_rsi(SimulatedLearner(), b, FewShotAccumulation(), iterations=3, feedback_budget=5)
    w = run_rsi(SimulatedLearner(), b, WeightUpdate(), iterations=3)
    sums = [summarize(t, ctrl) for t in (ctrl, few, w)]
    assert sums[0].level == "NONE" and sums[0].gain_vs_control == 0.0
    assert sums[1].level in {"SUSTAINED", "ACCELERATING"}
    assert sums[2].level == "ONE_SHOT"  # all corrections at once -> everything learned in round 1
    assert sums[1].ci_low is not None and sums[1].ci_low > 0
    assert rank(sums)[-1].mechanism == "none"
    # lookup facts are memorised on the feedback split but cannot transfer
    assert generalization_gap(w) > 0
    assert sums[1].alignment_drift is not None and sums[1].alignment_drift >= 0
    frontier = frontier_items([ctrl, few, w])
    assert frontier and all(i.split("/")[-1].startswith(("cipher", "lookup")) for i in frontier)
    assert all(i.startswith("rsibench-core-v0/seed0/") for i in frontier)
    assert markdown_table(sums).count("\n") == 4


def test_trajectory_roundtrip(tmp_path) -> None:
    b = make_benchmark(n_per_suite=8)
    t = run_rsi(SimulatedLearner(), b, FewShotAccumulation(), iterations=2)
    p = tmp_path / "t.json"
    t.save_json(p)
    t2 = Trajectory.load_json(p)
    assert t2.heldout_scores == t.heldout_scores
    assert t2.iterations[1].heldout_per_task == t.iterations[1].heldout_per_task


def test_relative_acceleration_ignores_ceiling() -> None:
    from rsibench.evaluate import closure_rates, relative_acceleration
    # constant-rate learner: closes half the remaining headroom each round
    s = [0.0]
    for _ in range(6):
        s.append(s[-1] + 0.5 * (1 - s[-1]))
    assert all(r == pytest.approx(0.5) for r in closure_rates(s))
    assert relative_acceleration(s) == pytest.approx(0.0, abs=1e-9)
    # compounding learner that saturates: raw acceleration < 0, relative > 0
    comp = [0.0, 0.05, 0.15, 0.4, 0.8, 0.95, 0.99]
    assert acceleration(comp) < 0  # masked by the ceiling
    assert relative_acceleration(comp) > 0
