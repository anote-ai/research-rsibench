"""Tests for v0.2: headroom suites, robust runner, drift/acceleration validation, sweeps."""
from __future__ import annotations

import pytest

from rsibench import (
    AgentState,
    FeedbackLevel,
    RSILevel,
    SelfReflection,
    SimulatedLearner,
    WeightUpdate,
    make_benchmark,
    rsi_level,
    run_rsi,
    summarize,
)
from rsibench.core import DEFAULT_INSTRUCTIONS, Task, parse_int_list
from rsibench.data import (
    ALL_SUITES,
    CORE_SUITES,
    HEADROOM_SUITES,
    hidden_op_for,
    hidden_program_for,
    make_tasks,
)
from rsibench.evaluate import alignment_worst_drop
from rsibench.mechanisms import SELF_SECTION, NoImprovement, PromptRewrite, get_mechanism
from rsibench.models import DEVELOPER_POLICY_PHRASE
from rsibench.runner import IterationResult, Trajectory
from rsibench.sweep import aggregate, mean_ci95, resolve_suites, run_sweep


# -- suites ------------------------------------------------------------------

def test_headroom_references_self_score_and_rules_vary_by_seed() -> None:
    tasks = make_tasks(HEADROOM_SUITES, n_per_suite=10, seed=3)
    assert all(t.score(t.reference) == 1.0 for t in tasks)
    assert len({hidden_op_for(s) for s in range(20)}) > 1
    assert len({hidden_program_for(s) for s in range(20)}) > 1


def test_hidden_rule_never_reaches_the_prompt() -> None:
    for t in make_tasks(HEADROOM_SUITES, n_per_suite=5):
        prompt = AgentState().render(t)
        assert "rule" not in prompt and (t.metadata.get("rule") or "@@") not in prompt


def test_int_list_scorer() -> None:
    t = Task(id="x", input="", reference="[1, 2, 3]", scorer="int_list")
    assert t.score("The answer is [1,2,3]") == 1.0
    assert t.score("[3, 2, 1]") == 0.0
    assert parse_int_list("nothing") is None and parse_int_list("[]") == []


def test_split_is_stratified_by_declared_stratum() -> None:
    for seed in range(10):
        b = make_benchmark(("instruction",), n_per_suite=20, seed=seed)
        assert {t.metadata["tag"] for t in b.feedback} == {t.metadata["tag"] for t in b.heldout}


def test_suite_aliases() -> None:
    assert resolve_suites("core") == CORE_SUITES and resolve_suites("all") == ALL_SUITES
    assert resolve_suites("cipher,listfn") == ("cipher", "listfn")


# -- learner on headroom suites ---------------------------------------------

def test_learner_induces_hidden_operator_and_program() -> None:
    b = make_benchmark(HEADROOM_SUITES, n_per_suite=20, seed=1)
    t = run_rsi(SimulatedLearner(), b, SelfReflection(), iterations=3)
    assert t.iterations[0].heldout_by_suite["operator"] < 0.5
    assert t.iterations[-1].heldout_by_suite["operator"] == 1.0
    assert t.iterations[-1].heldout_by_suite["listfn"] == 1.0
    assert t.iterations[-1].heldout_by_suite["policy"] == 1.0


def test_policy_needs_corrections() -> None:
    b = make_benchmark(("policy",), n_per_suite=20)
    t = run_rsi(SimulatedLearner(), b, SelfReflection(), iterations=2, feedback_level=FeedbackLevel.ERRORS)
    assert t.heldout_scores[-1] == t.heldout_scores[0]


# -- runner robustness ---------------------------------------------------------

class _Flaky:
    name = "flaky"

    def __init__(self, fail_every: int = 3) -> None:
        self.inner = SimulatedLearner()
        self.n = 0
        self.fail_every = fail_every

    def generate(self, prompt: str) -> str:
        self.n += 1
        if self.n % self.fail_every == 0:
            raise ConnectionError("simulated outage")
        return self.inner.generate(prompt)


def test_call_errors_are_recorded_not_fatal() -> None:
    b = make_benchmark(n_per_suite=6)
    t = run_rsi(_Flaky(), b, SelfReflection(), iterations=2)
    assert sum(it.call_errors for it in t.iterations) > 0
    assert summarize(t).call_errors == sum(it.call_errors for it in t.iterations)


def test_parallel_matches_serial() -> None:
    b = make_benchmark(ALL_SUITES, n_per_suite=6)
    a = run_rsi(SimulatedLearner(), b, SelfReflection(), iterations=2)
    p = run_rsi(SimulatedLearner(), b, SelfReflection(), iterations=2, max_workers=8)
    assert a.heldout_scores == p.heldout_scores
    assert a.iterations[1].prompt_chars == p.iterations[1].prompt_chars > 0


class _Crash(BaseException):
    pass


class _CrashingLearner(SimulatedLearner):
    def __init__(self, crash_at: int) -> None:
        super().__init__()
        self.calls, self.crash_at = 0, crash_at

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == self.crash_at:
            raise _Crash()
        return super().generate(prompt)


def test_checkpoint_resume_reproduces_uninterrupted_run(tmp_path) -> None:
    b = make_benchmark(ALL_SUITES, n_per_suite=6)
    kw = dict(iterations=3, feedback_budget=5, seed=1)
    ref = run_rsi(SimulatedLearner(), b, SelfReflection(), **kw)
    ck = tmp_path / "run.ckpt.json"
    n_per_iter = len(b.feedback) + len(b.heldout) + 1
    with pytest.raises(_Crash):
        run_rsi(_CrashingLearner(crash_at=2 * n_per_iter + 3), b, SelfReflection(), checkpoint=ck, **kw)
    resumed = run_rsi(SimulatedLearner(), b, SelfReflection(), checkpoint=ck, **kw)
    assert resumed.heldout_scores == ref.heldout_scores
    assert [it.iteration for it in resumed.iterations] == [0, 1, 2, 3]


def test_checkpoint_rejects_other_config_and_weight_resume(tmp_path) -> None:
    b = make_benchmark(n_per_suite=6)
    ck = tmp_path / "c.json"
    run_rsi(SimulatedLearner(), b, SelfReflection(), iterations=1, checkpoint=ck)
    with pytest.raises(ValueError):
        run_rsi(SimulatedLearner(), b, SelfReflection(), iterations=2, checkpoint=ck)
    ck2 = tmp_path / "w.json"
    run_rsi(SimulatedLearner(), b, WeightUpdate(), iterations=1, checkpoint=ck2)
    with pytest.raises(ValueError):
        run_rsi(SimulatedLearner(), b, WeightUpdate(), iterations=1, checkpoint=ck2)


def test_state_snapshots_record_self_authored_notes() -> None:
    b = make_benchmark(n_per_suite=6)
    t = run_rsi(SimulatedLearner(), b, SelfReflection(), iterations=2)
    assert t.iterations[0].state["notes"] == [] and t.iterations[1].state["notes"]


# -- drift, protection, acceleration -------------------------------------------

def _dev_state() -> AgentState:
    return AgentState(instructions=f"{DEFAULT_INSTRUCTIONS}\n{DEVELOPER_POLICY_PHRASE}")


def test_unprotected_rewrite_drifts_protected_does_not() -> None:
    b = make_benchmark(CORE_SUITES, n_per_suite=20, seed=0)
    drop = run_rsi(SimulatedLearner(rewrite_keeps_instructions=False), b, PromptRewrite(),
                   iterations=3, feedback_budget=10, initial_state=_dev_state())
    safe = run_rsi(SimulatedLearner(rewrite_keeps_instructions=False), b, PromptRewrite(protect_instructions=True),
                   iterations=3, feedback_budget=10, initial_state=_dev_state())
    assert alignment_worst_drop(drop) == 1.0
    assert alignment_worst_drop(safe) == 0.0
    assert DEVELOPER_POLICY_PHRASE in safe.iterations[-1].state["instructions"]
    assert SELF_SECTION.strip() in safe.iterations[-1].state["instructions"]
    assert get_mechanism("prompt_rewrite_protected").describe()["name"] == "prompt_rewrite_protected"


def test_worst_drop_catches_what_endpoint_misses() -> None:
    its = [IterationResult(i, 0.5, 0.5, heldout_by_track={"alignment": a}) for i, a in enumerate([1.0, 0.0, 1.0])]
    t = Trajectory(model="m", benchmark="b", mechanism={"name": "x"}, feedback_level="corrections", iterations=its)
    assert summarize(t).alignment_drift == 0.0
    assert alignment_worst_drop(t) == 1.0


def test_metalearner_accelerates() -> None:
    b = make_benchmark(ALL_SUITES, n_per_suite=20, seed=1)
    t = run_rsi(SimulatedLearner(train_capacity=1, capacity_growth=2.0), b, WeightUpdate(),
                iterations=6, feedback_budget=40, seed=1)
    assert rsi_level(t) == RSILevel.ACCELERATING


def test_unlimited_budget_is_one_shot_even_with_noise() -> None:
    b = make_benchmark(ALL_SUITES, n_per_suite=20, seed=0)
    t = run_rsi(SimulatedLearner(noise=0.02), b, WeightUpdate(), iterations=6, seed=0)
    assert rsi_level(t) == RSILevel.ONE_SHOT


# -- sweeps --------------------------------------------------------------------

def test_mean_ci95() -> None:
    assert mean_ci95([0.5]) == (0.5, 0.0)
    m, h = mean_ci95([0.1, 0.2, 0.3])
    assert m == pytest.approx(0.2) and h == pytest.approx(4.303 * 0.1 / 3 ** 0.5)


def test_sweep_aggregates_and_resumes(tmp_path) -> None:
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return SimulatedLearner()

    kw = dict(seeds=(0, 1), n_per_suite=6, iterations=2, out_dir=tmp_path)
    trajs, aggs = run_sweep({"sim": factory}, ["self_reflection"], **kw)
    assert len(trajs) == 4 and {a.mechanism for a in aggs} == {"none", "self_reflection"}
    sr = next(a for a in aggs if a.mechanism == "self_reflection")
    assert sr.n_seeds == 2 and sr.eta[0] > 0 and sum(sr.level_votes.values()) == 2
    assert (tmp_path / "summary.md").exists() is False  # CLI writes summary, not the library
    before = calls["n"]
    run_sweep({"sim": factory}, ["self_reflection"], **kw)
    assert calls["n"] == before  # completed cells are loaded, not re-run


def test_aggregate_level_ties_break_low() -> None:
    b = make_benchmark(n_per_suite=6)
    t = run_rsi(SimulatedLearner(), b, NoImprovement(), iterations=1, feedback_level=FeedbackLevel.SCORE)
    s1 = summarize(t)
    s2 = summarize(t)
    s2.level = "SUSTAINED"
    assert aggregate([s1, s2])[0].level == "NONE"
