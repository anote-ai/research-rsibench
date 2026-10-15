from __future__ import annotations

import pytest

from rsibench import (
    EchoModel,
    FeedbackLevel,
    FewShotAccumulation,
    NoImprovement,
    PromptRewrite,
    ScoreFeedback,
    SelfReflection,
    SimulatedLearner,
    WeightUpdate,
    make_benchmark,
    run_rsi,
)
from rsibench.core import AgentState, Feedback, FeedbackItem
from rsibench.mechanisms import get_mechanism


@pytest.fixture(scope="module")
def bench():
    return make_benchmark(n_per_suite=20)


def test_baseline_is_iteration_zero(bench) -> None:
    t = run_rsi(SimulatedLearner(), bench, NoImprovement(), iterations=3, feedback_level=FeedbackLevel.SCORE)
    assert [it.iteration for it in t.iterations] == [0, 1, 2, 3]
    assert len(set(t.heldout_scores)) == 1  # deterministic model, nothing changes


def test_score_only_feedback_does_not_help_simulated_learner(bench) -> None:
    t = run_rsi(SimulatedLearner(), bench, ScoreFeedback(), iterations=3, feedback_level=FeedbackLevel.SCORE)
    assert t.heldout_scores[-1] == t.heldout_scores[0]


@pytest.mark.parametrize("mech", [FewShotAccumulation(), SelfReflection(), PromptRewrite()])
def test_prompt_and_memory_mechanisms_improve(bench, mech) -> None:
    t = run_rsi(SimulatedLearner(), bench, mech, iterations=3)
    assert t.heldout_scores[-1] > t.heldout_scores[0]


def test_weight_update_needs_trainable_model(bench) -> None:
    with pytest.raises(TypeError):
        run_rsi(EchoModel(), bench, WeightUpdate(), iterations=1)
    t = run_rsi(SimulatedLearner(), bench, WeightUpdate(), iterations=2)
    assert t.heldout_scores[-1] > t.heldout_scores[0]


def test_incompatible_feedback_level_rejected(bench) -> None:
    with pytest.raises(ValueError):
        run_rsi(SimulatedLearner(), bench, FewShotAccumulation(), feedback_level=FeedbackLevel.SCORE)


def test_fewshot_respects_capacity() -> None:
    fb = Feedback(iteration=0, level=FeedbackLevel.CORRECTIONS, aggregate_score=0.0,
                  items=[FeedbackItem(task_id=str(i), input=f"q{i}", output="", score=0.0, reference="a")
                         for i in range(10)])
    s = FewShotAccumulation(max_examples=4).improve(AgentState(), fb, EchoModel())
    assert len(s.examples) == 4 and s.examples[-1][0] == "q9" and s.iteration == 1


def test_errors_only_reflection_cannot_see_references(bench) -> None:
    t = run_rsi(SimulatedLearner(), bench, SelfReflection(), iterations=3, feedback_level=FeedbackLevel.ERRORS)
    last = t.iterations[-1].heldout_by_suite
    assert last["cipher"] == 0.0 and last["instruction"] == 1.0


def test_feedback_budget_slows_learning(bench) -> None:
    full = run_rsi(SimulatedLearner(), bench, WeightUpdate(), iterations=1)
    small = run_rsi(SimulatedLearner(), bench, WeightUpdate(), iterations=1, feedback_budget=1)
    assert small.heldout_scores[1] < full.heldout_scores[1]


def test_model_calls_counted(bench) -> None:
    t = run_rsi(SimulatedLearner(), bench, SelfReflection(), iterations=1)
    n = len(bench.feedback) + len(bench.heldout)
    assert t.iterations[0].model_calls == n + 1  # + one reflection call
    assert t.iterations[1].model_calls == n


def test_registry() -> None:
    assert get_mechanism("self_reflection").self_authored
    with pytest.raises(KeyError):
        get_mechanism("nope")
