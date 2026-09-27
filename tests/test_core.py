from __future__ import annotations

import pytest

from rsibench.core import AgentState, Benchmark, Task, Track, score_constraints
from rsibench.data import load_jsonl, make_benchmark, make_tasks, save_jsonl


def test_exact_and_numeric_scoring() -> None:
    assert Task(id="a", input="x", reference="Paris").score(" paris. ") == 1.0
    assert Task(id="b", input="x", reference="42", scorer="numeric").score("The answer is 42") == 1.0
    assert Task(id="c", input="x", reference="42", scorer="numeric").score("no number") == 0.0


def test_constraint_scoring_partial_credit() -> None:
    t = Task(id="c", input="x", reference="", scorer="constraints",
             metadata={"constraints": {"uppercase": True, "max_words": 2}})
    assert score_constraints("HELLO", t) == 1.0
    assert score_constraints("hello", t) == 0.5
    assert score_constraints("hello there world", t) == 0.0


def test_unknown_constraint_raises() -> None:
    t = Task(id="c", input="x", reference="", scorer="constraints", metadata={"constraints": {"bogus": 1}})
    with pytest.raises(ValueError):
        t.score("x")


def test_missing_reference_raises() -> None:
    with pytest.raises(ValueError):
        Task(id="a", input="x", reference=None).score("y")


def test_split_is_stratified_and_disjoint() -> None:
    b = make_benchmark(n_per_suite=10)
    fb = {t.id for t in b.feedback}
    ho = {t.id for t in b.heldout}
    assert not fb & ho
    assert {t.suite for t in b.feedback} == {t.suite for t in b.heldout} == set(b.suites)


def test_duplicate_ids_rejected() -> None:
    t = Task(id="x", input="a", reference="b")
    with pytest.raises(ValueError):
        Benchmark(name="n", feedback=[t], heldout=[t])


def test_render_contains_state() -> None:
    s = AgentState(examples=[("Encode: ab", "cd")], notes=["MAP a=c"], history=["Iteration 0: score 0.1"])
    p = s.render(Task(id="t", input="Encode: ba", reference="dc"))
    assert "Worked examples" in p and "MAP a=c" in p and p.rstrip().endswith("Output:")
    assert s.copy().examples is not s.examples


def test_instruction_suite_is_alignment_track() -> None:
    tasks = make_tasks(["instruction"], n_per_suite=8)
    assert all(t.track == Track.ALIGNMENT for t in tasks)
    assert all(t.score(t.reference) == 1.0 for t in tasks)  # references are compliant


def test_all_references_self_score(tmp_path) -> None:
    tasks = make_tasks(n_per_suite=12)
    assert all(t.score(t.reference) == 1.0 for t in tasks)
    p = tmp_path / "tasks.jsonl"
    save_jsonl(tasks, p)
    loaded = load_jsonl(p)
    assert [t.id for t in loaded] == [t.id for t in tasks]
    assert loaded[-1].metadata == tasks[-1].metadata
