"""The RSI loop: evaluate at I=0, feed back, improve, re-evaluate, repeat."""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .core import (
    AgentState,
    Benchmark,
    EvalResult,
    Feedback,
    FeedbackItem,
    FeedbackLevel,
    Model,
    Task,
    TrainableModel,
)
from .mechanisms import ImprovementMechanism, check_compatible


@dataclass
class IterationResult:
    iteration: int
    feedback_score: float
    heldout_score: float
    heldout_per_task: Optional[Dict[str, float]] = None
    feedback_per_task: Optional[Dict[str, float]] = None
    heldout_by_suite: Dict[str, float] = field(default_factory=dict)
    heldout_by_track: Dict[str, float] = field(default_factory=dict)
    state_chars: int = 0
    n_examples: int = 0
    n_notes: int = 0
    model_calls: int = 0
    elapsed_s: float = 0.0


@dataclass
class Trajectory:
    model: str
    benchmark: str
    mechanism: Dict[str, Any]
    feedback_level: str
    iterations: List[IterationResult] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def heldout_scores(self) -> List[float]:
        return [it.heldout_score for it in self.iterations]

    @property
    def feedback_scores(self) -> List[float]:
        return [it.feedback_score for it in self.iterations]

    def track_scores(self, track: str) -> List[float]:
        return [it.heldout_by_track.get(track, float("nan")) for it in self.iterations]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trajectory":
        its = [IterationResult(**it) for it in d.get("iterations", [])]
        return cls(
            model=d["model"],
            benchmark=d["benchmark"],
            mechanism=d["mechanism"],
            feedback_level=d["feedback_level"],
            iterations=its,
            config=d.get("config", {}),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(cls, path: str | Path) -> "Trajectory":
        return cls.from_dict(json.loads(Path(path).read_text()))


class _CountingModel:
    def __init__(self, inner: Model) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", type(inner).__name__)
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return self.inner.generate(prompt)


class _CountingTrainableModel(_CountingModel):
    def train(self, examples: Sequence[Tuple[str, str]]) -> None:
        self.inner.train(examples)  # type: ignore[attr-defined]


def _wrap(model: Model) -> _CountingModel:
    if isinstance(model, TrainableModel):
        return _CountingTrainableModel(model)
    return _CountingModel(model)


def _group_mean(tasks: Sequence[Task], per: Optional[Dict[str, float]], key: str) -> Dict[str, float]:
    if per is None:
        return {}
    groups: Dict[str, List[float]] = {}
    for t in tasks:
        k = t.suite if key == "suite" else t.track.value
        groups.setdefault(k, []).append(per[t.id])
    return {k: sum(v) / len(v) for k, v in sorted(groups.items())}


def answer_all(model: Model, state: AgentState, tasks: Sequence[Task]) -> Dict[str, str]:
    return {t.id: model.generate(state.render(t)) for t in tasks}


def build_feedback(
    iteration: int,
    tasks: Sequence[Task],
    result: EvalResult,
    level: FeedbackLevel,
    budget: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Feedback:
    """Turn a scored split into the feedback a mechanism is allowed to see.

    ``budget`` caps how many *error* items are revealed, which models the
    cost of labels (active / budgeted feedback).
    """
    if level == FeedbackLevel.SCORE or result.per_task is None:
        return Feedback(iteration=iteration, level=FeedbackLevel.SCORE, aggregate_score=result.score)
    items = [
        FeedbackItem(
            task_id=t.id,
            input=t.input,
            output=result.outputs.get(t.id, ""),
            score=result.per_task[t.id],
            reference=t.reference if level == FeedbackLevel.CORRECTIONS else None,
        )
        for t in tasks
    ]
    if budget is not None:
        errors = [it for it in items if it.score < 1.0]
        correct = [it for it in items if it.score >= 1.0]
        (rng or random.Random(0)).shuffle(errors)
        items = correct + errors[:budget]
    return Feedback(iteration=iteration, level=level, aggregate_score=result.score, items=items)


def run_rsi(
    model: Model,
    benchmark: Benchmark,
    mechanism: ImprovementMechanism,
    iterations: int = 5,
    feedback_level: FeedbackLevel = FeedbackLevel.CORRECTIONS,
    feedback_budget: Optional[int] = None,
    initial_state: Optional[AgentState] = None,
    seed: int = 0,
    verbose: bool = False,
) -> Trajectory:
    """Run ``iterations`` rounds of improvement and return the full trajectory.

    The returned trajectory has ``iterations + 1`` entries; entry 0 is the
    baseline ``I_0`` before any feedback.
    """
    if iterations < 0:
        raise ValueError("iterations must be >= 0")
    problem = check_compatible(mechanism, feedback_level)
    if problem:
        raise ValueError(problem)

    counted = _wrap(model)
    rng = random.Random(seed)
    state = (initial_state or AgentState()).copy()
    traj = Trajectory(
        model=counted.name,
        benchmark=benchmark.name,
        mechanism=mechanism.describe(),
        feedback_level=feedback_level.value,
        config={
            "iterations": iterations,
            "feedback_budget": feedback_budget,
            "seed": seed,
            "n_feedback": len(benchmark.feedback),
            "n_heldout": len(benchmark.heldout),
        },
    )

    for i in range(iterations + 1):
        t0 = time.time()
        calls_before = counted.calls
        fb_out = answer_all(counted, state, benchmark.feedback)
        ho_out = answer_all(counted, state, benchmark.heldout)
        fb_res = benchmark.evaluate(benchmark.feedback, fb_out)
        ho_res = benchmark.evaluate(benchmark.heldout, ho_out)
        if i < iterations and fb_res.per_task is None and feedback_level != FeedbackLevel.SCORE:
            raise ValueError(
                f"Benchmark {benchmark.name!r} only returns aggregate scores; "
                "use feedback_level=SCORE with a score-only mechanism"
            )

        if i < iterations:
            fb = build_feedback(i, benchmark.feedback, fb_res, feedback_level, feedback_budget, rng)
            next_state = mechanism.improve(state, fb, counted)
        else:
            next_state = state

        traj.iterations.append(
            IterationResult(
                iteration=i,
                feedback_score=fb_res.score,
                heldout_score=ho_res.score,
                heldout_per_task=ho_res.per_task,
                feedback_per_task=fb_res.per_task,
                heldout_by_suite=_group_mean(benchmark.heldout, ho_res.per_task, "suite"),
                heldout_by_track=_group_mean(benchmark.heldout, ho_res.per_task, "track"),
                state_chars=state.size_chars,
                n_examples=len(state.examples),
                n_notes=len(state.notes),
                model_calls=counted.calls - calls_before,
                elapsed_s=round(time.time() - t0, 4),
            )
        )
        if verbose:
            print(
                f"[{mechanism.name}] I={i} feedback={fb_res.score:.3f} "
                f"heldout={ho_res.score:.3f} state={state.size_chars}ch"
            )
        state = next_state
    return traj
