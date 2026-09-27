"""Improvement mechanisms: the "algorithm that learns" between iterations.

Each mechanism declares the channel it writes to (prompt, memory, weights) and
the minimum feedback level it needs. The runner refuses to pair a mechanism
with a weaker feedback level than it declares, so results stay honest about
what information the system actually had.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

from .core import (
    AgentState,
    Feedback,
    FeedbackItem,
    FeedbackLevel,
    ImprovementChannel,
    Model,
    TrainableModel,
)

_LEVEL_ORDER = {FeedbackLevel.SCORE: 0, FeedbackLevel.ERRORS: 1, FeedbackLevel.CORRECTIONS: 2}


def feedback_at_least(have: FeedbackLevel, need: FeedbackLevel) -> bool:
    return _LEVEL_ORDER[have] >= _LEVEL_ORDER[need]


class ImprovementMechanism(ABC):
    """Maps (state, feedback, model) to a new state."""

    name: str = "base"
    channel: ImprovementChannel = ImprovementChannel.NONE
    requires: FeedbackLevel = FeedbackLevel.SCORE
    #: True when the model itself authors the change (the "self" in RSI).
    self_authored: bool = False

    @abstractmethod
    def improve(self, state: AgentState, feedback: Feedback, model: Model) -> AgentState: ...

    def describe(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "channel": self.channel.value,
            "requires": self.requires.value,
            "self_authored": self.self_authored,
        }


class NoImprovement(ImprovementMechanism):
    """Control arm. Measures run-to-run noise; any "gain" here is not RSI."""

    name = "none"
    channel = ImprovementChannel.NONE
    requires = FeedbackLevel.SCORE

    def improve(self, state: AgentState, feedback: Feedback, model: Model) -> AgentState:
        new = state.copy()
        new.iteration += 1
        return new


class ScoreFeedback(ImprovementMechanism):
    """Tell the model its cumulative scores and nothing else.

    This is the only mechanism available against a leaderboard with hidden
    labels, where submitting predictions returns a single aggregate score.
    """

    name = "score_feedback"
    channel = ImprovementChannel.PROMPT
    requires = FeedbackLevel.SCORE

    def improve(self, state: AgentState, feedback: Feedback, model: Model) -> AgentState:
        new = state.copy()
        new.history.append(f"Iteration {feedback.iteration}: score {feedback.aggregate_score:.3f}")
        new.iteration += 1
        return new


class FewShotAccumulation(ImprovementMechanism):
    """Recursive prompt improvement by feeding corrected errors back as examples.

    Corrections accumulate across iterations (newest last) up to
    ``max_examples``; beyond that, the oldest are evicted. This makes the
    context budget a first-class constraint.
    """

    name = "fewshot_accumulation"
    channel = ImprovementChannel.MEMORY
    requires = FeedbackLevel.CORRECTIONS

    def __init__(self, max_examples: int = 32) -> None:
        if max_examples < 1:
            raise ValueError("max_examples must be >= 1")
        self.max_examples = max_examples

    def improve(self, state: AgentState, feedback: Feedback, model: Model) -> AgentState:
        new = state.copy()
        seen = {i for i, _ in new.examples}
        for err in feedback.errors:
            if err.reference is None or err.input in seen:
                continue
            new.examples.append((err.input, err.reference))
            seen.add(err.input)
        new.examples = new.examples[-self.max_examples:]
        new.iteration += 1
        return new


def _format_errors(errors: List[FeedbackItem], with_reference: bool, limit: int) -> str:
    lines = []
    for e in errors[:limit]:
        line = f"Input: {e.input}\nYour output: {e.output}"
        if with_reference and e.reference is not None:
            line += f"\nExpected: {e.reference}"
        lines.append(line)
    return "\n\n".join(lines)


REFLECTION_PROMPT = """You are improving your own performance on a benchmark.
Your current score is {score:.3f}. Below are items you got wrong.

{errors}

Existing lessons:
{notes}

Write at most {k} short, general lessons that would fix these mistakes on NEW,
unseen items of the same kind. Do not copy specific answers unless they are a
general rule. One lesson per line, no numbering."""


class SelfReflection(ImprovementMechanism):
    """The model writes its own lessons from its errors (self-authored prompt edit)."""

    name = "self_reflection"
    channel = ImprovementChannel.PROMPT
    requires = FeedbackLevel.ERRORS
    self_authored = True

    def __init__(
        self,
        max_notes: int = 32,
        lessons_per_iteration: int = 8,
        max_errors_shown: int = 20,
        use_references: bool = True,
    ) -> None:
        self.max_notes = max_notes
        self.lessons_per_iteration = lessons_per_iteration
        self.max_errors_shown = max_errors_shown
        self.use_references = use_references

    def improve(self, state: AgentState, feedback: Feedback, model: Model) -> AgentState:
        new = state.copy()
        new.iteration += 1
        if not feedback.errors:
            return new
        with_ref = self.use_references and feedback.level == FeedbackLevel.CORRECTIONS
        prompt = REFLECTION_PROMPT.format(
            score=feedback.aggregate_score,
            errors=_format_errors(feedback.errors, with_ref, self.max_errors_shown),
            notes="\n".join(f"- {n}" for n in new.notes) or "(none)",
            k=self.lessons_per_iteration,
        )
        raw = model.generate(prompt)
        lessons = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in raw.splitlines()]
        lessons = [ln for ln in lessons if ln and not ln.upper().startswith("UNAVAILABLE")]
        existing = set(new.notes)
        for ln in lessons[: self.lessons_per_iteration]:
            if ln not in existing:
                new.notes.append(ln)
                existing.add(ln)
        new.notes = new.notes[-self.max_notes:]
        return new


REWRITE_PROMPT = """You are rewriting the instructions you follow on a benchmark.
Current instructions:
<<<
{instructions}
>>>
Current score: {score:.3f}. Items you got wrong:

{errors}

Write improved instructions that would avoid these mistakes on NEW items.
Return only the new instructions."""


class PromptRewrite(ImprovementMechanism):
    """The model rewrites its own system instructions (self-authored)."""

    name = "prompt_rewrite"
    channel = ImprovementChannel.PROMPT
    requires = FeedbackLevel.ERRORS
    self_authored = True

    def __init__(self, max_errors_shown: int = 20, max_chars: int = 4000) -> None:
        self.max_errors_shown = max_errors_shown
        self.max_chars = max_chars

    def improve(self, state: AgentState, feedback: Feedback, model: Model) -> AgentState:
        new = state.copy()
        new.iteration += 1
        if not feedback.errors:
            return new
        prompt = REWRITE_PROMPT.format(
            instructions=state.instructions,
            score=feedback.aggregate_score,
            errors=_format_errors(
                feedback.errors, feedback.level == FeedbackLevel.CORRECTIONS, self.max_errors_shown
            ),
        )
        out = model.generate(prompt).strip()
        if out and not out.upper().startswith("UNAVAILABLE"):
            new.instructions = out[: self.max_chars]
        return new


class WeightUpdate(ImprovementMechanism):
    """Recursive learning improvement: the model's parameters change.

    Requires a ``TrainableModel``. The adapter decides what training means
    (LoRA step, full fine-tune, a hosted fine-tuning job); RSI-Bench only
    fixes the data it receives: corrected errors from the feedback split.
    """

    name = "weight_update"
    channel = ImprovementChannel.WEIGHTS
    requires = FeedbackLevel.CORRECTIONS

    def __init__(self, replay: bool = True) -> None:
        #: When True, previously seen corrections are replayed each round
        #: (mitigates forgetting at the cost of more compute).
        self.replay = replay
        self._buffer: List[tuple] = []

    def improve(self, state: AgentState, feedback: Feedback, model: Model) -> AgentState:
        if not isinstance(model, TrainableModel):
            raise TypeError(f"{type(model).__name__} does not implement train(); cannot update weights")
        new = state.copy()
        new.iteration += 1
        batch = [(e.input, e.reference) for e in feedback.errors if e.reference is not None]
        if self.replay:
            seen = {i for i, _ in self._buffer}
            self._buffer.extend(p for p in batch if p[0] not in seen)
            batch = list(self._buffer)
        if batch:
            model.train(batch)
        return new


MECHANISMS: Dict[str, Type[ImprovementMechanism]] = {
    m.name: m
    for m in (NoImprovement, ScoreFeedback, FewShotAccumulation, SelfReflection, PromptRewrite, WeightUpdate)
}


def get_mechanism(name: str, **kwargs: object) -> ImprovementMechanism:
    if name not in MECHANISMS:
        raise KeyError(f"Unknown mechanism {name!r}; expected one of {sorted(MECHANISMS)}")
    return MECHANISMS[name](**kwargs)  # type: ignore[arg-type]


def required_level(mechanism: ImprovementMechanism) -> FeedbackLevel:
    return mechanism.requires


def check_compatible(mechanism: ImprovementMechanism, level: FeedbackLevel) -> Optional[str]:
    if not feedback_at_least(level, mechanism.requires):
        return (
            f"Mechanism {mechanism.name!r} needs feedback level {mechanism.requires.value!r}, "
            f"but the run provides {level.value!r}"
        )
    return None
