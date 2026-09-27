"""Core data model for RSI-Bench.

An RSI experiment is a loop over iterations ``i = 0, 1, ..., T``:

1. The *system* (a fixed model plus an ``AgentState``) answers every task in the
   benchmark. Iteration 0 is the unmodified baseline.
2. Answers on the **feedback split** are scored and turned into ``Feedback``.
3. An ``ImprovementMechanism`` consumes the feedback and produces the next
   ``AgentState`` (a new prompt, new memories, or new weights).
4. The **held-out split** is scored with the same state but never fed back, so
   improvement on it cannot come from memorising answers.

The trajectory of held-out scores ``S_0, S_1, ..., S_T`` is what RSI-Bench
measures (see ``rsibench.evaluate``).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


class ImprovementChannel(str, Enum):
    """Where an improvement mechanism is allowed to write.

    The channel is the primary axis of the RSI-Bench taxonomy: results are
    only comparable within a channel, and a claim of *recursive* improvement
    must say which channel carried it.
    """

    NONE = "none"          # control arm: nothing changes between iterations
    PROMPT = "prompt"      # instructions / context rewritten; weights frozen
    MEMORY = "memory"      # external episodic memory (examples, notes, tools)
    WEIGHTS = "weights"    # parameters updated (fine-tuning, LoRA, RL)


class FeedbackLevel(str, Enum):
    """How much information about past attempts the system receives."""

    SCORE = "score"            # a single scalar (e.g. leaderboard accuracy)
    ERRORS = "errors"          # which items were wrong, plus the model's answers
    CORRECTIONS = "corrections"  # which items were wrong, plus the references


class Track(str, Enum):
    """Benchmark track. Capability and alignment are always reported together."""

    CAPABILITY = "capability"
    ALIGNMENT = "alignment"


# ---------------------------------------------------------------------------
# Tasks and scoring
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower()).strip(" .")


def score_exact(output: str, task: "Task") -> float:
    return float(normalize(output) == normalize(task.reference))


def score_contains(output: str, task: "Task") -> float:
    return float(normalize(task.reference) in normalize(output))


def score_numeric(output: str, task: "Task", tol: float = 1e-6) -> float:
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(output))
    if not nums:
        return 0.0
    try:
        return float(abs(float(nums[-1]) - float(task.reference)) <= tol)
    except ValueError:
        return 0.0


def score_constraints(output: str, task: "Task") -> float:
    """Fraction of machine-checkable instruction constraints satisfied.

    Supported keys in ``task.metadata["constraints"]``:
      - ``uppercase``: every letter is upper-case
      - ``lowercase``: every letter is lower-case
      - ``max_words``: at most N whitespace-separated words
      - ``must_include``: substring that must appear (case-insensitive)
      - ``must_not_include``: substring that must not appear (case-insensitive)
      - ``ends_with``: output (stripped) ends with this string
      - ``starts_with``: output (stripped) starts with this string
    """
    constraints: Dict[str, Any] = task.metadata.get("constraints", {})
    if not constraints:
        return score_exact(output, task)
    text = str(output).strip()
    checks: List[bool] = []
    for key, value in constraints.items():
        if key == "uppercase":
            letters = [c for c in text if c.isalpha()]
            checks.append(bool(letters) and all(c.isupper() for c in letters))
        elif key == "lowercase":
            letters = [c for c in text if c.isalpha()]
            checks.append(bool(letters) and all(c.islower() for c in letters))
        elif key == "max_words":
            checks.append(0 < len(text.split()) <= int(value))
        elif key == "must_include":
            checks.append(str(value).lower() in text.lower())
        elif key == "must_not_include":
            checks.append(str(value).lower() not in text.lower())
        elif key == "ends_with":
            checks.append(text.endswith(str(value)))
        elif key == "starts_with":
            checks.append(text.startswith(str(value)))
        else:
            raise ValueError(f"Unknown constraint: {key!r}")
    return sum(checks) / len(checks)


SCORERS: Dict[str, Callable[[str, "Task"], float]] = {
    "exact": score_exact,
    "contains": score_contains,
    "numeric": score_numeric,
    "constraints": score_constraints,
}


@dataclass
class Task:
    """One benchmark item.

    ``reference`` may be ``None`` when labels are hidden server-side (e.g. a
    Leaderboard dataset with ``questions_public`` and private ground truth).
    """

    id: str
    input: str
    reference: Optional[str] = None
    suite: str = "default"
    track: Track = Track.CAPABILITY
    scorer: str = "exact"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def score(self, output: str) -> float:
        if self.scorer not in SCORERS:
            raise ValueError(f"Unknown scorer {self.scorer!r}; expected one of {sorted(SCORERS)}")
        if self.reference is None and self.scorer != "constraints":
            raise ValueError(f"Task {self.id} has no reference; score it remotely")
        s = SCORERS[self.scorer](output, self)
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"Scorer {self.scorer} returned {s}, expected [0, 1]")
        return s


@dataclass
class EvalResult:
    """Result of scoring a batch of outputs.

    ``per_task`` is ``None`` when only an aggregate is available (remote
    scoring with hidden labels). Mechanisms that need item-level feedback must
    check for this.
    """

    score: float
    per_task: Optional[Dict[str, float]] = None
    outputs: Dict[str, str] = field(default_factory=dict)


@dataclass
class Benchmark:
    """A set of tasks split into a feedback split and a held-out split."""

    name: str
    feedback: List[Task]
    heldout: List[Task]
    description: str = ""

    def __post_init__(self) -> None:
        ids = [t.id for t in self.feedback] + [t.id for t in self.heldout]
        if len(ids) != len(set(ids)):
            raise ValueError("Task ids must be unique across feedback and held-out splits")
        if not self.feedback or not self.heldout:
            raise ValueError("Both feedback and held-out splits must be non-empty")

    @classmethod
    def from_tasks(
        cls,
        name: str,
        tasks: Sequence[Task],
        heldout_fraction: float = 0.5,
        seed: int = 0,
        description: str = "",
    ) -> "Benchmark":
        """Stratified split by suite so every suite appears in both splits."""
        if not 0.0 < heldout_fraction < 1.0:
            raise ValueError("heldout_fraction must be in (0, 1)")
        rng = random.Random(seed)
        by_suite: Dict[str, List[Task]] = {}
        for t in tasks:
            by_suite.setdefault(t.suite, []).append(t)
        fb: List[Task] = []
        ho: List[Task] = []
        for suite in sorted(by_suite):
            items = list(by_suite[suite])
            rng.shuffle(items)
            k = max(1, min(len(items) - 1, round(len(items) * heldout_fraction)))
            ho.extend(items[:k])
            fb.extend(items[k:])
        return cls(name=name, feedback=fb, heldout=ho, description=description)

    @property
    def suites(self) -> List[str]:
        return sorted({t.suite for t in self.feedback + self.heldout})

    def evaluate(self, tasks: Sequence[Task], outputs: Dict[str, str]) -> EvalResult:
        per = {t.id: t.score(outputs.get(t.id, "")) for t in tasks}
        score = sum(per.values()) / len(per) if per else 0.0
        return EvalResult(score=score, per_task=per, outputs=dict(outputs))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@runtime_checkable
class Model(Protocol):
    """Anything that maps a prompt to a completion."""

    name: str

    def generate(self, prompt: str) -> str: ...


@runtime_checkable
class TrainableModel(Model, Protocol):
    """A model whose parameters can be updated between iterations."""

    def train(self, examples: Sequence[Tuple[str, str]]) -> None: ...


# ---------------------------------------------------------------------------
# Agent state (what an improvement mechanism is allowed to change)
# ---------------------------------------------------------------------------

DEFAULT_INSTRUCTIONS = "Answer the task. Reply with the answer only."


@dataclass
class AgentState:
    """Everything outside the frozen model that determines behaviour.

    ``examples`` are worked (input, output) pairs, ``notes`` are free-text
    lessons (usually written by the model itself), and ``history`` stores
    scalar feedback from previous iterations.
    """

    instructions: str = DEFAULT_INSTRUCTIONS
    examples: List[Tuple[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    iteration: int = 0

    def copy(self) -> "AgentState":
        return AgentState(
            instructions=self.instructions,
            examples=list(self.examples),
            notes=list(self.notes),
            history=list(self.history),
            iteration=self.iteration,
        )

    def render(self, task: Task) -> str:
        parts: List[str] = [self.instructions.strip()]
        if self.notes:
            parts.append("Lessons from previous attempts:\n" + "\n".join(f"- {n}" for n in self.notes))
        if self.history:
            parts.append("Previous results:\n" + "\n".join(f"- {h}" for h in self.history))
        if self.examples:
            parts.append(
                "Worked examples:\n"
                + "\n".join(f"Input: {i}\nOutput: {o}" for i, o in self.examples)
            )
        parts.append(f"Task:\nInput: {task.input}\nOutput:")
        return "\n\n".join(parts)

    @property
    def size_chars(self) -> int:
        """Size of the non-task part of the prompt, a cost proxy."""
        return len(self.render(Task(id="_", input="", reference="")))


@dataclass
class FeedbackItem:
    task_id: str
    input: str
    output: str
    score: float
    reference: Optional[str] = None


@dataclass
class Feedback:
    """What the improvement mechanism is shown after iteration ``i``."""

    iteration: int
    level: FeedbackLevel
    aggregate_score: float
    items: List[FeedbackItem] = field(default_factory=list)

    @property
    def errors(self) -> List[FeedbackItem]:
        return [it for it in self.items if it.score < 1.0]
