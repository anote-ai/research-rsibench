from __future__ import annotations

import pytest

from rsibench.core import AgentState, Task
from rsibench.models import EchoModel, SimulatedLearner, from_spec


def test_echo_returns_task_input() -> None:
    p = AgentState().render(Task(id="t", input="hello", reference="x"))
    assert EchoModel().generate(p) == "hello"


def test_simulated_learner_learns_cipher_from_examples() -> None:
    m = SimulatedLearner()
    t = Task(id="t", input="Encode: ab", reference="xy")
    assert m.generate(AgentState().render(t)) == "ab"
    s = AgentState(examples=[("Encode: ba", "yx")])
    assert m.generate(s.render(t)) == "xy"


def test_context_window_truncates() -> None:
    m = SimulatedLearner(context_chars=50)
    s = AgentState(examples=[("Encode: ba", "yx")])
    assert m.generate(s.render(Task(id="t", input="Encode: ab", reference="xy"))) == "ab"


def test_train_writes_weights() -> None:
    m = SimulatedLearner()
    m.train([("Compute: 12 * 12", "144")])
    assert m.generate(AgentState().render(Task(id="t", input="Compute: 11 * 11", reference="121"))) == "121"


def test_from_spec() -> None:
    assert from_spec("anthropic:claude-opus-5").name == "claude-opus-5"
    assert from_spec("simulated").name == "simulated-learner"
    with pytest.raises(KeyError):
        from_spec("nope:x")
