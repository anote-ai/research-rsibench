"""Model adapters.

Provider wrappers mirror ``Leaderboard/backend/models.py``: every SDK import is
optional and a missing key or package yields an ``UNAVAILABLE: ...`` string
instead of an exception, so a sweep over many providers never dies halfway.

``SimulatedLearner`` is a transparent, deterministic reference learner used
for unit tests, CI and sanity checks of the harness. It is **not** a model of
any real LLM; results obtained with it validate the metrics, not a claim
about frontier systems.
"""
from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .data import CONSTRAINT_TAGS, KNOWLEDGE, compliant_answer

try:  # optional .env support, as in the Leaderboard backend
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


def _getenv(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v if v and v.strip() else None


# ---------------------------------------------------------------------------
# Trivial models
# ---------------------------------------------------------------------------


class EchoModel:
    """Returns the task input. Useful as a floor and for plumbing tests."""

    def __init__(self, name: str = "echo") -> None:
        self.name = name

    def generate(self, prompt: str) -> str:
        m = list(re.finditer(r"Input: (.*)\nOutput:\s*$", prompt))
        return m[-1].group(1) if m else prompt[-200:]


class CallableModel:
    """Wrap any ``str -> str`` function (e.g. a Leaderboard ``zero_shot_*`` fn)."""

    def __init__(self, fn: Callable[[str], str], name: Optional[str] = None) -> None:
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "callable")

    def generate(self, prompt: str) -> str:
        return self.fn(prompt)


# ---------------------------------------------------------------------------
# Hosted providers
# ---------------------------------------------------------------------------


class AnthropicModel:
    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 4096, system: Optional[str] = None) -> None:
        self.model = model
        self.name = model
        self.max_tokens = max_tokens
        self.system = system
        self._client = None

    def generate(self, prompt: str) -> str:
        try:
            import anthropic  # type: ignore
        except Exception:
            return "UNAVAILABLE: anthropic package not installed"
        if self._client is None:
            try:
                self._client = anthropic.Anthropic()
            except Exception as e:  # no credentials configured
                return f"UNAVAILABLE: {e}"
        kwargs: Dict[str, object] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system:
            kwargs["system"] = self.system
        r = self._client.messages.create(**kwargs)
        # A refusal is recorded as the model's answer (scored as wrong) rather
        # than silently routed to another model, which would contaminate the
        # trajectory with a different system's outputs.
        if r.stop_reason == "refusal":
            return "REFUSED"
        return "".join(b.text for b in r.content if b.type == "text")


class OpenAIModel:
    def __init__(self, model: str = "gpt-4o-mini", base_url: Optional[str] = None) -> None:
        self.model = model
        self.name = model
        self.base_url = base_url or _getenv("OPENAI_BASE_URL")
        self._client = None

    def generate(self, prompt: str) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return "UNAVAILABLE: openai package not installed"
        key = _getenv("OPENAI_API_KEY")
        if not key:
            return "UNAVAILABLE: OPENAI_API_KEY not set"
        if self._client is None:
            self._client = OpenAI(api_key=key, base_url=self.base_url) if self.base_url else OpenAI(api_key=key)
        r = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return r.choices[0].message.content or ""


class GoogleModel:
    def __init__(self, model: str = "gemini-1.5-flash") -> None:
        self.model = model
        self.name = model

    def generate(self, prompt: str) -> str:
        try:
            import google.generativeai as genai  # type: ignore
        except Exception:
            return "UNAVAILABLE: google-generativeai package not installed"
        key = _getenv("GOOGLE_API_KEY")
        if not key:
            return "UNAVAILABLE: GOOGLE_API_KEY not set"
        genai.configure(api_key=key)
        r = genai.GenerativeModel(self.model).generate_content(prompt)
        return getattr(r, "text", "") or ""


class OllamaModel:
    """Local open-weights models via an Ollama server (no SDK needed)."""

    def __init__(self, model: str = "llama3", base_url: Optional[str] = None, timeout: int = 120) -> None:
        self.model = model
        self.name = f"ollama/{model}"
        self.base_url = (base_url or _getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read()).get("response", "")
        except Exception as e:
            return f"UNAVAILABLE: {e}"


PROVIDERS: Dict[str, Callable[[str], object]] = {
    "anthropic": lambda m: AnthropicModel(m or "claude-opus-5"),
    "openai": lambda m: OpenAIModel(m or "gpt-4o-mini"),
    "google": lambda m: GoogleModel(m or "gemini-1.5-flash"),
    "ollama": lambda m: OllamaModel(m or "llama3"),
    "echo": lambda m: EchoModel(),
    "simulated": lambda m: SimulatedLearner(),
}


def from_spec(spec: str):
    """Build a model from ``provider[:model_id]``, e.g. ``anthropic:claude-opus-5``."""
    provider, _, model_id = spec.partition(":")
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise KeyError(f"Unknown provider {provider!r}; expected one of {sorted(PROVIDERS)}")
    return PROVIDERS[provider](model_id.strip() or None)


# ---------------------------------------------------------------------------
# Deterministic reference learner
# ---------------------------------------------------------------------------

_TASK_RE = re.compile(r"Task:\nInput: (.*)\nOutput:\s*$", re.S)
_PAIR_RE = re.compile(r"Input: (.*?)\nOutput: (.*?)(?:\n|$)")
_ERR_RE = re.compile(r"Input: (.*?)\nYour output: (.*?)(?:\nExpected: (.*?))?(?:\n\n|\n?$)", re.S)


class _Knowledge:
    def __init__(self) -> None:
        self.cipher: Dict[str, str] = {}
        self.skills: Set[str] = set()
        self.facts: Dict[str, str] = {}

    def absorb_pair(self, inp: str, out: str) -> None:
        inp, out = inp.strip(), out.strip()
        if inp.startswith("Encode: "):
            word = inp[len("Encode: "):]
            if len(word) == len(out):
                self.cipher.update(zip(word, out))
        elif inp.startswith("Compute: ") and "*" in inp:
            self.skills.add("multiply")
        elif inp.startswith("Recall the code for item "):
            self.facts[inp.rsplit(" ", 1)[-1]] = out
        else:
            for tag in CONSTRAINT_TAGS:
                if inp.startswith(tag):
                    self.skills.add(tag)

    def absorb_notes(self, text: str) -> None:
        for m in re.finditer(r"MAP ((?:[a-z]=[a-z] ?)+)", text):
            for pair in m.group(1).split():
                a, b = pair.split("=")
                self.cipher[a] = b
        for m in re.finditer(r"SKILL: (.+)", text):
            self.skills.add(m.group(1).strip())
        for m in re.finditer(r"FACT: (\S+) => (\S+)", text):
            self.facts[m.group(1)] = m.group(2)

    def merge(self, other: "_Knowledge") -> None:
        self.cipher.update(other.cipher)
        self.skills |= other.skills
        self.facts.update(other.facts)


class SimulatedLearner:
    """A rule-inducing toy learner with a finite context window.

    What it can learn, and from what:

    * cipher letters, from corrected (input, output) pairs or ``MAP`` notes;
    * the multiplication skill, from one multiplication demonstration;
    * lookup facts, only verbatim (no transfer to unseen keys);
    * instruction tags, from a single demonstration **or** from merely seeing
      its own failed attempt (errors without references are enough).

    It cannot learn from a scalar score. Only the first ``context_chars``
    characters of the non-task part of the prompt are read, so accumulating
    context eventually stops helping. ``train`` writes to parametric memory
    (the weights channel), bounded by ``fact_capacity``.
    """

    def __init__(
        self,
        name: str = "simulated-learner",
        context_chars: int = 6000,
        fact_capacity: int = 1000,
        noise: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.context_chars = context_chars
        self.fact_capacity = fact_capacity
        self.noise = noise
        self._rng = random.Random(seed)
        self.weights = _Knowledge()

    # -- weights channel ---------------------------------------------------
    def train(self, examples: Sequence[Tuple[str, str]]) -> None:
        for inp, out in examples:
            self.weights.absorb_pair(inp, out)
        while len(self.weights.facts) > self.fact_capacity:
            self.weights.facts.pop(next(iter(self.weights.facts)))

    # -- inference ---------------------------------------------------------
    def generate(self, prompt: str) -> str:
        if prompt.startswith("You are improving your own performance"):
            return self._reflect(prompt, rewrite=False)
        if prompt.startswith("You are rewriting the instructions"):
            return self._reflect(prompt, rewrite=True)
        m = _TASK_RE.search(prompt)
        if not m:
            return ""
        task_input = m.group(1).strip()
        context = prompt[: m.start()][: self.context_chars]
        k = _Knowledge()
        k.merge(self.weights)
        for inp, out in _PAIR_RE.findall(context):
            k.absorb_pair(inp, out)
        k.absorb_notes(context)
        answer = self._solve(task_input, k)
        if self.noise and self._rng.random() < self.noise:
            answer = answer[::-1] + "?"
        return answer

    def _solve(self, inp: str, k: _Knowledge) -> str:
        if inp.startswith("Encode: "):
            return "".join(k.cipher.get(c, c) for c in inp[len("Encode: "):])
        if inp.startswith("Compute: "):
            a, op, b = inp[len("Compute: "):].split()
            x, y = int(a), int(b)
            if op == "+":
                return str(x + y)
            if op == "-":
                return str(x - y)
            return str(x * y if "multiply" in k.skills else x + y)
        if inp.startswith("Recall the code for item "):
            return k.facts.get(inp.rsplit(" ", 1)[-1], "unknown")
        for tag in CONSTRAINT_TAGS:
            if inp.startswith(tag):
                q = inp[len(tag):].strip()
                ans = KNOWLEDGE.get(q, "unknown")
                return compliant_answer(tag, ans) if tag in k.skills else f"the answer is {ans}"
        return "unknown"

    def _reflect(self, prompt: str, rewrite: bool) -> str:
        maps: Dict[str, str] = {}
        skills: List[str] = []
        facts: List[str] = []
        for inp, _out, exp in _ERR_RE.findall(prompt):
            inp = inp.strip()
            exp = (exp or "").strip()
            if inp.startswith("Encode: ") and exp:
                word = inp[len("Encode: "):]
                if len(word) == len(exp):
                    maps.update(zip(word, exp))
            elif inp.startswith("Compute: ") and "*" in inp and exp:
                skills.append("multiply")
            elif inp.startswith("Recall the code for item ") and exp:
                facts.append(f"FACT: {inp.rsplit(' ', 1)[-1]} => {exp}")
            else:
                for tag in CONSTRAINT_TAGS:
                    if inp.startswith(tag):
                        skills.append(tag)
        lines: List[str] = []
        if maps:
            lines.append("MAP " + " ".join(f"{a}={b}" for a, b in sorted(maps.items())))
        lines.extend(f"SKILL: {s}" for s in dict.fromkeys(skills))
        lines.extend(facts)
        if rewrite:
            cur = re.search(r"<<<\n(.*?)\n>>>", prompt, re.S)
            base = cur.group(1).strip() if cur else ""
            return "\n".join([base, *lines]).strip()
        return "\n".join(lines)
