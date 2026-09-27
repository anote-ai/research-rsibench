"""Bridge to the Anote Model Leaderboard API (github.com/anote-ai/Leaderboard).

Three integration points:

1. **Datasets in.** ``LeaderboardBenchmark.from_api`` pulls a dataset's public
   inputs from ``GET /public/dataset_questions``. Labels stay on the server.
2. **Scoring.** Each evaluation posts predictions to ``POST /public/submit_model``
   (as private submissions) and reads back the aggregate score. Because the
   server returns only a scalar, the only admissible feedback level is
   ``SCORE``, i.e. the "cumulative scores fed back to the model" setting.
3. **Results out.** ``publish_trajectory`` writes one row per iteration to the
   curated leaderboard via ``POST /api/leaderboard/add_model`` so RSI curves
   show up next to static scores.

Only the standard library is used so this works wherever ``rsibench`` does.
"""
from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .core import Benchmark, EvalResult, Task, Track
from .models import CallableModel
from .runner import Trajectory


class LeaderboardAPIError(RuntimeError):
    pass


class LeaderboardClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.base_url = (base_url or os.getenv("LEADERBOARD_API_BASE", "http://localhost:5001")).rstrip("/")
        self.api_key = api_key or os.getenv("LEADERBOARD_API_KEY")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None,
                 params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            hint = ""
            if e.code == 429:
                hint = (" (an RSI run needs 2*(iterations+1) submissions; see required_submissions(). "
                        "For a local backend raise DAILY_SUBMISSION_LIMIT.)")
            raise LeaderboardAPIError(f"{method} {path} -> HTTP {e.code}: {detail}{hint}") from e
        if isinstance(out, dict) and (out.get("success") is False or out.get("status") == "error"):
            raise LeaderboardAPIError(f"{method} {path} failed: {out.get('error') or out.get('message')}")
        return out

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def list_datasets(self) -> Dict[str, Any]:
        return self._request("GET", "/public/datasets")

    def submission_format(self, dataset: str) -> Dict[str, Any]:
        return self._request("GET", "/public/submission_format", params={"dataset": dataset})

    def dataset_questions(self, dataset: str) -> Dict[str, Any]:
        return self._request("GET", "/public/dataset_questions", params={"dataset": dataset})

    def submit(
        self,
        dataset: str,
        model_name: str,
        results: Sequence[str],
        ids: Sequence[int],
        metadata: Optional[Dict[str, Any]] = None,
        is_public: bool = False,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "benchmarkDatasetName": dataset,
            "modelName": model_name,
            "modelResults": list(results),
            "sentence_ids": list(ids),
            "is_public": is_public,
        }
        if metadata:
            body["metadata"] = metadata
        return self._request("POST", "/public/submit_model", body=body)

    def add_dataset(self, name: str, task_type: str, description: str = "", url: str = "") -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": name, "task_type": task_type}
        if description:
            body["description"] = description
        if url:
            body["url"] = url
        return self._request("POST", "/api/leaderboard/add_dataset", body=body)

    def add_model(self, dataset_name: str, model: str, score: float, rank: Optional[int] = None,
                  updated: Optional[str] = None) -> Dict[str, Any]:
        body = {
            "dataset_name": dataset_name,
            "model": model,
            "rank": rank,
            "score": score,
            "updated": updated or _dt.date.today().strftime("%b %Y"),
        }
        return self._request("POST", "/api/leaderboard/add_model", body=body)


class LeaderboardBenchmark(Benchmark):
    """A Leaderboard dataset as an RSI-Bench benchmark with server-side scoring."""

    def __init__(self, name: str, feedback: List[Task], heldout: List[Task], client: LeaderboardClient,
                 dataset: str, run_label: str = "rsibench", description: str = "",
                 min_interval_s: float = 0.0, submission_format: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name, feedback=feedback, heldout=heldout, description=description)
        self.client = client
        self.dataset = dataset
        self.run_label = run_label
        #: Seconds to wait between submissions; the Leaderboard rate-limits
        #: ``/public/submit_model`` (SUBMIT_MODEL_RATE_LIMIT, default 10/minute).
        self.min_interval_s = min_interval_s
        self.submission_format = submission_format or {}
        self.n_submissions = 0
        self._last_submit = 0.0

    def default_instructions(self) -> str:
        """Base instructions that tell the model what the scorer expects."""
        fmt = self.submission_format
        parts = [f"You are solving items from the '{self.dataset}' benchmark."]
        if fmt.get("task_type"):
            parts.append(f"Task type: {fmt['task_type']}.")
        if fmt.get("evaluation_metric"):
            parts.append(f"Scored by {fmt['evaluation_metric']}.")
        if fmt.get("allowed_outputs"):
            parts.append("Reply with exactly one of: " + ", ".join(map(str, fmt["allowed_outputs"])) + ".")
        parts.append("Reply with the answer only.")
        return " ".join(parts)

    @classmethod
    def from_api(
        cls,
        client: LeaderboardClient,
        dataset: str,
        heldout_fraction: float = 0.5,
        limit: Optional[int] = None,
        seed: int = 0,
        run_label: str = "rsibench",
        min_interval_s: float = 0.0,
    ) -> "LeaderboardBenchmark":
        payload = client.dataset_questions(dataset)
        try:
            fmt = client.submission_format(dataset)
        except LeaderboardAPIError:
            fmt = {}
        if not payload.get("questions_public", True):
            raise LeaderboardAPIError(
                f"Questions for {dataset!r} are hidden; RSI-Bench needs the inputs to run a model"
            )
        qs = payload.get("questions") or []
        if limit:
            qs = qs[:limit]
        tasks = []
        for q in qs:
            text = q.get("input", "")
            if q.get("context"):
                text = f"Context: {q['context']}\n{text}"
            if q.get("options"):
                text += "\nOptions: " + ", ".join(map(str, q["options"]))
            tasks.append(Task(id=str(q["id"]), input=text, reference=None, suite=dataset,
                              track=Track.CAPABILITY, metadata={"leaderboard_id": int(q["id"])}))
        split = Benchmark.from_tasks(dataset, tasks, heldout_fraction=heldout_fraction, seed=seed)
        return cls(name=f"leaderboard:{dataset}", feedback=split.feedback, heldout=split.heldout,
                   client=client, dataset=dataset, run_label=run_label,
                   min_interval_s=min_interval_s, submission_format=fmt)

    def evaluate(self, tasks: Sequence[Task], outputs: Dict[str, str]) -> EvalResult:
        wait = self.min_interval_s - (time.time() - self._last_submit)
        if wait > 0:
            time.sleep(wait)
        self._last_submit = time.time()
        self.n_submissions += 1
        ids = [int(t.metadata["leaderboard_id"]) for t in tasks]
        res = self.client.submit(
            self.dataset,
            model_name=f"{self.run_label}-eval{self.n_submissions:03d}",
            results=[outputs.get(t.id, "") for t in tasks],
            ids=ids,
            metadata={"source": "rsibench", "n_items": len(ids)},
            is_public=False,
        )
        score = float(res["score"])
        if score > 1.0:  # some metrics are reported on a 0-100 scale
            score /= 100.0
        return EvalResult(score=max(0.0, min(1.0, score)), per_task=None, outputs=dict(outputs))


def required_submissions(iterations: int) -> int:
    """Submissions an RSI run consumes: feedback + held-out split at each of T+1 iterations.

    The Leaderboard enforces a per-submitter daily quota (``DAILY_SUBMISSION_LIMIT``,
    default 5), so check this before starting a long run.
    """
    return 2 * (iterations + 1)


def trajectory_rows(traj: Trajectory) -> List[Dict[str, Any]]:
    mech = traj.mechanism.get("name")
    return [
        {"model": f"{traj.model} [{mech} I={it.iteration}]", "score": round(it.heldout_score, 4),
         "iteration": it.iteration}
        for it in traj.iterations
    ]


def publish_trajectory(client: LeaderboardClient, traj: Trajectory, dataset_name: str = "RSI-Bench") -> List[Dict]:
    """Post one curated-leaderboard row per iteration, creating the dataset if needed."""
    rows = trajectory_rows(traj)
    try:
        first = client.add_model(dataset_name, rows[0]["model"], rows[0]["score"])
    except LeaderboardAPIError as e:
        if "HTTP 404" not in str(e):
            raise
        client.add_dataset(
            dataset_name,
            task_type="rsi",
            description="RSI-Bench trajectories: held-out score after each self-improvement iteration.",
            url="https://github.com/anote-ai/research-rsibench",
        )
        first = client.add_model(dataset_name, rows[0]["model"], rows[0]["score"])
    return [first] + [client.add_model(dataset_name, r["model"], r["score"]) for r in rows[1:]]


def load_leaderboard_models(backend_dir: str | Path) -> List[CallableModel]:
    """Reuse the Leaderboard backend's ``zero_shot_*`` functions as RSI-Bench models.

    ``backend_dir`` is the path to ``Leaderboard/backend``. The default model
    list from ``models.list_models()`` is honoured.
    """
    path = Path(backend_dir) / "models.py"
    spec = importlib.util.spec_from_file_location("leaderboard_models", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = mod.get_model_functions()
    return [CallableModel(fns[d["fn"]], name=d["name"]) for d in mod.list_models() if d["fn"] in fns]
