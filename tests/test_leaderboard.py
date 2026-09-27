"""Leaderboard adapter tests against an in-process fake of the Leaderboard API."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from rsibench import FeedbackLevel, ScoreFeedback, SelfReflection, SimulatedLearner, run_rsi
from rsibench.leaderboard import (
    LeaderboardAPIError,
    LeaderboardBenchmark,
    LeaderboardClient,
    publish_trajectory,
    required_submissions,
    trajectory_rows,
)

QUESTIONS = [{"id": i, "input": f"Compute: {10 + i} + {i}", "context": None} for i in range(10)]
ANSWERS = [str(10 + 2 * i) for i in range(10)]


class _Handler(BaseHTTPRequestHandler):
    submissions: list = []
    curated: list = []
    datasets: set = set()

    def log_message(self, *a):  # silence
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/public/dataset_questions":
            ds = parse_qs(u.query).get("dataset", [""])[0]
            if ds == "hidden":
                return self._send(200, {"success": True, "questions_public": False, "question_count": 10})
            return self._send(200, {"success": True, "questions_public": True, "questions": QUESTIONS})
        self._send(404, {"success": False, "error": "nope"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/public/submit_model":
            _Handler.submissions.append(body)
            ids, res = body["sentence_ids"], body["modelResults"]
            score = sum(r.strip() == ANSWERS[i] for i, r in zip(ids, res)) / len(ids)
            return self._send(200, {"success": True, "score": score, "metric": "accuracy"})
        if self.path == "/api/leaderboard/add_dataset":
            _Handler.datasets.add(body["name"])
            return self._send(200, {"status": "success", "ok": True})
        if self.path == "/api/leaderboard/add_model":
            if body["dataset_name"] not in _Handler.datasets:
                return self._send(404, {"message": "Dataset not found.", "status": "error", "ok": False})
            _Handler.curated.append(body)
            return self._send(200, {"status": "success", "ok": True})
        self._send(404, {"success": False, "error": "nope"})


@pytest.fixture(scope="module")
def client():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield LeaderboardClient(base_url=f"http://127.0.0.1:{srv.server_port}")
    srv.shutdown()


def test_benchmark_from_api_and_remote_scoring(client) -> None:
    bench = LeaderboardBenchmark.from_api(client, "toy", heldout_fraction=0.5)
    assert len(bench.feedback) + len(bench.heldout) == 10
    assert all(t.reference is None for t in bench.feedback)
    traj = run_rsi(SimulatedLearner(), bench, ScoreFeedback(), iterations=2, feedback_level=FeedbackLevel.SCORE)
    assert traj.heldout_scores == [1.0, 1.0, 1.0]  # the simulated learner can add
    assert traj.iterations[0].heldout_per_task is None
    sub = _Handler.submissions[-1]
    assert sub["is_public"] is False and sub["modelName"].startswith("rsibench-eval")


def test_item_level_mechanisms_rejected_on_hidden_labels(client) -> None:
    bench = LeaderboardBenchmark.from_api(client, "toy")
    with pytest.raises(ValueError):
        run_rsi(SimulatedLearner(), bench, SelfReflection(), iterations=1, feedback_level=FeedbackLevel.ERRORS)


def test_hidden_questions_raise(client) -> None:
    with pytest.raises(LeaderboardAPIError):
        LeaderboardBenchmark.from_api(client, "hidden")


def test_publish_trajectory(client) -> None:
    bench = LeaderboardBenchmark.from_api(client, "toy")
    traj = run_rsi(SimulatedLearner(), bench, ScoreFeedback(), iterations=1, feedback_level=FeedbackLevel.SCORE)
    rows = trajectory_rows(traj)
    assert rows[0]["model"].endswith("[score_feedback I=0]")
    out = publish_trajectory(client, traj, "RSI-Bench")  # dataset is created on first publish
    assert len(out) == 2 and "RSI-Bench" in _Handler.datasets
    assert _Handler.curated[-1]["dataset_name"] == "RSI-Bench"


def test_default_instructions_mention_format(client) -> None:
    bench = LeaderboardBenchmark.from_api(client, "toy")
    bench.submission_format = {"allowed_outputs": ["0", "1"], "evaluation_metric": "accuracy"}
    assert "exactly one of: 0, 1" in bench.default_instructions()


def test_required_submissions() -> None:
    assert required_submissions(3) == 8


def test_http_error_surfaces(client) -> None:
    with pytest.raises(LeaderboardAPIError):
        client._request("GET", "/does-not-exist")
