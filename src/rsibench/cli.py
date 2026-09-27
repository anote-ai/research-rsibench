"""Command-line entry point: ``rsibench run | leaderboard | summarize``."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .core import AgentState, FeedbackLevel
from .data import load_jsonl, make_benchmark
from .core import Benchmark
from .evaluate import markdown_table, rank, summarize
from .mechanisms import MECHANISMS, get_mechanism
from .models import from_spec
from .runner import Trajectory, run_rsi


def _level_for(mech_name: str, requested: Optional[str]) -> FeedbackLevel:
    if requested:
        return FeedbackLevel(requested)
    # Local suites have labels, so give every item-level mechanism full corrections.
    if MECHANISMS[mech_name].requires == FeedbackLevel.SCORE:
        return FeedbackLevel.SCORE
    return FeedbackLevel.CORRECTIONS


def cmd_run(args: argparse.Namespace) -> None:
    if args.tasks:
        bench = Benchmark.from_tasks(Path(args.tasks).stem, load_jsonl(args.tasks), seed=args.seed)
    else:
        bench = make_benchmark(suites=args.suites.split(","), n_per_suite=args.n, seed=args.seed)
    model = from_spec(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajs: List[Trajectory] = []
    for name in args.mechanisms.split(","):
        traj = run_rsi(
            model if name != "weight_update" else from_spec(args.model),
            bench,
            get_mechanism(name),
            iterations=args.iterations,
            feedback_level=_level_for(name, args.feedback),
            feedback_budget=args.budget,
            seed=args.seed,
            verbose=True,
        )
        traj.save_json(out_dir / f"{traj.model.replace('/', '_')}__{name}.json")
        trajs.append(traj)
    control = next((t for t in trajs if t.mechanism.get("name") == "none"), None)
    print()
    print(markdown_table(rank(summarize(t, control) for t in trajs)))


def cmd_leaderboard(args: argparse.Namespace) -> None:
    from .leaderboard import (
        LeaderboardBenchmark,
        LeaderboardClient,
        publish_trajectory,
        required_submissions,
    )

    client = LeaderboardClient(base_url=args.base_url)
    bench = LeaderboardBenchmark.from_api(client, args.dataset, limit=args.limit, seed=args.seed,
                                          run_label=f"rsibench-{args.model.replace(':', '-')}",
                                          min_interval_s=args.min_interval)
    print(f"this run uses {required_submissions(args.iterations)} submissions")
    traj = run_rsi(from_spec(args.model), bench, get_mechanism("score_feedback"),
                   iterations=args.iterations, feedback_level=FeedbackLevel.SCORE,
                   initial_state=AgentState(instructions=bench.default_instructions()), verbose=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    traj.save_json(args.out)
    print(markdown_table([summarize(traj)]))
    if args.publish:
        publish_trajectory(client, traj, args.publish)
        print(f"published {len(traj.iterations)} rows to leaderboard dataset {args.publish!r}")


def cmd_summarize(args: argparse.Namespace) -> None:
    trajs = [Trajectory.load_json(p) for p in args.files]
    print(markdown_table(rank(summarize(t) for t in trajs)))


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="rsibench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run RSI loops on built-in suites or a JSONL task file")
    r.add_argument("--model", default="simulated", help="provider[:model], e.g. anthropic:claude-opus-5")
    r.add_argument("--mechanisms", default="none,fewshot_accumulation,self_reflection")
    r.add_argument("--suites", default="cipher,arithmetic,lookup,instruction")
    r.add_argument("--tasks", help="JSONL task file (overrides --suites)")
    r.add_argument("--n", type=int, default=40, help="tasks per built-in suite")
    r.add_argument("--iterations", type=int, default=5)
    r.add_argument("--budget", type=int, default=None, help="max corrected errors revealed per round")
    r.add_argument("--feedback", choices=[l.value for l in FeedbackLevel], default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="results/runs")
    r.set_defaults(func=cmd_run)

    lb = sub.add_parser("leaderboard", help="score-feedback RSI loop against a Leaderboard dataset")
    lb.add_argument("--dataset", required=True)
    lb.add_argument("--model", required=True)
    lb.add_argument("--base-url", default=None)
    lb.add_argument("--iterations", type=int, default=3)
    lb.add_argument("--limit", type=int, default=50)
    lb.add_argument("--seed", type=int, default=0)
    lb.add_argument("--min-interval", type=float, default=6.5,
                    help="seconds between submissions (Leaderboard default limit is 10/minute)")
    lb.add_argument("--out", default="results/leaderboard_run.json")
    lb.add_argument("--publish", default=None, help="curated leaderboard dataset name to publish rows to")
    lb.set_defaults(func=cmd_leaderboard)

    s = sub.add_parser("summarize", help="summarize saved trajectories")
    s.add_argument("files", nargs="+")
    s.set_defaults(func=cmd_summarize)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
