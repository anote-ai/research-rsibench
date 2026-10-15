"""Command-line entry point: ``rsibench run | sweep | leaderboard | summarize``."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .core import AgentState, Benchmark, FeedbackLevel
from .data import load_jsonl, make_benchmark
from .evaluate import markdown_table, rank, summarize
from .mechanisms import get_mechanism
from .models import from_spec
from .runner import Trajectory, run_rsi
from .sweep import aggregate_table, default_feedback, resolve_suites, run_sweep


def _level_for(mech_name: str, requested: Optional[str]) -> FeedbackLevel:
    if requested and mech_name != "none":
        return FeedbackLevel(requested)
    return default_feedback(mech_name)


def cmd_run(args: argparse.Namespace) -> None:
    if args.tasks:
        bench = Benchmark.from_tasks(Path(args.tasks).stem, load_jsonl(args.tasks), seed=args.seed)
    else:
        bench = make_benchmark(suites=resolve_suites(args.suites), n_per_suite=args.n, seed=args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajs: List[Trajectory] = []
    for name in args.mechanisms.split(","):
        stem = f"{args.model.replace('/', '_').replace(':', '_')}__{name}__seed{args.seed}"
        traj = run_rsi(
            from_spec(args.model),  # fresh model per mechanism: weights never leak between arms
            bench,
            get_mechanism(name),
            iterations=args.iterations,
            feedback_level=_level_for(name, args.feedback),
            feedback_budget=args.budget,
            seed=args.seed,
            max_workers=args.workers,
            checkpoint=out_dir / f"{stem}.ckpt.json",
            verbose=True,
        )
        traj.save_json(out_dir / f"{stem}.json")
        (out_dir / f"{stem}.ckpt.json").unlink(missing_ok=True)
        trajs.append(traj)
    control = next((t for t in trajs if t.mechanism.get("name") == "none"), None)
    print()
    print(markdown_table(rank(summarize(t, control) for t in trajs)))


def cmd_sweep(args: argparse.Namespace) -> None:
    factories = {spec: (lambda spec=spec: from_spec(spec)) for spec in args.models.split(",")}
    _, aggs = run_sweep(
        factories,
        mechanisms=args.mechanisms.split(","),
        seeds=[int(s) for s in args.seeds.split(",")],
        suites=resolve_suites(args.suites),
        n_per_suite=args.n,
        iterations=args.iterations,
        feedback_budget=args.budget,
        feedback=FeedbackLevel(args.feedback) if args.feedback else None,
        out_dir=args.out,
        max_workers=args.workers,
        verbose=True,
    )
    table = aggregate_table(aggs)
    Path(args.out, "summary.md").write_text(table + "\n")
    print()
    print(table)


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


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mechanisms", default="none,score_feedback,fewshot_accumulation,self_reflection,prompt_rewrite")
    p.add_argument("--suites", default="all", help="core | headroom | all | comma-separated suite names")
    p.add_argument("--n", type=int, default=20, help="tasks per built-in suite")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--budget", type=int, default=10, help="max corrected errors revealed per round")
    p.add_argument("--feedback", choices=[lvl.value for lvl in FeedbackLevel], default=None,
                   help="override feedback level for non-control arms")
    p.add_argument("--workers", type=int, default=4, help="parallel model calls")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="rsibench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one model, one seed; resumable")
    r.add_argument("--model", default="simulated", help="provider[:model], e.g. anthropic:claude-opus-5")
    _common(r)
    r.add_argument("--tasks", help="JSONL task file (overrides --suites)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="results/runs")
    r.set_defaults(func=cmd_run)

    sw = sub.add_parser("sweep", help="models x mechanisms x seeds, aggregated with 95%% CIs; resumable")
    sw.add_argument("--models", default="simulated", help="comma-separated provider[:model] specs")
    _common(sw)
    sw.add_argument("--seeds", default="0,1,2")
    sw.add_argument("--out", default="results/sweep")
    sw.set_defaults(func=cmd_sweep)

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
