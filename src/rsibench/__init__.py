"""RSI-Bench: an open benchmark for measuring recursive self-improvement in AI systems."""
from .core import (
    DEFAULT_INSTRUCTIONS,
    AgentState,
    Benchmark,
    EvalResult,
    Feedback,
    FeedbackItem,
    FeedbackLevel,
    ImprovementChannel,
    Model,
    Task,
    TrainableModel,
    Track,
)
from .data import load_jsonl, make_benchmark, make_tasks, save_jsonl
from .evaluate import (
    RSILevel,
    Summary,
    acceleration,
    alignment_drift,
    auic,
    closure_rates,
    eta,
    fit_saturating,
    forgetting_rate,
    frontier_items,
    generalization_gap,
    markdown_table,
    normalized_gain,
    paired_bootstrap_ci,
    rank,
    relative_acceleration,
    rsi_level,
    sign_test_p,
    step_gains,
    summarize,
    total_gain,
)
from .mechanisms import (
    MECHANISMS,
    FewShotAccumulation,
    ImprovementMechanism,
    NoImprovement,
    PromptRewrite,
    ScoreFeedback,
    SelfReflection,
    WeightUpdate,
    get_mechanism,
)
from .models import EchoModel, SimulatedLearner, from_spec
from .sweep import SeedAggregate, aggregate, aggregate_table, run_sweep
from .runner import IterationResult, Trajectory, run_rsi

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_INSTRUCTIONS", "AgentState", "Benchmark", "EvalResult", "Feedback", "FeedbackItem", "FeedbackLevel",
    "ImprovementChannel", "Model", "Task", "TrainableModel", "Track",
    "load_jsonl", "make_benchmark", "make_tasks", "save_jsonl",
    "RSILevel", "Summary", "acceleration", "alignment_drift", "auic", "closure_rates", "eta", "fit_saturating",
    "forgetting_rate", "frontier_items", "generalization_gap", "markdown_table", "normalized_gain",
    "paired_bootstrap_ci", "rank", "relative_acceleration", "rsi_level", "sign_test_p", "step_gains", "summarize", "total_gain",
    "MECHANISMS", "FewShotAccumulation", "ImprovementMechanism", "NoImprovement", "PromptRewrite",
    "ScoreFeedback", "SelfReflection", "WeightUpdate", "get_mechanism",
    "EchoModel", "SimulatedLearner", "from_spec",
    "IterationResult", "Trajectory", "run_rsi",
    "SeedAggregate", "aggregate", "aggregate_table", "run_sweep",
]
