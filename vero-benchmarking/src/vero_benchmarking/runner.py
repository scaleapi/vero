#!/usr/bin/env python
"""Run VeRO benchmark optimization and baseline evaluation sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, NamedTuple
from uuid import uuid4

import pandas as pd
from vero.agents import AgentCandidateProducer
from vero.evaluation import (
    CaseIds,
    CaseRange,
    EvaluationBudget,
    EvaluationLimits,
    EvaluationRecord,
    EvaluationSet,
    MetricSelector,
    ObjectiveSpec,
    PythonTaskBackend,
    PythonTaskBackendConfig,
    RetryPolicy,
)
from vero.optimization import OptimizationResult, SequentialStrategy
from vero.runtime import OptimizationSession, create_local_optimization_session

from vero_benchmarking.cases import materialize_cases
from vero_benchmarking.constants import DEFAULT_RESULTS_DIR, DEFAULT_SEED
from vero_benchmarking.tasks import load_task
from vero_benchmarking.tasks.base import OptimizationTask


MODELS: dict[str, str] = {
    "sonnet": "anthropic/claude-sonnet-4-5-20250929",
    "opus": "anthropic/claude-opus-4-5-20251101",
    "haiku": "anthropic/claude-haiku-4-5-20251001",
    "gpt": "gpt-5.2-codex",
}

DEFAULT_EVAL_TIMEOUT = 600
DEFAULT_CASE_TIMEOUT = 180
BACKEND_ID = "python-task"
PASSTHROUGH_ENVIRONMENT = [
    "ANTHROPIC_API_KEY",
    "LITELLM_API_KEY",
    "LITELLM_BASE_URL",
    "OPENAI_API_KEY",
    "SERPER_API_KEY",
    "SERPAPI_API_KEY",
    "UV_CACHE_DIR",
]


class RunOptimizationOutput(NamedTuple):
    session_id: str
    best_commit: str | None


def _benchmark_harness_root() -> Path:
    return Path(__file__).resolve().parents[2] / "evaluator"


def _default_sessions_root() -> Path:
    home = Path(os.environ.get("VERO_HOME", Path.home() / ".vero"))
    return home.expanduser().resolve() / "sessions" / "benchmarks"


def _resolved_model(model: str) -> str:
    return MODELS.get(model, model)


def _instruction(task: OptimizationTask) -> str:
    threshold = (
        f" A score of {task.score_threshold:g} is a useful target."
        if task.score_threshold is not None
        else ""
    )
    return (
        f"Improve the program for benchmark {task.task!r}. The objective is to "
        f"{task.direction} the aggregate {task.metric!r} metric on the immutable "
        f"{task.partition!r} evaluation set.{threshold} Use evaluate_current to "
        "measure meaningful checkpoints and get_evaluation_budget before spending "
        "the final evaluation. Make only changes that improve the target program."
    )


def _make_coding_agent(
    agent_name: Literal["vero", "claude"],
    model: str,
):
    resolved_model = _resolved_model(model)
    if agent_name == "claude":
        from claude_agent_sdk import ClaudeAgentOptions
        from vero.agents import ClaudeCodeAgent

        claude_model = resolved_model.split("/")[-1]
        return ClaudeCodeAgent(
            options=ClaudeAgentOptions(
                model=claude_model,
                permission_mode="bypassPermissions",
                allowed_tools=["WebSearch", "WebFetch", "Task", "Bash"],
            )
        )

    from agents import Agent, ModelSettings
    from vero.agents import VeroAgent
    from vero_benchmarking.utils import get_model

    return VeroAgent(
        oai_agent=Agent(
            name="VeroAgent",
            model=get_model(resolved_model),
            model_settings=ModelSettings(include_usage=True),
        )
    )


def _case_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def _evaluation_set(
    task: OptimizationTask,
    *,
    case_count: int,
    case_ids: list[str] | None,
) -> EvaluationSet:
    if case_ids:
        selection = CaseIds(ids=case_ids)
    elif (
        task.max_cases_per_evaluation is not None
        and task.max_cases_per_evaluation < case_count
    ):
        selection = CaseRange(stop=task.max_cases_per_evaluation)
    else:
        return EvaluationSet(name=task.task, partition=task.partition)
    return EvaluationSet(
        name=task.task,
        partition=task.partition,
        selection=selection,
    )


async def build_benchmark_session(
    *,
    task_name: str,
    model: str,
    task_definition: OptimizationTask | None = None,
    agent_name: Literal["vero", "claude"] | None = "vero",
    session_id: str | None = None,
    session_dir: Path | str | None = None,
    evaluation_budget: int | None = None,
    max_candidates: int | None = None,
    max_rounds: int = 100,
    max_turns: int = 200,
    evaluation_concurrency: int = 100,
    candidate_concurrency: int = 1,
    evaluation_timeout: float = DEFAULT_EVAL_TIMEOUT,
    case_timeout: float = DEFAULT_CASE_TIMEOUT,
    retry: RetryPolicy | None = None,
    case_ids: list[str] | None = None,
    seed: int | None = DEFAULT_SEED,
    parameters: dict | None = None,
    metadata: dict | None = None,
) -> OptimizationSession:
    """Compose one canonical benchmark session without starting it."""

    task = task_definition or load_task(task_name)
    if session_dir is not None:
        resolved_session_dir = Path(session_dir).expanduser().resolve()
        resolved_session_id = session_id or resolved_session_dir.name
    else:
        resolved_session_id = session_id or str(uuid4())
        resolved_session_dir = (
            _default_sessions_root() / task_name / resolved_session_id
        )
    cases_path = materialize_cases(
        dataset_path=task.dataset_path,
        partition=task.partition,
        output_path=resolved_session_dir / "inputs" / "cases.jsonl",
    )
    evaluation_set = _evaluation_set(
        task,
        case_count=_case_count(cases_path),
        case_ids=case_ids,
    )

    resolved_evaluation_budget = (
        task.evaluation_budget if evaluation_budget is None else evaluation_budget
    )
    if resolved_evaluation_budget < 1:
        raise ValueError("evaluation_budget must include at least the baseline")
    if max_candidates is None:
        max_candidates = resolved_evaluation_budget - 1 if agent_name is not None else 0
    if max_candidates < 0:
        raise ValueError("max_candidates must be non-negative")
    if agent_name is None and max_candidates:
        raise ValueError("baseline-only sessions cannot produce candidates")
    if max_candidates > resolved_evaluation_budget - 1:
        raise ValueError(
            "max_candidates must leave at least one evaluation for the baseline"
        )

    producer = None
    producers = {}
    instruction = _instruction(task)
    if agent_name is not None:
        producer = AgentCandidateProducer(
            _make_coding_agent(agent_name, model),
            prompt=instruction,
            max_turns=max_turns,
        )
        producers["default"] = producer

    backend = PythonTaskBackend(
        PythonTaskBackendConfig(
            harness_root=str(_benchmark_harness_root()),
            module=task.resolved_module,
            task=task.task,
            cases_path=str(cases_path),
            evaluation_set_name=task.task,
            partition=task.partition,
            passthrough_environment=PASSTHROUGH_ENVIRONMENT,
        )
    )
    budget = EvaluationBudget(
        backend_id=BACKEND_ID,
        evaluation_set_key=evaluation_set.budget_key(BACKEND_ID),
        total_runs=resolved_evaluation_budget,
        total_cases=task.total_case_budget,
        max_cases_per_run=task.max_cases_per_evaluation,
    )
    session_metadata = {
        "benchmark": task_name,
        "target_task": task.task,
        "partition": task.partition,
        "model": _resolved_model(model),
        "agent": agent_name or "baseline",
        **(metadata or {}),
    }
    session = await create_local_optimization_session(
        project_path=task.project_path,
        session_dir=resolved_session_dir,
        session_id=resolved_session_id,
        backend_id=BACKEND_ID,
        backend=backend,
        objective=ObjectiveSpec(
            selector=MetricSelector(metric=task.metric),
            direction=task.direction,
        ),
        evaluation_set=evaluation_set,
        strategy=SequentialStrategy(instruction=instruction),
        producers=producers,
        parameters={
            "model": _resolved_model(model),
            **task.parameters,
            **(parameters or {}),
        },
        limits=EvaluationLimits(
            timeout_seconds=evaluation_timeout,
            case_timeout_seconds=case_timeout,
            max_concurrency=evaluation_concurrency,
            retry=retry or RetryPolicy(),
        ),
        budgets=[budget],
        seed=seed,
        max_candidates=max_candidates,
        max_rounds=max_rounds,
        max_concurrency=candidate_concurrency,
        metadata=session_metadata,
    )
    if producer is not None:
        producer.artifacts = session.artifacts
    return session


async def run_optimization(
    session: OptimizationSession,
    *,
    batch_id: str | None = None,
    config_name: str | None = None,
    skip_initial_eval: bool = False,
) -> RunOptimizationOutput:
    """Run a composed benchmark session and return its durable identity and winner."""

    if batch_id is not None:
        session.metadata["batch_id"] = batch_id
    if config_name is not None:
        session.metadata["config_name"] = config_name
    result = await session.run(skip_baseline_evaluation=skip_initial_eval)
    best_commit = (
        result.best.request.candidate.version if result.best is not None else None
    )
    return RunOptimizationOutput(session.id, best_commit)


def evaluation_record_dataframe(
    record: EvaluationRecord,
    *,
    model: str,
    task_name: str,
) -> pd.DataFrame:
    candidate = record.request.candidate
    rows = []
    for case in record.report.cases:
        row = {
            "case_id": case.case_id,
            "status": case.status.value,
            "model": _resolved_model(model),
            "task": task_name,
            "split": record.request.evaluation_set.partition,
            "commit": candidate.version,
            "input": json.dumps(case.input, ensure_ascii=False),
            "output": json.dumps(case.output, ensure_ascii=False),
            "feedback": case.feedback,
            "errors": json.dumps(
                [error.model_dump(mode="json") for error in case.errors],
                ensure_ascii=False,
            ),
            **case.metrics,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def print_baseline_stats(df: pd.DataFrame, model: str) -> None:
    print(f"\n=== {model} ===")
    print(f"  Cases: {len(df)}")
    if "score" in df.columns:
        scores = df["score"].dropna()
        print(f"  Mean score: {scores.mean():.4f}")
        print(f"  Errors: {(df['status'] == 'error').sum()}")


async def run_baseline(
    *,
    task_name: str,
    models: list[str],
    output_path: Path | None = None,
    sessions_root: Path | None = None,
    case_ids: list[str] | None = None,
    evaluation_concurrency: int = 100,
    evaluation_timeout: float = DEFAULT_EVAL_TIMEOUT,
    case_timeout: float = DEFAULT_CASE_TIMEOUT,
    retry: RetryPolicy | None = None,
) -> pd.DataFrame:
    """Evaluate the current target version once for each model."""

    all_rows: list[dict] = []
    for model in models:
        session_id = str(uuid4())
        session_dir = (
            sessions_root / task_name / session_id
            if sessions_root is not None
            else None
        )
        session = await build_benchmark_session(
            task_name=task_name,
            model=model,
            agent_name=None,
            session_id=session_id,
            session_dir=session_dir,
            max_candidates=0,
            case_ids=case_ids,
            evaluation_concurrency=evaluation_concurrency,
            evaluation_timeout=evaluation_timeout,
            case_timeout=case_timeout,
            retry=retry,
        )
        result: OptimizationResult = await session.run()
        frame = evaluation_record_dataframe(
            result.baseline,
            model=model,
            task_name=task_name,
        )
        print_baseline_stats(frame, model)
        all_rows.extend(frame.to_dict(orient="records"))

    combined = pd.DataFrame(all_rows)
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_RESULTS_DIR / f"{task_name}_{timestamp}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path)
    print(f"\nResults saved to: {output_path}")
    return combined


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True)
    parser.add_argument("--evaluation-concurrency", type=int, default=100)
    parser.add_argument(
        "--evaluation-timeout", type=float, default=DEFAULT_EVAL_TIMEOUT
    )
    parser.add_argument("--case-timeout", type=float, default=DEFAULT_CASE_TIMEOUT)
    parser.add_argument("--retry-max-attempts", type=int, default=3)
    parser.add_argument("--retry-initial-delay", type=float, default=4.0)
    parser.add_argument("--retry-maximum-delay", type=float, default=120.0)
    parser.add_argument("--retry-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--no-retry-on-timeout",
        action="store_false",
        dest="retry_on_timeout",
    )


def _retry_policy(arguments: argparse.Namespace) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=arguments.retry_max_attempts,
        initial_delay_seconds=arguments.retry_initial_delay,
        maximum_delay_seconds=arguments.retry_maximum_delay,
        multiplier=arguments.retry_multiplier,
        retry_on_timeout=arguments.retry_on_timeout,
    )


def _run_optimize(arguments: argparse.Namespace) -> None:
    async def run() -> RunOptimizationOutput:
        session = await build_benchmark_session(
            task_name=arguments.task,
            model=arguments.model,
            agent_name=arguments.agent,
            session_id=arguments.session_id,
            session_dir=arguments.session_dir,
            max_candidates=arguments.max_candidates,
            max_rounds=arguments.max_rounds,
            max_turns=arguments.max_turns,
            evaluation_concurrency=arguments.evaluation_concurrency,
            candidate_concurrency=arguments.candidate_concurrency,
            evaluation_timeout=arguments.evaluation_timeout,
            case_timeout=arguments.case_timeout,
            retry=_retry_policy(arguments),
        )
        return await run_optimization(
            session,
            batch_id=arguments.batch_id,
            config_name=arguments.config_name,
            skip_initial_eval=arguments.skip_initial_eval,
        )

    output = asyncio.run(run())
    print(f"Session: {output.session_id}")
    print(f"Best commit: {output.best_commit or 'none'}")


def _run_baseline(arguments: argparse.Namespace) -> None:
    asyncio.run(
        run_baseline(
            task_name=arguments.task,
            models=arguments.models,
            output_path=Path(arguments.output).resolve() if arguments.output else None,
            case_ids=arguments.case_ids,
            evaluation_concurrency=arguments.evaluation_concurrency,
            evaluation_timeout=arguments.evaluation_timeout,
            case_timeout=arguments.case_timeout,
            retry=_retry_policy(arguments),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize = subparsers.add_parser("optimize", help="Run a coding-agent optimizer")
    _add_runtime_arguments(optimize)
    optimize.add_argument("--agent", choices=["vero", "claude"], default="vero")
    optimize.add_argument("--model", default="sonnet")
    optimize.add_argument("--session-id")
    optimize.add_argument("--session-dir", type=Path)
    optimize.add_argument("--max-candidates", type=int)
    optimize.add_argument("--max-rounds", type=int, default=100)
    optimize.add_argument("--max-turns", type=int, default=200)
    optimize.add_argument("--candidate-concurrency", type=int, default=1)
    optimize.add_argument("--batch-id")
    optimize.add_argument("--config-name")
    optimize.add_argument("--skip-initial-eval", action="store_true")
    optimize.set_defaults(handler=_run_optimize)

    baseline = subparsers.add_parser("baseline", help="Evaluate current baselines")
    _add_runtime_arguments(baseline)
    baseline.add_argument("--models", nargs="+", required=True)
    baseline.add_argument("--case-ids", nargs="*")
    baseline.add_argument("--output")
    baseline.set_defaults(handler=_run_baseline)

    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
