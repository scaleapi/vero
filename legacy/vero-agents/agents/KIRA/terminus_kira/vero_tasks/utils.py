"""Utilities for the TerminalBench vero_task integration.

Provides config builders, result collation, and dataset creation for
wrapping Harbor's Job API within vero's evaluation framework.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from datasets import Dataset, DatasetDict
from harbor import JobConfig
from harbor.models.job.config import OrchestratorConfig, RegistryDatasetConfig, RetryConfig
from harbor.models.registry import RemoteRegistryInfo
from harbor.models.trial.config import AgentConfig, EnvironmentConfig
from harbor.models.trial.result import TrialResult
from pydantic import Field
from vero.core.db.result import TaskResult
from vero.core.evaluation import EvaluationParameters, TaskParameters


def load_trial_results(job_dir: Path) -> list[TrialResult]:
    """Load all trial results from a completed Harbor job directory.

    Harbor doesn't populate trial_results on the JobResult returned by job.run(),
    so we read the individual trial result.json files from disk.
    """
    results = []
    for result_file in job_dir.glob("*/result.json"):
        # Skip the job-level result.json
        if result_file.parent == job_dir:
            continue
        results.append(TrialResult.model_validate_json(result_file.read_text()))
    return results


class TerminalBenchTask(TypedDict):
    task_name: str


# Default Harbor configuration for TerminusKira
DEFAULT_AGENT_IMPORT_PATH = "terminus_kira.terminus_kira:TerminusKira"
DEFAULT_MODEL = "anthropic/claude-haiku-4-5"
DEFAULT_DATASET = "terminal-bench"
DEFAULT_DATASET_VERSION = "2.0"
DEFAULT_ENVIRONMENT = "modal"
PUBLIC_REGISTRY_URL = "https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json"


class KiraParameters(TaskParameters):
    """Typed parameters for KIRA/TerminalBench evaluation."""

    agent_import_path: str = DEFAULT_AGENT_IMPORT_PATH
    model: str = DEFAULT_MODEL
    dataset_name: str = DEFAULT_DATASET
    dataset_version: str = DEFAULT_DATASET_VERSION
    environment: str = DEFAULT_ENVIRONMENT
    n_attempts: int = 1
    max_retries: int = 2
    agent_kwargs: dict = Field(default_factory=dict)
    jobs_dir: str = "jobs"


def build_job_config(
    tasks: list[TerminalBenchTask],
    evaluation_parameters: EvaluationParameters,
) -> JobConfig:
    """Build a Harbor JobConfig from vero evaluation parameters.

    Args:
        tasks: List of tasks to evaluate, each with a task_name.
        evaluation_parameters: Vero evaluation parameters with typed KiraParameters.
    """
    params = evaluation_parameters.parse_task_params(KiraParameters)

    task_names = [t["task_name"] for t in tasks]

    return JobConfig(
        jobs_dir=Path(params.jobs_dir),
        n_attempts=params.n_attempts,
        orchestrator=OrchestratorConfig(
            n_concurrent_trials=evaluation_parameters.max_concurrency,
            quiet=True,
            retry=RetryConfig(
                max_retries=params.max_retries,
                include_exceptions=[
                    "ExecutionError",
                    "RuntimeError",
                    "EnvironmentStartTimeoutError",
                    "ConnectError",
                ],
                wait_multiplier=10.0,
                min_wait_sec=10.0,
                max_wait_sec=120.0,
            ),
        ),
        environment=EnvironmentConfig(
            type=params.environment,
        ),
        agents=[
            AgentConfig(
                import_path=params.agent_import_path,
                model_name=params.model,
                kwargs=params.agent_kwargs,
            ),
        ],
        datasets=[
            RegistryDatasetConfig(
                registry=RemoteRegistryInfo(url=PUBLIC_REGISTRY_URL),
                name=params.dataset_name,
                version=params.dataset_version,
                task_names=task_names,
            ),
        ],
    )


def _load_trajectory(trial_dir: Path) -> dict | None:
    """Load the agent trajectory.json from a trial directory."""
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    if trajectory_path.exists():
        return json.loads(trajectory_path.read_text())
    return None


def _extract_reward_score(tr: TrialResult) -> float:
    """Extract a scalar score from a TrialResult's verifier rewards."""
    if tr.verifier_result and tr.verifier_result.rewards:
        rewards = tr.verifier_result.rewards
        # Use "pass" if available, then "reward", then mean of all
        if "pass" in rewards:
            return float(rewards["pass"])
        if "reward" in rewards:
            return float(rewards["reward"])
        return sum(float(v) for v in rewards.values()) / len(rewards)
    return 0.0


def collate_results(
    trial_results: list[TrialResult],
    tasks: list[TerminalBenchTask],
    job_dir: Path,
) -> list[TaskResult]:
    """Convert Harbor TrialResults to vero TaskResults, ordered by input tasks.

    Includes full agent trajectory, token/cost metrics, timing info, and
    verifier rewards in the result.
    """
    # Index trial results and their directories by task_name
    results_by_name: dict[str, list[TrialResult]] = {}
    for tr in trial_results:
        results_by_name.setdefault(tr.task_name, []).append(tr)

    task_results = []
    for task in tasks:
        name = task["task_name"]
        trials = results_by_name.get(name, [])

        if not trials:
            task_results.append(TaskResult(score=0.0, error=f"No trial result for task {name}"))
            continue

        # Score: average across trials (usually just 1)
        scores = [_extract_reward_score(tr) for tr in trials]
        score = sum(scores) / len(scores)

        # Output: rich summary per trial
        outputs = []
        for tr in trials:
            trial_output: dict = {
                "task_name": tr.task_name,
                "trial_name": tr.trial_name,
                "rewards": tr.verifier_result.rewards if tr.verifier_result else None,
                "agent_info": tr.agent_info.model_dump() if tr.agent_info else None,
            }
            if tr.agent_result:
                trial_output["tokens"] = {
                    "input": tr.agent_result.n_input_tokens,
                    "output": tr.agent_result.n_output_tokens,
                    "cache": tr.agent_result.n_cache_tokens,
                }
                trial_output["cost_usd"] = tr.agent_result.cost_usd
                trial_output["n_episodes"] = (
                    tr.agent_result.metadata.get("n_episodes") if tr.agent_result.metadata else None
                )
            if tr.started_at and tr.finished_at:
                trial_output["duration_sec"] = (tr.finished_at - tr.started_at).total_seconds()
            # Phase timing
            for phase in ("environment_setup", "agent_setup", "agent_execution", "verifier"):
                timing = getattr(tr, phase, None)
                if timing and timing.started_at and timing.finished_at:
                    trial_output[f"{phase}_sec"] = (timing.finished_at - timing.started_at).total_seconds()
            outputs.append(trial_output)

        # Execution trace: full trajectory from trajectory.json
        trajectories = []
        for tr in trials:
            trial_dir = job_dir / tr.trial_name
            trajectory = _load_trajectory(trial_dir)
            if trajectory:
                trajectories.append(trajectory)

        # Metrics: numeric values for aggregation
        metrics: dict[str, float] = {}
        for tr in trials:
            if tr.agent_result:
                metrics["input_tokens"] = metrics.get("input_tokens", 0) + (tr.agent_result.n_input_tokens or 0)
                metrics["output_tokens"] = metrics.get("output_tokens", 0) + (tr.agent_result.n_output_tokens or 0)
                metrics["cache_tokens"] = metrics.get("cache_tokens", 0) + (tr.agent_result.n_cache_tokens or 0)
                metrics["cost_usd"] = metrics.get("cost_usd", 0) + (tr.agent_result.cost_usd or 0)
            if tr.started_at and tr.finished_at:
                metrics["duration_sec"] = (
                    metrics.get("duration_sec", 0) + (tr.finished_at - tr.started_at).total_seconds()
                )

        # Feedback: verifier rewards as JSON
        feedback_data = {}
        for i, tr in enumerate(trials):
            feedback_data[f"trial_{i}"] = {
                "rewards": tr.verifier_result.rewards if tr.verifier_result else None,
                "score": scores[i],
            }
        feedback = json.dumps(feedback_data)

        # Errors: only report if the verifier didn't produce a passing result
        # (e.g. agent timed out but verifier confirmed the task was solved)
        errors = []
        tracebacks = []
        for i, tr in enumerate(trials):
            if tr.exception_info and not (tr.verifier_result and scores[i] > 0):
                errors.append(f"Trial {i}: {tr.exception_info.exception_type}: {tr.exception_info.exception_message}")
                if tr.exception_info.exception_traceback:
                    tracebacks.append(f"Trial {i}: {tr.exception_info.exception_traceback}")

        task_results.append(
            TaskResult(
                output=outputs[0] if len(outputs) == 1 else outputs,
                score=score,
                error="\n".join(errors) if errors else None,
                error_traceback="\n".join(tracebacks) if tracebacks else None,
                execution_trace=trajectories if trajectories else None,
                feedback=feedback,
                metrics=metrics,
            )
        )

    return task_results


def create_terminal_bench_dataset(
    dataset_name: str = DEFAULT_DATASET,
    dataset_version: str = DEFAULT_DATASET_VERSION,
) -> DatasetDict:
    """Create a HuggingFace DatasetDict from a Harbor terminal-bench dataset.

    Queries the Harbor registry for task names and creates a dataset with a
    single 'test' split (TerminalBench has no canonical train/test distinction).

    Args:
        dataset_name: Harbor dataset name (default: terminal-bench)
        dataset_version: Harbor dataset version (default: 2.0)

    Returns:
        DatasetDict with a 'test' split containing task_name column.
    """
    config = RegistryDatasetConfig(
        registry=RemoteRegistryInfo(url=PUBLIC_REGISTRY_URL),
        name=dataset_name,
        version=dataset_version,
    )

    task_configs = config.get_task_configs()
    task_names = sorted(tc.path.name for tc in task_configs)

    return DatasetDict(
        {
            "test": Dataset.from_dict({"task_name": task_names}),
        }
    )


if __name__ == "__main__":
    import sys

    dataset_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET
    dataset_version = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DATASET_VERSION

    out_dir = Path(__file__).parent.parent.parent / "datasets"
    out_dir.mkdir(exist_ok=True)

    print(f"Creating {dataset_name}@{dataset_version} dataset...")
    ds = create_terminal_bench_dataset(dataset_name, dataset_version)
    save_path = out_dir / dataset_name.replace("-", "_")
    ds.save_to_disk(save_path)
    print(f"Saved to {save_path}")
    print(ds)
    for split_name in ds:
        print(f"\n{split_name} tasks ({len(ds[split_name])}):")
        for row in ds[split_name]:
            print(f"  {row['task_name']}")
