import json
from typing import Literal, TypedDict

from datasets import Dataset, DatasetDict
from vero.core.db.result import TaskResult
from vero.core.evaluation import EvaluationParameters, TaskParameters

from tau_bench.constants import DEFAULT_AGENT_STRATEGY
from tau_bench.types import EnvRunResult, RunConfig


class TauBenchTask(TypedDict):
    task_id: int
    task_split: str


class TauBenchParameters(TaskParameters):
    """Typed parameters for tau-bench evaluation."""

    agent_strategy: str = DEFAULT_AGENT_STRATEGY
    model_provider: str = "openai"
    user_model_provider: str = "openai"
    model: str = "gpt-4.1-mini"
    user_model: str = "gpt-4.1-mini"
    num_trials: int = 1
    temperature: float = 0.0
    log_dir: str = "results"
    seed: int = 10
    shuffle: int = 0
    user_strategy: str = "llm"
    few_shot_displays_path: str | None = None


def build_run_config(
    env: Literal["retail", "airline"], tasks: list[TauBenchTask], evaluation_parameters: EvaluationParameters
) -> RunConfig:
    params = evaluation_parameters.parse_task_params(TauBenchParameters)
    task_split = tasks[0]["task_split"]

    return RunConfig(
        agent_strategy=params.agent_strategy,
        model_provider=params.model_provider,
        user_model_provider=params.user_model_provider,
        model=params.model,
        user_model=params.user_model,
        num_trials=params.num_trials,
        env=env,
        temperature=params.temperature,
        task_split=task_split,
        task_ids=[task["task_id"] for task in tasks],
        log_dir=params.log_dir,
        max_concurrency=evaluation_parameters.max_concurrency,
        seed=params.seed,
        shuffle=params.shuffle,
        user_strategy=params.user_strategy,
        few_shot_displays_path=params.few_shot_displays_path,
    )


def collate_results(results: list[EnvRunResult]) -> list[TaskResult]:
    from collections import defaultdict

    results_by_task = defaultdict(list)
    for result in results:
        results_by_task[result.task_id].append(result)

    task_results = []
    for _, trial_results in results_by_task.items():
        outputs = []
        for result in trial_results:
            if len(result.traj) > 1:
                outputs.append(result.traj[-1])
            else:
                outputs.append(None)

        score = sum(result.reward for result in trial_results) / len(trial_results)
        trajectories = [json.dumps(result.traj) for result in trial_results]
        feedbacks_str = json.dumps(
            {f"trial_{i}": result.info.get("feedback", None) for i, result in enumerate(trial_results)}
        )

        trial_errors = [result.info.get("error", None) for result in trial_results]
        error = None
        if any(trial_errors):
            error = "\n".join([f"Trial {i}: {e}" for i, e in enumerate(trial_errors) if e is not None])

        trial_tracebacks = [result.info.get("traceback", None) for result in trial_results]
        traceback = None
        if any(trial_tracebacks):
            traceback = "\n".join([f"Trial {i}: {tb}" for i, tb in enumerate(trial_tracebacks) if tb is not None])

        task_result = TaskResult(
            output=outputs,
            score=score,
            error=error,
            error_traceback=traceback,
            execution_trace=trajectories,
            feedback=feedbacks_str,
        )
        task_results.append(task_result)

    return task_results


def create_tau_bench_dataset(env: str) -> DatasetDict:
    """Create a HuggingFace DatasetDict with task_id column for tau-bench tasks.

    Args:
        env: Environment name ("retail" or "airline")

    Returns:
        DatasetDict with train/dev/test splits (where available)
    """
    splits = {}

    if env == "retail":
        from tau_bench.envs.retail.tasks_dev import TASKS_DEV as dev_tasks
        from tau_bench.envs.retail.tasks_test import TASKS_TEST as test_tasks
        from tau_bench.envs.retail.tasks_train import TASKS_TRAIN as train_tasks

        splits["train"] = Dataset.from_dict(
            {"task_id": list(range(len(train_tasks))), "task_split": ["train"] * len(train_tasks)}
        )
        splits["validation"] = Dataset.from_dict(
            {"task_id": list(range(len(dev_tasks))), "task_split": ["dev"] * len(dev_tasks)}
        )
        splits["test"] = Dataset.from_dict(
            {"task_id": list(range(len(test_tasks))), "task_split": ["test"] * len(test_tasks)}
        )

    elif env == "airline":
        from tau_bench.envs.airline.tasks_test import TASKS as test_tasks

        # Airline only has test split
        splits["test"] = Dataset.from_dict(
            {"task_id": list(range(len(test_tasks))), "task_split": ["test"] * len(test_tasks)}
        )

    else:
        raise ValueError(f"Unknown environment: {env}")

    return DatasetDict(splits)


# Pre-built datasets for convenience
def get_retail_dataset() -> DatasetDict:
    return create_tau_bench_dataset("retail")


def get_airline_dataset() -> DatasetDict:
    return create_tau_bench_dataset("airline")


if __name__ == "__main__":
    from pathlib import Path

    out_dir = Path(__file__).parent.parent.parent / "datasets"
    out_dir.mkdir(exist_ok=True)

    print("Creating retail dataset...")
    retail_ds = get_retail_dataset()
    retail_ds.save_to_disk(out_dir / "retail")
    print(f"Saved retail dataset to {out_dir / 'retail'}")
    print(retail_ds)

    print("\nCreating airline dataset...")
    airline_ds = get_airline_dataset()
    airline_ds.save_to_disk(out_dir / "airline")
    print(f"Saved airline dataset to {out_dir / 'airline'}")
    print(airline_ds)
