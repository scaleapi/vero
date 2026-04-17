"""VeroTask evaluation scaffold for tau-bench.

This file defines the evaluation tasks and dataset generation for vero optimization.
"""

from vero.core.db.result import TaskResult
from vero.core.evaluation import EvaluationParameters
from vero.core.task import create_task

from tau_bench.run import run
from tau_bench.types import EnvRunResult
from tau_bench.vero_tasks.utils import TauBenchTask, build_run_config, collate_results

retail_task = create_task("retail", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])


@retail_task("run_inference")
async def run_inference(task: TauBenchTask, evaluation_parameters: EvaluationParameters):
    """Run inference on a single task. For TauBench, we use canonical inference + evaluation logic defined under run.py"""
    return None


@retail_task("run_evaluation", batch=True)
async def run_evaluation(
    tasks: list[TauBenchTask],
    outputs: list[None],  # run_inferences doesn't return anything
    evaluation_parameters: EvaluationParameters,
) -> TaskResult:
    """Evaluate the inference output for a single task.

    Args:
        task: The task data (raw dict from the dataset, or custom object if using @task("create_task"))
        output: Output from run_inference
        evaluation_parameters: Evaluation parameters

    Returns:
        TaskResult with score and optional feedback"""

    run_config = build_run_config(env="retail", tasks=tasks, evaluation_parameters=evaluation_parameters)
    results: list[EnvRunResult] = await run(run_config)
    return collate_results(results)
