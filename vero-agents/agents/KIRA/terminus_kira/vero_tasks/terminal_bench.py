"""VeroTask for Terminal-Bench 2.0 via Harbor.

Wraps Harbor's Job API to run TerminusKira (or any Harbor agent) on
Terminal-Bench tasks within vero's evaluation framework.

Inference is a no-op — Harbor handles both agent execution and verification.
Evaluation is batched: all tasks in a batch are run as a single Harbor Job.
"""

from __future__ import annotations

from harbor import Job
from vero_tasks import TaskContext, TaskResult, create_task

from terminus_kira.vero_tasks.utils import (
    TerminalBenchTask,
    build_job_config,
    collate_results,
    load_trial_results,
)

terminal_bench_2 = create_task("terminal_bench_2.0", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])


@terminal_bench_2("run_inference")
async def run_inference(task: TerminalBenchTask, evaluation_parameters: TaskContext):
    """No-op — Harbor handles inference and evaluation together."""
    return None


@terminal_bench_2("run_evaluation", batch=True)
async def run_evaluation(
    tasks: list[TerminalBenchTask],
    outputs: list[None],
    evaluation_parameters: TaskContext,
) -> list[TaskResult]:
    """Run a Harbor Job for a batch of Terminal-Bench tasks.

    Builds a JobConfig from evaluation_parameters, runs the Job, and
    collates Harbor TrialResults into vero TaskResults.
    """
    job_config = build_job_config(tasks=tasks, evaluation_parameters=evaluation_parameters)
    job = Job(job_config)
    await job.run()
    # Harbor doesn't populate trial_results on the returned JobResult,
    # so we load them from disk after the job completes.
    trial_results = load_trial_results(job.job_dir)
    return collate_results(trial_results=trial_results, tasks=tasks, job_dir=job.job_dir)
