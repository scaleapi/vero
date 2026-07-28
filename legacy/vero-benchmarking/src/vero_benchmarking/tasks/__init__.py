"""
Benchmarking task definitions.

This module provides a unified registry of all optimization tasks.
"""

from vero_benchmarking.tasks.aflow import AFLOW_TASKS
from vero_benchmarking.tasks.base import OptimizationTask
from vero_benchmarking.tasks.facts_search import facts_search_task
from vero_benchmarking.tasks.gaia import GAIA_TASKS
from vero_benchmarking.tasks.gpqa import GPQA_TASKS
from vero_benchmarking.tasks.simple_qa import simple_qa_verified_with_val_agent_task
from vero_benchmarking.tasks.tau_bench import TAU_BENCH_TASKS
from vero_benchmarking.tasks.terminal_bench import TERMINAL_BENCH_TASKS

# =============================================================================
# Task Registry
# =============================================================================

ALL_TASKS: dict[str, OptimizationTask] = {
    **AFLOW_TASKS,
    **GAIA_TASKS,
    **GPQA_TASKS,
    **TAU_BENCH_TASKS,
    **TERMINAL_BENCH_TASKS,
    "simpleqa": simple_qa_verified_with_val_agent_task,
    "facts-search": facts_search_task,
}


def load_task(task_name: str) -> OptimizationTask:
    """Load an OptimizationTask by name."""
    if task_name not in ALL_TASKS:
        available = ", ".join(sorted(ALL_TASKS.keys()))
        raise ValueError(f"Unknown task: '{task_name}'. Available tasks: {available}.")

    return ALL_TASKS[task_name]


BENCHMARK_TASKS = [
    "gaia",
    "gpqa-nosplit",
    "math",
    "simpleqa",
    "tau-bench",
]

__all__ = [
    "OptimizationTask",
    "ALL_TASKS",
    "BENCHMARK_TASKS",
    "load_task",
]
