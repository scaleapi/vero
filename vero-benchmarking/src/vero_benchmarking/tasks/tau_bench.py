"""
Tau Bench benchmarking tasks.

Tau Bench is a benchmark for evaluating AI agents on real-world tasks
involving tool use and multi-step reasoning.
"""

from vero_benchmarking.tasks.base import OptimizationTask
from vero_benchmarking.constants import DEFAULT_DATASETS_DIR
from vero_benchmarking.utils import get_path_to_vero_agents

path_to_vero_agents = get_path_to_vero_agents()
path_to_tau_bench = path_to_vero_agents / "agents/tau-bench"

tau_bench_task = OptimizationTask(
    project_path=path_to_tau_bench,
    dataset_path=DEFAULT_DATASETS_DIR / "tau_bench_retail",
    score_threshold=0.95,
    max_cases_per_evaluation=512,
    evaluation_budget=8,
    task="retail",
)

TAU_BENCH_TASKS = {
    "tau-bench": tau_bench_task,
}
