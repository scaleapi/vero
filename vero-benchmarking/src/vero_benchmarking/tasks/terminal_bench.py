"""
Terminal Bench 2.0 benchmarking task.

Terminal Bench tests whether an AI agent can complete real-world terminal tasks
in sandboxed environments. Budget is in samples (not runs) due to high cost per eval.

Only a single split ("test") exists — budget and eval both target it.
"""

from vero.tools.experiment_runner import SplitBudget

from vero_benchmarking.tasks.base import OptimizationTask
from vero_benchmarking.utils import get_path_to_vero_agents

path_to_vero_agents = get_path_to_vero_agents()

terminal_bench_task = OptimizationTask(
    project_path=path_to_vero_agents / "agents/KIRA",
    dataset_path=path_to_vero_agents / "agents/KIRA/datasets/terminal_bench",
    task="terminal_bench_2.0",
    budget=[SplitBudget(split="test", total_sample_budget=50)],
    eval_split="test",
)

TERMINAL_BENCH_TASKS = {
    "terminal-bench": terminal_bench_task,
}
