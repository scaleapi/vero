"""
GAIA benchmarking tasks.

GAIA (General AI Assistants) is a benchmark for evaluating AI assistants
on real-world tasks requiring multi-step reasoning, tool use, and web browsing.
"""

from vero_benchmarking.tasks.base import OptimizationTask
from vero_benchmarking.constants import DEFAULT_DATASETS_DIR
from vero_benchmarking.utils import get_path_to_vero_agents

path_to_vero_agents = get_path_to_vero_agents()

gaia_task = OptimizationTask(
    project_path=path_to_vero_agents / "agents/generic-agent",
    dataset_path=DEFAULT_DATASETS_DIR / "gaia_pure_language",
    score_threshold=0.95,
    max_cases_per_evaluation=512,
    evaluation_budget=8,
    task="gaia",
    partition="validation",
)

GAIA_TASKS = {
    "gaia": gaia_task,
}
