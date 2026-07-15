"""
Facts Search benchmarking tasks.
"""

from vero_benchmarking.tasks.base import OptimizationTask
from vero_benchmarking.constants import DEFAULT_DATASETS_DIR
from vero_benchmarking.utils import get_path_to_vero_agents

path_to_vero_agents = get_path_to_vero_agents()

facts_search_task = OptimizationTask(
    project_path=path_to_vero_agents / "agents/web_search_agent/",
    dataset_path=DEFAULT_DATASETS_DIR / "facts_search",
    score_threshold=0.9,
    max_cases_per_evaluation=512,
    evaluation_budget=8,
    task="facts_search",
)
