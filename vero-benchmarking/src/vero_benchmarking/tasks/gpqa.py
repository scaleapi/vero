"""
GPQA benchmarking tasks.

GPQA Diamond is a subset of GPQA (Graduate-Level Google-Proof Q&A) containing
expert-level science questions that are difficult to answer using web search.
"""

from vero_benchmarking.tasks.base import OptimizationTask
from vero_benchmarking.constants import DEFAULT_DATASETS_DIR
from vero_benchmarking.utils import get_path_to_vero_agents

path_to_vero_agents = get_path_to_vero_agents()

gpqa_diamond_no_split_task = OptimizationTask(
    project_path=path_to_vero_agents / "agents/generic-agent",
    dataset_path=DEFAULT_DATASETS_DIR / "gpqa_diamond_no_split",
    score_threshold=0.95,
    batch_size=512,
    train_budget=8,
    validation_budget=None,  # No validation split
    task="gpqa",
    resource_namespace="gpqa",
)

GPQA_TASKS = {
    "gpqa-nosplit": gpqa_diamond_no_split_task,
}
