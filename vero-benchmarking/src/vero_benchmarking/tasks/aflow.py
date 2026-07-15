"""
AFLOW benchmarking tasks.

These tasks use the AFLOW datasets which are curated subsets of standard benchmarks.
"""

from vero_benchmarking.tasks.base import OptimizationTask
from vero_benchmarking.constants import DEFAULT_DATASETS_DIR
from vero_benchmarking.utils import get_path_to_vero_agents

path_to_vero_agents = get_path_to_vero_agents()

GENERIC_AGENT_PATH = path_to_vero_agents / "agents/generic-agent"
SCORE_THRESHOLD = 0.95
BATCH_SIZE = 512
TRAIN_BUDGET = 8

# Mapping from AFLOW dataset names to HuggingFace dataset identifiers (i.e. path, revision tuples)
AFLOW_TO_HF_DATASETS = {
    "drop": "ucinlp/drop",
    "gsm8k": ("openai/gsm8k", "main"),
    "hotpotqa": ("hotpotqa/hotpot_qa", "fullwiki"),
    "humaneval": "openai/openai_humaneval",
    "math": "lighteval/MATH-Hard",
    "mbpp": "google-research-datasets/mbpp",
}

aflow_drop_task = OptimizationTask(
    project_path=GENERIC_AGENT_PATH,
    dataset_path=DEFAULT_DATASETS_DIR / "aflow_drop",
    score_threshold=SCORE_THRESHOLD,
    max_cases_per_evaluation=BATCH_SIZE,
    evaluation_budget=TRAIN_BUDGET,
    task="drop",
)

aflow_drop_single_answer_task = OptimizationTask(
    project_path=GENERIC_AGENT_PATH,
    dataset_path=DEFAULT_DATASETS_DIR / "aflow_drop_single_answer",
    score_threshold=SCORE_THRESHOLD,
    max_cases_per_evaluation=BATCH_SIZE,
    evaluation_budget=TRAIN_BUDGET,
    task="drop_single_answer",
)

aflow_gsm8k_task = OptimizationTask(
    project_path=GENERIC_AGENT_PATH,
    dataset_path=DEFAULT_DATASETS_DIR / "aflow_gsm8k",
    score_threshold=SCORE_THRESHOLD,
    max_cases_per_evaluation=BATCH_SIZE,
    evaluation_budget=TRAIN_BUDGET,
    task="gsm8k",
)

aflow_hotpotqa_task = OptimizationTask(
    project_path=GENERIC_AGENT_PATH,
    dataset_path=DEFAULT_DATASETS_DIR / "aflow_hotpotqa",
    score_threshold=SCORE_THRESHOLD,
    max_cases_per_evaluation=BATCH_SIZE,
    evaluation_budget=TRAIN_BUDGET,
    task="hotpot_qa",
)

aflow_humaneval_task = OptimizationTask(
    project_path=GENERIC_AGENT_PATH,
    dataset_path=DEFAULT_DATASETS_DIR / "aflow_humaneval",
    score_threshold=SCORE_THRESHOLD,
    max_cases_per_evaluation=BATCH_SIZE,
    evaluation_budget=TRAIN_BUDGET,
    task="human_eval",
)

aflow_humaneval_no_split_task = OptimizationTask(
    project_path=GENERIC_AGENT_PATH,
    dataset_path=DEFAULT_DATASETS_DIR / "aflow_humaneval_no_split",
    score_threshold=SCORE_THRESHOLD,
    max_cases_per_evaluation=BATCH_SIZE,
    evaluation_budget=TRAIN_BUDGET,
    task="human_eval",
)

aflow_math_task = OptimizationTask(
    project_path=GENERIC_AGENT_PATH,
    dataset_path=DEFAULT_DATASETS_DIR / "aflow_math",
    score_threshold=SCORE_THRESHOLD,
    max_cases_per_evaluation=BATCH_SIZE,
    evaluation_budget=TRAIN_BUDGET,
    task="math",
)

aflow_mbpp_task = OptimizationTask(
    project_path=GENERIC_AGENT_PATH,
    dataset_path=DEFAULT_DATASETS_DIR / "aflow_mbpp",
    score_threshold=SCORE_THRESHOLD,
    max_cases_per_evaluation=BATCH_SIZE,
    evaluation_budget=TRAIN_BUDGET,
    task="mbpp",
)

AFLOW_TASKS = {
    "drop-single": aflow_drop_single_answer_task,
    "gsm8k": aflow_gsm8k_task,
    "hotpotqa": aflow_hotpotqa_task,
    "humaneval-nosplit": aflow_humaneval_no_split_task,
    "math": aflow_math_task,
    "mbpp": aflow_mbpp_task,
}
