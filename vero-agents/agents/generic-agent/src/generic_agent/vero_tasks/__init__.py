"""Task definitions for generic-agent benchmarks.

Import this module to register all tasks with the VeroTask registry.
"""

from .drop import drop_single_answer_task, drop_task
from .gaia import gaia_task
from .gpqa import gpqa_task
from .gsm8k import gsm8k_task
from .hotpot_qa import hotpot_qa_task
from .human_eval import human_eval_task
from .math import math_task
from .mbpp import mbpp_task

__all__ = [
    "drop_single_answer_task",
    "drop_task",
    "gaia_task",
    "gpqa_task",
    "gsm8k_task",
    "hotpot_qa_task",
    "human_eval_task",
    "math_task",
    "mbpp_task",
]
