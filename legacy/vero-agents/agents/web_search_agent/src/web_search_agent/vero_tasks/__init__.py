"""Task definitions for web-search-agent benchmarks.

Import this module to register all tasks with the VeroTask registry.
"""

from .facts_search import facts_search_task
from .hle import hle_task
from .simple_qa import simple_qa_task

__all__ = [
    "facts_search_task",
    "hle_task",
    "simple_qa_task",
]
