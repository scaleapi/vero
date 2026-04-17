"""Task definitions for pharma-summarizer benchmarks.

Import this module to register all tasks with the VeroTask registry.
"""

from .pharma_summarizer import pharma_summarizer_task

__all__ = [
    "pharma_summarizer_task",
]
