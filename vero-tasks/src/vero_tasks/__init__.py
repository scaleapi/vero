"""Narrow Python task protocol for VeRO evaluation harnesses."""

from vero_tasks.models import TaskContext, TaskOutput, TaskParameters, TaskResult, TaskT
from vero_tasks.task import TaskDefinition, create_task

__all__ = [
    "TaskContext",
    "TaskDefinition",
    "TaskOutput",
    "TaskParameters",
    "TaskResult",
    "TaskT",
    "create_task",
]
