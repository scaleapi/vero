from pathlib import Path

import pytest

from vero_benchmarking.tasks.base import OptimizationTask


def test_task_derives_python_module_from_target_package():
    task = OptimizationTask(
        project_path=Path("/targets/generic-agent"),
        dataset_path=Path("/datasets/cases.jsonl"),
        task="gsm8k",
    )

    assert task.resolved_module == "generic_agent.vero_tasks.gsm8k"


def test_unknown_target_requires_explicit_module():
    task = OptimizationTask(
        project_path=Path("/targets/custom"),
        dataset_path=Path("/datasets/cases.jsonl"),
        task="custom",
    )

    with pytest.raises(ValueError, match="module must be explicit"):
        _ = task.resolved_module


@pytest.mark.parametrize("evaluation_budget", [0, -1])
def test_task_budget_includes_baseline(evaluation_budget):
    with pytest.raises(ValueError, match="at least the baseline"):
        OptimizationTask(
            project_path="target",
            dataset_path="cases.jsonl",
            task="task",
            evaluation_budget=evaluation_budget,
        )
