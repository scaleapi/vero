"""Task discovery and execution utilities.

This module provides isolated task discovery and execution via subprocess.
It is invoked by the vero evaluator using `uv run --project <path> python -m vero.core.task.utils`.

Commands:
    discover: Import task module and return registered task info as JSON
    run: Execute a specific task and return metrics as JSON

Task modules can be auto-discovered from the project's package ({package}.vero_tasks)
or specified explicitly via --task-module for tasks that live outside the agent project.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
import tomllib
from pathlib import Path

from vero.core.evaluation import EvaluationParameters
from vero.core.task.hooks import TaskHookRegistry
from vero.core.task.task import VeroTask

logger = logging.getLogger(__name__)

MODULE_PATH = "vero.core.task.utils"


def get_discover_cmd(task_module: str | None = None) -> list[str]:
    """Get the command suffix for task discovery.

    Args:
        task_module: Explicit module to import (e.g. "my_eval_tasks.vero_tasks").
            If None, auto-discovers from the project's package.

    Returns:
        Command list to append after uv run parameters.
    """
    cmd = ["python", "-m", MODULE_PATH, "discover"]
    if task_module:
        cmd.extend(["--task-module", task_module])
    return cmd


def get_run_cmd(
    task_name: str,
    params_file: str | Path,
    hooks: list[str] | None = None,
    task_module: str | None = None,
) -> list[str]:
    """Get the command suffix for task execution.

    Args:
        task_name: Name of the task to execute.
        params_file: Path to the params file.
        hooks: Optional list of hook names to execute.
        task_module: Explicit module to import.

    Returns:
        Command list to append after uv run parameters.
    """
    cmd = [
        "python",
        "-m",
        MODULE_PATH,
        "run",
        "--task",
        task_name,
        "--params-file",
        str(params_file),
    ]
    if task_module:
        cmd.extend(["--task-module", task_module])
    if hooks:
        for hook_name in hooks:
            cmd.extend(["--hook", hook_name])
    return cmd


def detect_package_name() -> str:
    """Detect package name from pyproject.toml in current working directory.

    Returns:
        Package name with hyphens converted to underscores.

    Raises:
        FileNotFoundError: If pyproject.toml doesn't exist.
        KeyError: If project.name is not defined.
    """
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    return pyproject["project"]["name"].replace("-", "_")


def _import_tasks(task_module: str | None = None) -> str:
    """Import the task module and return the package/module name used.

    Args:
        task_module: Explicit module path to import.
            If None, auto-discovers from {package}.vero_tasks.

    Returns:
        The module path that was imported.
    """
    VeroTask.clear_registry()
    if task_module:
        importlib.import_module(task_module)
        return task_module
    else:
        package = detect_package_name()
        module = f"{package}.vero_tasks"
        importlib.import_module(module)
        return module


def discover_tasks(task_module: str | None = None) -> dict:
    """Import task module and return registered task info.

    Args:
        task_module: Explicit module to import. If None, auto-discovers.

    Returns:
        Dictionary with module name and task metadata:
        {
            "package": "my_agent" or "my_eval_tasks.vero_tasks",
            "tasks": {
                "task_name": {
                    "name": "task_name",
                    "has_inference": True,
                    "has_evaluation": True,
                }
            }
        }
    """
    module = _import_tasks(task_module)

    return {
        "package": module,
        "tasks": {
            name: {
                "name": name,
                "has_inference": task.has("run_inference") or task.has("run_inference", batch=True),
                "has_evaluation": task.has("run_evaluation")
                or task.has("run_evaluation", batch=True),
                "required_env_vars": task.required_env_vars,
            }
            for name, task in VeroTask._registry.items()
        },
    }


async def run_task(
    task_name: str,
    params: str | None = None,
    params_file: Path | None = None,
    hooks: list[str] | None = None,
    task_module: str | None = None,
) -> dict:
    """Execute a task and return metrics.

    Args:
        task_name: Name of the registered task to execute.
        params: JSON string containing EvaluationParameters.
        params_file: Path to JSON file containing EvaluationParameters.
            One of params or params_file must be provided.
        hooks: List of hook names to execute before the task.
        task_module: Explicit module to import. If None, auto-discovers.

    Returns:
        Metrics dictionary from task execution.

    Raises:
        KeyError: If task_name is not found in registry.
        ValueError: If neither params nor params_file is provided.
    """
    if params_file is not None:
        params_json = params_file.read_text()
    elif params is not None:
        params_json = params
    else:
        raise ValueError("Either --params or --params-file must be provided")

    _import_tasks(task_module)

    task = VeroTask.get_task(task_name)
    evaluation_params = EvaluationParameters.model_validate_json(params_json)

    # Execute hooks before task execution
    if hooks:
        TaskHookRegistry.execute(hooks, evaluation_params)

    return await task.run(evaluation_params)


def main():
    parser = argparse.ArgumentParser(description="Vero task discovery and execution utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover", help="Discover registered tasks")
    discover_parser.add_argument(
        "--task-module",
        help="Explicit Python module to import for task registration (e.g. my_eval_tasks.vero_tasks)",
    )

    run_parser = subparsers.add_parser("run", help="Run a specific task")
    run_parser.add_argument("--task", required=True, help="Task name to execute")
    run_parser.add_argument(
        "--params",
        help="JSON string containing EvaluationParameters",
    )
    run_parser.add_argument(
        "--params-file",
        type=Path,
        help="Path to JSON file containing EvaluationParameters",
    )
    run_parser.add_argument(
        "--task-module",
        help="Explicit Python module to import for task registration",
    )
    run_parser.add_argument(
        "--hook",
        action="append",
        dest="hooks",
        default=[],
        help="Hook name to execute (can be specified multiple times)",
    )

    args = parser.parse_args()

    if args.command == "discover":
        result = discover_tasks(task_module=args.task_module)
        json.dump(result, sys.stdout)
        sys.stdout.flush()
    elif args.command == "run":
        params_file = getattr(args, "params_file", None)
        params = getattr(args, "params", None)
        hooks = getattr(args, "hooks", [])
        task_module = getattr(args, "task_module", None)
        if not params and not params_file:
            parser.error("Either --params or --params-file must be provided")
        result = asyncio.run(
            run_task(
                args.task,
                params=params,
                params_file=params_file,
                hooks=hooks,
                task_module=task_module,
            )
        )
        # Write metrics to file instead of stdout (stdout may have noise from libraries)
        metrics_path = Path(params_file).parent / "metrics.json"
        metrics_path.write_text(json.dumps(result))

    os._exit(0)


if __name__ == "__main__":
    main()
