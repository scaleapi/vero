#!/usr/bin/env python3
"""Run canonical VeRO evaluation-budget ablations."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_benchmark import (  # noqa: E402
    DEFAULT_CONFIGS,
    SCAFFOLDS,
    is_already_completed,
    write_to_manifest,
)
from vero_benchmarking.runner import (  # noqa: E402
    MODELS,
    build_benchmark_session,
    run_optimization,
)
from vero_benchmarking.tasks import BENCHMARK_TASKS  # noqa: E402


DEFAULT_BUDGETS = [2, 4, 8, 16]
DEFAULT_CONFIG = "vero-sonnet"
DEFAULT_ITERATIONS = 3


async def run_single_ablation(
    *,
    config_name: str,
    scaffold_name: str,
    model_name: str,
    task_name: str,
    budget: int,
    batch_id: str,
    dry_run: bool,
) -> bool:
    experiment_name = f"{config_name}-budget{budget}"
    print(f"\nRunning: {experiment_name} | {task_name}")
    if dry_run:
        print(f"  Agent: {SCAFFOLDS[scaffold_name]['agent_name']}")
        print(f"  Model: {MODELS[model_name]}")
        print(f"  Total evaluation runs: {budget}")
        return True

    try:
        session = await build_benchmark_session(
            task_name=task_name,
            model=MODELS[model_name],
            agent_name=SCAFFOLDS[scaffold_name]["agent_name"],
            evaluation_budget=budget,
            metadata={
                "batch_id": batch_id,
                "config_name": experiment_name,
                "ablation_evaluation_budget": budget,
            },
        )
        result = await run_optimization(
            session,
            batch_id=batch_id or None,
            config_name=experiment_name,
        )
    except Exception as error:
        print(f"Failed: {experiment_name} | {task_name}: {error}")
        return False

    write_to_manifest(
        batch_id,
        experiment_name,
        task_name,
        result.session_id,
        result.best_commit,
    )
    print(f"Completed: {experiment_name} | {task_name}")
    return True


async def run_ablation(
    experiments: list[tuple[str, str, str, str, int]],
    *,
    batch_id: str,
    dry_run: bool,
    continue_on_error: bool,
) -> tuple[int, int, int]:
    succeeded = failed = skipped = 0
    for config, scaffold, model, task, budget in experiments:
        experiment = f"{config}-budget{budget}"
        if is_already_completed(batch_id, experiment, task):
            skipped += 1
            continue
        success = await run_single_ablation(
            config_name=config,
            scaffold_name=scaffold,
            model_name=model,
            task_name=task,
            budget=budget,
            batch_id=batch_id,
            dry_run=dry_run,
        )
        if success:
            succeeded += 1
        else:
            failed += 1
            if not continue_on_error:
                break
    print(f"\nSucceeded: {succeeded}; failed: {failed}; skipped: {skipped}")
    return succeeded, failed, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=BENCHMARK_TASKS)
    parser.add_argument("--config", default=DEFAULT_CONFIG, choices=DEFAULT_CONFIGS)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("-n", "--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if any(budget < 1 for budget in arguments.budgets):
        raise ValueError("budgets must include at least one baseline evaluation")
    scaffold, model = DEFAULT_CONFIGS[arguments.config]
    experiments = [
        (
            f"{arguments.config}-iter{iteration + 1}"
            if arguments.iterations > 1
            else arguments.config,
            scaffold,
            model,
            arguments.task,
            budget,
        )
        for budget in sorted(arguments.budgets)
        for iteration in range(arguments.iterations)
    ]
    _, failed, _ = asyncio.run(
        run_ablation(
            experiments,
            batch_id=arguments.batch_id,
            dry_run=arguments.dry_run,
            continue_on_error=arguments.continue_on_error,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
