#!/usr/bin/env python3
"""
Run budget ablation experiments.

Ablates train+validation budget for a single config and task,
running multiple iterations at each budget level.

Usage:
    # Ablate budget on math with default config
    uv run python scripts/run_ablation.py --task math

    # Ablate with specific config and budgets
    uv run python scripts/run_ablation.py --task math --config vero-cookbook-sonnet --budgets 2 4 8 16

    # Dry run
    uv run python scripts/run_ablation.py --task math --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vero_benchmarking.runner import (  # noqa: E402
    MODELS,
    run_optimization,
)
from vero_benchmarking.tasks import BENCHMARK_TASKS  # noqa: E402

# Import scaffolds and build_policy from run_benchmark
from run_benchmark import (  # noqa: E402
    DEFAULT_CONFIGS,
    SCAFFOLDS,
    build_policy,
    get_manifest_path,
    is_already_completed,
    write_to_manifest,
)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_BUDGETS = [2, 4, 8, 16]
DEFAULT_CONFIG = "vero-cookbook-sonnet"
DEFAULT_WANDB_PROJECT = "vero-icml-ablation"
DEFAULT_ITERATIONS = 3


# =============================================================================
# Execution
# =============================================================================


async def run_single_ablation(
    config_name: str,
    scaffold_name: str,
    model_name: str,
    task_name: str,
    budget: int,
    batch_id: str,
    wandb_project: str,
    push_to_origin: bool,
    dry_run: bool = False,
) -> bool:
    """Run a single ablation experiment. Returns True on success."""
    experiment_name = f"{config_name}-budget{budget}"

    print(f"\n{'=' * 60}")
    print(f"Running: {experiment_name} | {task_name}")
    print(f"{'=' * 60}")

    if dry_run:
        print(f"  Model: {MODELS[model_name]}")
        print(f"  Train budget: {budget}")
        print(f"  Validation budget: {budget}")
        return True

    try:
        from vero_benchmarking.tasks import load_task as _load_task

        task = _load_task(task_name)
        policy = build_policy(
            scaffold_name=scaffold_name,
            model_name=model_name,
            task_name=task_name,
            enable_wandb=True,
            wandb_project=wandb_project,
            train_budget=budget,
            validation_budget=budget,
        )

        result = await run_optimization(
            policy,
            batch_id=batch_id,
            config_name=experiment_name,
            push_to_origin=push_to_origin,
            eval_split=task.eval_split,
        )

        print(f"\n✓ Completed: {experiment_name} | {task_name}")
        print(f"  Session ID: {result.session_id}")
        print(f"  Best commit: {result.best_commit}")

        write_to_manifest(batch_id, experiment_name, task_name, result.session_id, result.best_commit)
        return True

    except Exception as e:
        print(f"\n✗ Failed: {experiment_name} | {task_name}")
        print(f"  Error: {e}")
        return False

    finally:
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass


async def run_ablation(
    experiments: list[tuple[str, str, str, str, int]],  # (config_name, scaffold, model, task, budget)
    batch_id: str,
    wandb_project: str,
    push_to_origin: bool,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> tuple[int, int, int]:
    """Run all ablation experiments. Returns (succeeded, failed, skipped) counts."""
    succeeded = 0
    failed = 0
    skipped = 0

    print("=" * 60)
    print("Vero Budget Ablation Experiments")
    print("=" * 60)
    print(f"Batch ID: {batch_id or '(none)'}")
    print(f"Experiments: {len(experiments)}")
    print(f"Dry run: {dry_run}")
    print("=" * 60)

    for i, (config_name, scaffold_name, model_name, task, budget) in enumerate(experiments, 1):
        experiment_name = f"{config_name}-budget{budget}"
        if is_already_completed(batch_id, experiment_name, task):
            print(f"\n[SKIP] {experiment_name} | {task} (already completed)")
            skipped += 1
            continue

        print(f"\n[{i}/{len(experiments)}] {experiment_name} | {task}")

        success = await run_single_ablation(
            config_name=config_name,
            scaffold_name=scaffold_name,
            model_name=model_name,
            task_name=task,
            budget=budget,
            batch_id=batch_id,
            wandb_project=wandb_project,
            push_to_origin=push_to_origin,
            dry_run=dry_run,
        )

        if success:
            succeeded += 1
        else:
            failed += 1
            if not continue_on_error:
                print("\nStopping (use --continue-on-error to proceed)")
                break

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total: {len(experiments)}")
    if skipped:
        print(f"Skipped: {skipped}")
    if dry_run:
        print("Mode: DRY RUN")
    else:
        print(f"Succeeded: {succeeded}")
        print(f"Failed: {failed}")
    print("=" * 60)

    return succeeded, failed, skipped


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run budget ablation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available configs: {", ".join(DEFAULT_CONFIGS.keys())}
Available tasks: {", ".join(BENCHMARK_TASKS)}
Default budgets: {DEFAULT_BUDGETS}
""",
    )

    parser.add_argument(
        "--task", type=str, required=True,
        help=f"Task to ablate. Available: {', '.join(BENCHMARK_TASKS)}",
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        choices=list(DEFAULT_CONFIGS.keys()),
        help=f"Config to use (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS,
        help=f"Budget levels to test (default: {DEFAULT_BUDGETS})",
    )
    parser.add_argument(
        "-n", "--iterations", type=int, default=DEFAULT_ITERATIONS,
        help=f"Iterations per budget level (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument("--batch-id", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument(
        "--sgp-account-id", type=str, default=os.environ.get("SGP_ACCOUNT_ID", ""),
    )
    parser.add_argument("--push-to-origin", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup SGP tracing if configured
    if args.sgp_account_id:
        from vero.logging import setup_sgp_agents_sdk_tracing

        setup_sgp_agents_sdk_tracing(account_id=args.sgp_account_id)

    scaffold_name, model_name = DEFAULT_CONFIGS[args.config]

    # Build experiment grid: config x budget x iteration
    experiments: list[tuple[str, str, str, str, int]] = []
    for budget in sorted(args.budgets):
        for i in range(args.iterations):
            iter_suffix = f"-iter{i + 1}" if args.iterations > 1 else ""
            name = f"{args.config}{iter_suffix}"
            experiments.append((name, scaffold_name, model_name, args.task, budget))

    if not experiments:
        print("No experiments to run")
        return 0

    print(f"Config: {args.config}")
    print(f"Task: {args.task}")
    print(f"Budgets: {args.budgets}")
    print(f"Iterations: {args.iterations}")
    print(f"Total experiments: {len(experiments)}")

    succeeded, failed, skipped = asyncio.run(
        run_ablation(
            experiments=experiments,
            batch_id=args.batch_id,
            wandb_project=args.wandb_project,
            push_to_origin=args.push_to_origin,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
    )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
