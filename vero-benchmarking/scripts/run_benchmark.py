#!/usr/bin/env python3
"""
Run benchmarking experiments.

Usage:
    # Run a specific config on a task
    uv run python scripts/run_benchmark.py --config vero-cookbook-sonnet --task gsm8k

    # Run all default configs on a task
    uv run python scripts/run_benchmark.py --all-configs --task math

    # Run a scaffold with a specific model
    uv run python scripts/run_benchmark.py --scaffold vero-orchestrator-cookbook --model sonnet --task gpqa-nosplit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vero_benchmarking.runner import (  # noqa: E402
    MODELS,
    make_claude_code_policy,
    make_vero_policy,
    run_optimization,
)
from vero_benchmarking.tasks import BENCHMARK_TASKS  # noqa: E402

# =============================================================================
# Constants
# =============================================================================

DEFAULT_WANDB_PROJECT = "vero-icml-benchmarking"

# =============================================================================
# Scaffold Definitions
# =============================================================================

SCAFFOLDS: dict[str, dict[str, Any]] = {
    "vero-default": dict(factory=make_vero_policy),
    "vero-prompts-only": dict(
        factory=make_vero_policy,
        use_resources_only=True,
        instructions_template="instructions/few_shot_resources_only_instructions.j2",
    ),
    "vero-cookbook": dict(factory=make_vero_policy, enable_context_store=True),
    "vero-orchestrator": dict(
        factory=make_vero_policy,
        orchestrator_mode=True,
        instructions_template="instructions/few_shot_orchestrator_instructions.j2",
    ),
    "vero-orchestrator-cookbook": dict(
        factory=make_vero_policy,
        orchestrator_mode=True,
        enable_context_store=True,
        instructions_template="instructions/few_shot_orchestrator_instructions.j2",
    ),
    "claude-code-vmf": dict(factory=make_claude_code_policy),
    "claude-code-vmf-cookbook": dict(
        factory=make_claude_code_policy, enable_context_store=True
    ),
    "claude-code-pure": dict(
        factory=make_claude_code_policy,
        use_pure=True,
        prompt_template="prompts/claude_code_prompt.j2",
    ),
    "gepa": dict(runner="gepa", wandb_project="vero-gepa-benchmarking"),
}

# Pre-defined configs: scaffold + model
DEFAULT_CONFIGS: dict[str, tuple[str, str]] = {
    "vero-cookbook-sonnet": ("vero-cookbook", "sonnet"),
    "vero-orchestrator-cookbook-sonnet": ("vero-orchestrator-cookbook", "sonnet"),
    "vero-orchestrator-cookbook-opus": ("vero-orchestrator-cookbook", "opus"),
    "vero-orchestrator-cookbook-gpt": ("vero-orchestrator-cookbook", "gpt"),
    "vero-prompts-only-sonnet": ("vero-prompts-only", "sonnet"),
    "claude-code-vmf-cookbook-sonnet": ("claude-code-vmf-cookbook", "sonnet"),
    "claude-code-pure-sonnet": ("claude-code-pure", "sonnet"),
    "gepa-sonnet": ("gepa", "sonnet"),
}


def build_policy(
    scaffold_name: str, model_name: str, task_name: str, **extra_kwargs: Any
) -> None:
    """Build a Policy from a scaffold name, model name, and task name."""
    scaffold = SCAFFOLDS[scaffold_name].copy()
    factory = scaffold.pop("factory")
    model = MODELS[model_name]
    return factory(model=model, task_name=task_name, **scaffold, **extra_kwargs)


# =============================================================================
# Manifest
# =============================================================================


def get_manifest_path(batch_id: str) -> Path:
    from vero_benchmarking.constants import DEFAULT_LOG_DIR

    manifest_dir = DEFAULT_LOG_DIR / "batch_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir / f"{batch_id}.jsonl"


def is_already_completed(batch_id: str, config_name: str, task: str) -> bool:
    if not batch_id:
        return False
    manifest_path = get_manifest_path(batch_id)
    if not manifest_path.exists():
        return False
    try:
        with open(manifest_path) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if (
                        entry.get("config_name") == config_name
                        and entry.get("task") == task
                    ):
                        return True
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return False


def write_to_manifest(
    batch_id: str, config_name: str, task: str, session_id: str, best_commit: str | None
) -> None:
    if not batch_id:
        return
    import fcntl
    from datetime import datetime

    manifest_path = get_manifest_path(batch_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "config_name": config_name,
        "task": task,
        "session_id": session_id,
        "best_commit": best_commit,
        "timestamp": datetime.now().isoformat(),
    }

    with open(manifest_path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(entry) + "\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# =============================================================================
# Execution
# =============================================================================


async def run_single_experiment(
    config_name: str,
    scaffold_name: str,
    model_name: str,
    task_name: str,
    batch_id: str,
    wandb_project: str,
    push_to_origin: bool,
    dry_run: bool = False,
    max_turns: int | None = None,
    git_ref: str = "main",
    skip_initial_eval: bool = False,
) -> bool:
    """Run a single experiment. Returns True on success."""
    print(f"\n{'=' * 60}")
    print(f"Running: {config_name} | {task_name}")
    print(f"{'=' * 60}")

    scaffold = SCAFFOLDS[scaffold_name]
    is_gepa = scaffold.get("runner") == "gepa"

    if dry_run:
        print(f"  Model: {MODELS[model_name]}")
        if is_gepa:
            print("  Agent: GEPA")
        else:
            print(
                f"  Agent: {'VeroAgent' if scaffold.get('factory') == make_vero_policy else 'ClaudeCodeAgent'}"
            )
            print(
                f"  Instructions: {scaffold.get('instructions_template', 'instructions/few_shot_instructions.j2')}"
            )
        return True

    try:
        if is_gepa:
            from vero_benchmarking.gepa import run_gepa

            gepa_result = run_gepa(
                task_name=task_name,
                model=model_name,
                enable_wandb=True,
                wandb_project=scaffold.get("wandb_project", wandb_project),
            )
            session_id = gepa_result["session_id"]
            best_commit = None
        else:
            from vero_benchmarking.tasks import load_task as _load_task

            task = _load_task(task_name)
            extra_kwargs = dict(
                enable_wandb=True,
                wandb_project=wandb_project,
                git_ref=git_ref,
            )
            if max_turns is not None:
                extra_kwargs["max_turns"] = max_turns
            policy = build_policy(
                scaffold_name=scaffold_name,
                model_name=model_name,
                task_name=task_name,
                **extra_kwargs,
            )

            result = await run_optimization(
                policy,
                batch_id=batch_id,
                config_name=config_name,
                push_to_origin=push_to_origin,
                eval_split=task.eval_split,
                skip_initial_eval=skip_initial_eval,
            )
            session_id = result.session_id
            best_commit = result.best_commit

        print(f"\n✓ Completed: {config_name} | {task_name}")
        print(f"  Session ID: {session_id}")
        print(f"  Best commit: {best_commit}")

        write_to_manifest(batch_id, config_name, task_name, session_id, best_commit)
        return True

    except Exception as e:
        print(f"\n✗ Failed: {config_name} | {task_name}")
        print(f"  Error: {e}")
        return False

    finally:
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass


async def run_experiments(
    experiments: list[
        tuple[str, str, str, str]
    ],  # (config_name, scaffold_name, model_name, task)
    batch_id: str,
    wandb_project: str,
    push_to_origin: bool,
    dry_run: bool = False,
    continue_on_error: bool = False,
    max_turns: int | None = None,
    git_ref: str = "main",
    skip_initial_eval: bool = False,
) -> tuple[int, int, int]:
    """Run all experiments. Returns (succeeded, failed, skipped) counts."""
    succeeded = 0
    failed = 0
    skipped = 0

    print("=" * 60)
    print("Vero Benchmarking Experiments")
    print("=" * 60)
    print(f"Batch ID: {batch_id or '(none)'}")
    print(f"Experiments: {len(experiments)}")
    print(f"Dry run: {dry_run}")
    print("=" * 60)

    for i, (config_name, scaffold_name, model_name, task) in enumerate(experiments, 1):
        if is_already_completed(batch_id, config_name, task):
            print(f"\n[SKIP] {config_name} | {task} (already completed)")
            skipped += 1
            continue

        print(f"\n[{i}/{len(experiments)}] {config_name} | {task}")

        success = await run_single_experiment(
            config_name=config_name,
            scaffold_name=scaffold_name,
            model_name=model_name,
            task_name=task,
            batch_id=batch_id,
            wandb_project=wandb_project,
            push_to_origin=push_to_origin,
            dry_run=dry_run,
            max_turns=max_turns,
            git_ref=git_ref,
            skip_initial_eval=skip_initial_eval,
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
        description="Run Vero benchmarking experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available scaffolds: {", ".join(SCAFFOLDS.keys())}
Available models: {", ".join(MODELS.keys())}
Available tasks: {", ".join(BENCHMARK_TASKS)}

Default configs:
{chr(10).join(f"  - {name}" for name in DEFAULT_CONFIGS.keys())}
""",
    )

    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--config", type=str, choices=list(DEFAULT_CONFIGS.keys())
    )
    config_group.add_argument("--scaffold", type=str, choices=list(SCAFFOLDS.keys()))
    config_group.add_argument("--all-configs", action="store_true")

    parser.add_argument("--model", type=str, choices=list(MODELS.keys()))
    parser.add_argument(
        "--task", type=str, help=f"Task to run. Available: {', '.join(BENCHMARK_TASKS)}"
    )
    parser.add_argument("--batch-id", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument(
        "--sgp-account-id", type=str, default=os.environ.get("SGP_ACCOUNT_ID", "")
    )
    parser.add_argument("--push-to-origin", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--max-turns", type=int, default=None, help="Override max agent turns")
    parser.add_argument("--git-ref", type=str, default="main", help="Git ref to isolate from (branch or commit)")
    parser.add_argument("--skip-initial-eval", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("-n", "--iterations", type=int, default=1)
    parser.add_argument("--skip-configs", type=str, nargs="+", default=[])

    return parser.parse_args()


def list_configs():
    print("=" * 60)
    print("Scaffolds")
    print("=" * 60)
    for name in SCAFFOLDS:
        print(f"  {name}")

    print("\n" + "=" * 60)
    print("Models")
    print("=" * 60)
    for short, full in MODELS.items():
        print(f"  {short}: {full}")

    print("\n" + "=" * 60)
    print("Default Configs")
    print("=" * 60)
    for name, (scaffold, model) in DEFAULT_CONFIGS.items():
        print(f"  {name} ({scaffold} + {model})")

    print("\n" + "=" * 60)
    print("Tasks")
    print("=" * 60)
    for task in BENCHMARK_TASKS:
        print(f"  {task}")


def main():
    args = parse_args()

    if args.list:
        list_configs()
        return 0

    # Setup SGP tracing if configured
    if args.sgp_account_id:
        from vero.logging import setup_sgp_agents_sdk_tracing

        setup_sgp_agents_sdk_tracing(account_id=args.sgp_account_id)

    if not args.task:
        print("Error: --task is required")
        return 1

    # Build list of experiments: (config_name, scaffold_name, model_name, task)
    base_experiments: list[tuple[str, str, str, str]] = []

    if args.config:
        scaffold_name, model_name = DEFAULT_CONFIGS[args.config]
        base_experiments.append((args.config, scaffold_name, model_name, args.task))

    elif args.scaffold:
        if not args.model:
            print("Error: --model is required with --scaffold")
            return 1
        config_name = f"{args.scaffold}-{args.model}"
        base_experiments.append((config_name, args.scaffold, args.model, args.task))

    elif args.all_configs:
        for name, (scaffold_name, model_name) in DEFAULT_CONFIGS.items():
            if name in args.skip_configs:
                continue
            base_experiments.append((name, scaffold_name, model_name, args.task))

    else:
        default_name = "vero-orchestrator-cookbook-sonnet"
        scaffold_name, model_name = DEFAULT_CONFIGS[default_name]
        base_experiments.append((default_name, scaffold_name, model_name, args.task))

    # Expand for iterations
    experiments: list[tuple[str, str, str, str]] = []
    for name, scaffold, model, task in base_experiments:
        for i in range(args.iterations):
            iter_name = f"{name}-iter{i + 1}" if args.iterations > 1 else name
            experiments.append((iter_name, scaffold, model, task))

    if not experiments:
        print("No experiments to run")
        return 0

    succeeded, failed, skipped = asyncio.run(
        run_experiments(
            experiments=experiments,
            batch_id=args.batch_id,
            wandb_project=args.wandb_project,
            push_to_origin=args.push_to_origin,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
            max_turns=args.max_turns,
            git_ref=args.git_ref,
            skip_initial_eval=args.skip_initial_eval,
        )
    )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
