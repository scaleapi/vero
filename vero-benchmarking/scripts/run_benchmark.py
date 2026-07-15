#!/usr/bin/env python3
"""Run repeatable batches of canonical VeRO benchmark sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vero_benchmarking.runner import (  # noqa: E402
    MODELS,
    build_benchmark_session,
    run_optimization,
)
from vero_benchmarking.tasks import BENCHMARK_TASKS  # noqa: E402


SCAFFOLDS: dict[str, dict[str, Any]] = {
    "vero-default": {"agent_name": "vero"},
    "claude-code": {"agent_name": "claude"},
}

DEFAULT_CONFIGS: dict[str, tuple[str, str]] = {
    "vero-sonnet": ("vero-default", "sonnet"),
    "vero-opus": ("vero-default", "opus"),
    "vero-gpt": ("vero-default", "gpt"),
    "claude-code-sonnet": ("claude-code", "sonnet"),
}


def get_manifest_path(batch_id: str) -> Path:
    from vero_benchmarking.constants import DEFAULT_LOG_DIR

    path = DEFAULT_LOG_DIR / "batch_manifests" / f"{batch_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def is_already_completed(batch_id: str, config_name: str, task: str) -> bool:
    if not batch_id:
        return False
    path = get_manifest_path(batch_id)
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("config_name") == config_name and entry.get("task") == task:
            return True
    return False


def write_to_manifest(
    batch_id: str,
    config_name: str,
    task: str,
    session_id: str,
    best_commit: str | None,
) -> None:
    if not batch_id:
        return
    import fcntl

    entry = {
        "config_name": config_name,
        "task": task,
        "session_id": session_id,
        "best_commit": best_commit,
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    with get_manifest_path(batch_id).open("a", encoding="utf-8") as manifest:
        fcntl.flock(manifest.fileno(), fcntl.LOCK_EX)
        try:
            manifest.write(json.dumps(entry) + "\n")
        finally:
            fcntl.flock(manifest.fileno(), fcntl.LOCK_UN)


async def run_single_experiment(
    *,
    config_name: str,
    scaffold_name: str,
    model_name: str,
    task_name: str,
    batch_id: str,
    dry_run: bool = False,
    max_turns: int = 200,
    max_candidates: int | None = None,
) -> bool:
    scaffold = SCAFFOLDS[scaffold_name]
    print(f"\nRunning: {config_name} | {task_name}")
    if dry_run:
        print(f"  Agent: {scaffold['agent_name']}")
        print(f"  Model: {MODELS[model_name]}")
        return True

    try:
        session = await build_benchmark_session(
            task_name=task_name,
            model=MODELS[model_name],
            agent_name=scaffold["agent_name"],
            max_turns=max_turns,
            max_candidates=max_candidates,
            metadata={"batch_id": batch_id, "config_name": config_name},
        )
        result = await run_optimization(
            session,
            batch_id=batch_id or None,
            config_name=config_name,
        )
    except Exception as error:
        print(f"Failed: {config_name} | {task_name}: {error}")
        return False

    print(f"Completed: {config_name} | {task_name}")
    print(f"  Session ID: {result.session_id}")
    print(f"  Best commit: {result.best_commit}")
    write_to_manifest(
        batch_id,
        config_name,
        task_name,
        result.session_id,
        result.best_commit,
    )
    return True


async def run_experiments(
    experiments: list[tuple[str, str, str, str]],
    *,
    batch_id: str,
    dry_run: bool,
    continue_on_error: bool,
    max_turns: int,
    max_candidates: int | None,
) -> tuple[int, int, int]:
    succeeded = failed = skipped = 0
    for index, (config, scaffold, model, task) in enumerate(experiments, 1):
        if is_already_completed(batch_id, config, task):
            print(f"[{index}/{len(experiments)}] Skipped: {config} | {task}")
            skipped += 1
            continue
        success = await run_single_experiment(
            config_name=config,
            scaffold_name=scaffold,
            model_name=model,
            task_name=task,
            batch_id=batch_id,
            dry_run=dry_run,
            max_turns=max_turns,
            max_candidates=max_candidates,
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--config", choices=sorted(DEFAULT_CONFIGS))
    group.add_argument("--scaffold", choices=sorted(SCAFFOLDS))
    group.add_argument("--all-configs", action="store_true")
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--task", choices=BENCHMARK_TASKS)
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("-n", "--iterations", type=int, default=1)
    parser.add_argument("--skip-configs", nargs="+", default=[])
    return parser.parse_args()


def list_configs() -> None:
    print("Scaffolds:", ", ".join(sorted(SCAFFOLDS)))
    print("Models:", ", ".join(sorted(MODELS)))
    print("Configs:", ", ".join(sorted(DEFAULT_CONFIGS)))
    print("Tasks:", ", ".join(BENCHMARK_TASKS))


def main() -> int:
    arguments = parse_args()
    if arguments.list:
        list_configs()
        return 0
    if arguments.task is None:
        print("Error: --task is required")
        return 1

    base: list[tuple[str, str, str, str]] = []
    if arguments.config:
        scaffold, model = DEFAULT_CONFIGS[arguments.config]
        base.append((arguments.config, scaffold, model, arguments.task))
    elif arguments.scaffold:
        if arguments.model is None:
            print("Error: --model is required with --scaffold")
            return 1
        base.append(
            (
                f"{arguments.scaffold}-{arguments.model}",
                arguments.scaffold,
                arguments.model,
                arguments.task,
            )
        )
    elif arguments.all_configs:
        base.extend(
            (name, scaffold, model, arguments.task)
            for name, (scaffold, model) in DEFAULT_CONFIGS.items()
            if name not in arguments.skip_configs
        )
    else:
        scaffold, model = DEFAULT_CONFIGS["vero-sonnet"]
        base.append(("vero-sonnet", scaffold, model, arguments.task))

    experiments = [
        (
            f"{name}-iter{iteration + 1}" if arguments.iterations > 1 else name,
            scaffold,
            model,
            task,
        )
        for name, scaffold, model, task in base
        for iteration in range(arguments.iterations)
    ]
    _, failed, _ = asyncio.run(
        run_experiments(
            experiments,
            batch_id=arguments.batch_id,
            dry_run=arguments.dry_run,
            continue_on_error=arguments.continue_on_error,
            max_turns=arguments.max_turns,
            max_candidates=arguments.max_candidates,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
