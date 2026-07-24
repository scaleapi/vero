#!/usr/bin/env python3
"""Create and verify the committed SWE-bench-Pro development/validation/test split.

STUB STATUS
-----------
This script is a faithful adaptation of ``gaia/scripts/partition_gaia.py`` for
SWE-bench-Pro, but it cannot produce real partitions until the SWE-bench-Pro
Harbor task source is pinned. Two constants and one reader must be confirmed
against the real task package before running without ``--check``:

  1. ``DATASET_DIGEST`` — the real content digest (see baseline/build.yaml).
  2. ``TOTAL_TASKS`` / ``TARGET_COUNTS`` — the real task count and split sizes.
  3. ``_read_tasks`` — how each task exposes its canonical name and the field
     used to stratify (assumed here to be the source repository). Adjust the
     TOML keys once the pinned task.toml layout is known.

The deterministic, sha256-keyed stratified split and the ``--check`` mode mirror
the GAIA script exactly, so once the constants are correct the committed split is
reproducible and verifiable.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tomllib
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

DATASET_NAME = "scale-ai/swe-bench-pro"
# TODO: replace with the real content digest once the task source is pinned.
DATASET_DIGEST = "REPLACE_WITH_REAL_DIGEST"
TASK_SOURCE = f"{DATASET_NAME}@sha256:{DATASET_DIGEST}"
SEED = "vero-swe-bench-pro-v1"
PARTITIONS = ("development", "validation", "test")
RATIOS = {
    "development": Fraction(1, 5),
    "validation": Fraction(2, 5),
    "test": Fraction(2, 5),
}

# TODO: set to the real task count and split sizes (dev/val/test) once the task
# source is pinned. These placeholders keep the arithmetic self-consistent
# (20/40/40 of TOTAL_TASKS) but are NOT the real dataset size.
TOTAL_TASKS = 0
TARGET_COUNTS = {"development": 0, "validation": 0, "test": 0}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="Directory containing the exported SWE-bench-Pro task directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[1] / "partitions",
    )
    parser.add_argument("--fetch-registry", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def _task_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if len(list(path.glob("*/task.toml"))) == TOTAL_TASKS:
        return path
    nested = path / "swe-bench-pro"
    if len(list(nested.glob("*/task.toml"))) == TOTAL_TASKS:
        return nested
    raise ValueError(
        f"{path} does not contain exactly {TOTAL_TASKS} exported "
        "SWE-bench-Pro tasks (update TOTAL_TASKS once the source is pinned)"
    )


def _read_tasks(path: Path) -> list[dict[str, str]]:
    """Read the exported tasks and the field used to stratify.

    STUB: the exact task.toml layout for SWE-bench-Pro is not yet known. This
    reader assumes each task exposes a canonical ``[task].name`` and a source
    repository under ``[metadata].repository`` (or a ``repo:<name>`` tag). Adjust
    the key lookups below to match the real package before generating partitions.
    """
    tasks: list[dict[str, str]] = []
    for task_dir in sorted(_task_root(path).iterdir()):
        if not task_dir.is_dir():
            continue
        config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        canonical_name = config.get("task", {}).get("name")
        if not canonical_name:
            raise ValueError(f"{task_dir.name} has no [task].name")
        metadata = config.get("metadata", {})
        repository = metadata.get("repository")
        if not repository:
            tags = metadata.get("tags", [])
            repo_tags = [t for t in tags if t.startswith("repo:")]
            repository = repo_tags[0].removeprefix("repo:") if repo_tags else "unknown"
        tasks.append({"name": canonical_name, "repository": repository})
    if len(tasks) != TOTAL_TASKS:
        raise ValueError(f"expected {TOTAL_TASKS} tasks, found {len(tasks)}")
    return tasks


async def _fetch_registry_refs() -> tuple[str, dict[str, str]]:
    try:
        from harbor.registry.client.package import PackageDatasetClient
    except ImportError as error:
        raise RuntimeError(
            "--fetch-registry requires the exactly pinned Harbor package"
        ) from error

    metadata = await PackageDatasetClient().get_dataset_metadata(TASK_SOURCE)
    refs = {task.get_name(): str(task.ref) for task in metadata.task_ids}
    return str(metadata.version), refs


def _existing_refs(output_dir: Path) -> tuple[str, dict[str, str]]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("no existing manifest; rerun with --fetch-registry")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    refs = {item["name"]: item["ref"] for item in manifest["tasks"]}
    return manifest["dataset_version"], refs


def _stable_key(task_name: str) -> str:
    return hashlib.sha256(f"{SEED}:{task_name}".encode()).hexdigest()


def _allocate(tasks: list[dict[str, str]]) -> dict[str, list[str]]:
    strata: dict[str, list[dict[str, str]]] = defaultdict(list)
    for task in tasks:
        strata[task["repository"]].append(task)

    allocation: dict[str, dict[str, int]] = {}
    remaining_by_stratum: dict[str, int] = {}
    totals: Counter[str] = Counter()
    for stratum, members in strata.items():
        row = {
            partition: int(len(members) * RATIOS[partition]) for partition in PARTITIONS
        }
        allocation[stratum] = row
        totals.update(row)
        remaining_by_stratum[stratum] = len(members) - sum(row.values())

    deficits = {
        partition: TARGET_COUNTS[partition] - totals[partition]
        for partition in PARTITIONS
    }
    while sum(remaining_by_stratum.values()):
        choices: list[tuple[Fraction, str, str]] = []
        for stratum, remaining in remaining_by_stratum.items():
            if remaining == 0:
                continue
            size = len(strata[stratum])
            for partition in PARTITIONS:
                if deficits[partition] == 0:
                    continue
                ideal = size * RATIOS[partition]
                shortfall = ideal - allocation[stratum][partition]
                choices.append((shortfall, stratum, partition))
        if not choices:
            raise RuntimeError("could not satisfy exact partition sizes")
        _, stratum, partition = max(
            choices,
            key=lambda item: (item[0], -PARTITIONS.index(item[2]), item[1]),
        )
        allocation[stratum][partition] += 1
        remaining_by_stratum[stratum] -= 1
        deficits[partition] -= 1

    result: dict[str, list[str]] = {partition: [] for partition in PARTITIONS}
    for stratum in sorted(strata):
        members = sorted(strata[stratum], key=lambda task: _stable_key(task["name"]))
        cursor = 0
        for partition in PARTITIONS:
            count = allocation[stratum][partition]
            result[partition].extend(
                task["name"] for task in members[cursor : cursor + count]
            )
            cursor += count
    for partition in PARTITIONS:
        result[partition].sort()
        if len(result[partition]) != TARGET_COUNTS[partition]:
            raise RuntimeError(f"wrong {partition} size: {len(result[partition])}")
    return result


def _render(
    tasks: list[dict[str, str]],
    partitions: dict[str, list[str]],
    dataset_version: str,
    refs: dict[str, str],
) -> dict[str, str]:
    names = {task["name"] for task in tasks}
    if names != set(refs):
        raise ValueError(
            "downloaded task names do not match the pinned registry dataset"
        )
    membership = {
        name: partition
        for partition, partition_tasks in partitions.items()
        for name in partition_tasks
    }
    partition_digest = hashlib.sha256(
        json.dumps(partitions, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    stratum_counts = Counter(task["repository"] for task in tasks)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task_source": TASK_SOURCE,
        "dataset_name": DATASET_NAME,
        "dataset_version": dataset_version,
        "seed": SEED,
        "ratios": {partition: float(RATIOS[partition]) for partition in PARTITIONS},
        "stratified_by": ["repository"],
        "partition_counts": TARGET_COUNTS,
        "partition_digest": f"sha256:{partition_digest}",
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "tasks": [
            {
                **task,
                "ref": refs[task["name"]],
                "partition": membership[task["name"]],
            }
            for task in sorted(tasks, key=lambda item: item["name"])
        ],
    }
    rendered = {
        f"{partition}.json": json.dumps(partitions[partition], indent=2) + "\n"
        for partition in PARTITIONS
    }
    rendered["manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    return rendered


def main() -> None:
    if DATASET_DIGEST == "REPLACE_WITH_REAL_DIGEST" or TOTAL_TASKS == 0:
        raise SystemExit(
            "STUB: pin the real DATASET_DIGEST and TOTAL_TASKS/TARGET_COUNTS "
            "(and confirm _read_tasks against the real task.toml) before running. "
            "See the module docstring and ../../README.md."
        )
    args = _parse_args()
    tasks = _read_tasks(args.tasks_dir)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.fetch_registry:
        dataset_version, refs = asyncio.run(_fetch_registry_refs())
    else:
        dataset_version, refs = _existing_refs(args.output_dir)
    rendered = _render(tasks, _allocate(tasks), dataset_version, refs)

    if args.check:
        changed = [
            filename
            for filename, content in rendered.items()
            if not (args.output_dir / filename).is_file()
            or (args.output_dir / filename).read_text(encoding="utf-8") != content
        ]
        if changed:
            raise SystemExit("partition files are stale: " + ", ".join(changed))
        print("SWE-bench-Pro partitions match the pinned dataset and split algorithm")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in rendered.items():
        (args.output_dir / filename).write_text(content, encoding="utf-8")
    print(f"wrote {len(rendered)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
