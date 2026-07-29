#!/usr/bin/env python3
"""Create and verify the deterministic dabstep split.

Why this is not `scripts/partition_dataset.py`: that script derives each task's
canonical name from a `[task] name` key in the task's own `task.toml`, and
dabstep's tasks do not carry one. The registry is the only source of canonical
names here (`adyen/<id>`), while the difficulty tag used for stratification lives
in the local export (`dabstep-<id>/task.toml`). This script joins the two on the
numeric id.

It also applies a difficulty quota, which the shared script has no notion of.
dabstep is 72 easy and 378 hard; a proportional sample of the whole dataset would
put the seed near the floor (o4-mini scores 76.4% on easy and 14.55% on hard, and
the next best model on hard is 13.76%). Taking every easy task plus a stratified
hard sample places the seed mid-range instead, and records the mix as a parameter
rather than leaving it implicit.

    python scripts/partition_dabstep.py --tasks-dir <export> --fetch-registry
    python scripts/partition_dabstep.py --tasks-dir <export> --check
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

DATASET_NAME = "adyen/dabstep"
DATASET_DIGEST = "0edf62c0bdf7003b1d1f934f1547df1c051877e076d5b6f6a2d99caf8b6432b3"
TASK_SOURCE = f"{DATASET_NAME}@sha256:{DATASET_DIGEST}"
SEED = "vero-dabstep-v1"

# Every easy task the dataset has, plus a hard sample to reach 165. 165 matches
# browsecomp-plus, which keeps the finalize wall at 9 waves
# (ceil(66 x 3 / 24)) instead of the 23 a proportional 20/40/40 over all 450
# would cost.
QUOTAS = {"easy": 72, "hard": 93}
PARTITIONS = ("development", "validation", "test")
RATIOS = {
    "development": Fraction(1, 5),
    "validation": Fraction(2, 5),
    "test": Fraction(2, 5),
}
TARGET_COUNTS = {"development": 33, "validation": 66, "test": 66}


def _stable_key(task_name: str) -> str:
    return hashlib.sha256(f"{SEED}:{task_name}".encode()).hexdigest()


def _read_difficulties(tasks_dir: Path) -> dict[str, str]:
    """Map the numeric task id to its declared difficulty."""
    root = tasks_dir.expanduser().resolve()
    if not list(root.glob("dabstep-*/task.toml")):
        root = root / "dabstep"
    directories = sorted(root.glob("dabstep-*/task.toml"))
    if len(directories) != 450:
        raise ValueError(f"{root} holds {len(directories)} tasks, expected 450")
    difficulties: dict[str, str] = {}
    for path in directories:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
        difficulty = config.get("metadata", {}).get("difficulty")
        if difficulty not in QUOTAS:
            raise ValueError(f"{path.parent.name} has difficulty {difficulty!r}")
        difficulties[path.parent.name.removeprefix("dabstep-")] = str(difficulty)
    return difficulties


async def _fetch_registry_refs() -> tuple[str, dict[str, str]]:
    from harbor.registry.client.package import PackageDatasetClient

    metadata = await PackageDatasetClient().get_dataset_metadata(TASK_SOURCE)
    refs = {task.get_name(): str(task.ref) for task in metadata.task_ids}
    if len(refs) != 450:
        raise ValueError(f"registry returned {len(refs)} tasks, expected 450")
    return str(metadata.version), refs


def _existing_refs(output_dir: Path) -> tuple[str, dict[str, str]]:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest["dataset_version"], {t["name"]: t["ref"] for t in manifest["tasks"]}


def _select(names: list[str], difficulties: dict[str, str]) -> list[dict[str, str]]:
    """Apply the difficulty quota deterministically."""
    by_difficulty: dict[str, list[str]] = defaultdict(list)
    for name in names:
        task_id = name.split("/", 1)[1]
        if task_id not in difficulties:
            raise ValueError(f"{name} has no local task directory")
        by_difficulty[difficulties[task_id]].append(name)

    selected: list[dict[str, str]] = []
    for difficulty, quota in QUOTAS.items():
        members = sorted(by_difficulty[difficulty], key=_stable_key)
        if len(members) < quota:
            raise ValueError(
                f"quota {quota} exceeds the {len(members)} {difficulty} tasks available"
            )
        selected.extend(
            {"name": name, "stratum": difficulty} for name in members[:quota]
        )
    return selected


def _allocate(tasks: list[dict[str, str]]) -> dict[str, list[str]]:
    """Split each stratum 20/40/40, then settle remainders by largest deficit."""
    strata: dict[str, list[dict[str, str]]] = defaultdict(list)
    for task in tasks:
        strata[task["stratum"]].append(task)

    allocation: dict[str, dict[str, int]] = {}
    remaining: dict[str, int] = {}
    totals: Counter[str] = Counter()
    for stratum, members in strata.items():
        row = {p: int(len(members) * RATIOS[p]) for p in PARTITIONS}
        allocation[stratum] = row
        totals.update(row)
        remaining[stratum] = len(members) - sum(row.values())

    deficits = {p: TARGET_COUNTS[p] - totals[p] for p in PARTITIONS}
    while sum(remaining.values()):
        choices = [
            (len(strata[stratum]) * RATIOS[p] - allocation[stratum][p], stratum, p)
            for stratum, left in remaining.items()
            if left
            for p in PARTITIONS
            if deficits[p]
        ]
        if not choices:
            raise RuntimeError("could not satisfy the exact partition sizes")
        _, stratum, partition = max(
            choices, key=lambda item: (item[0], -PARTITIONS.index(item[2]), item[1])
        )
        allocation[stratum][partition] += 1
        remaining[stratum] -= 1
        deficits[partition] -= 1

    result: dict[str, list[str]] = {p: [] for p in PARTITIONS}
    for stratum, members in strata.items():
        ordered = sorted(members, key=lambda task: _stable_key(task["name"]))
        cursor = 0
        for partition in PARTITIONS:
            count = allocation[stratum][partition]
            result[partition].extend(task["name"] for task in ordered[cursor : cursor + count])
            cursor += count
    return {p: sorted(names) for p, names in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent.parent / "partitions")
    parser.add_argument("--fetch-registry", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    difficulties = _read_difficulties(args.tasks_dir)
    if args.fetch_registry:
        version, refs = asyncio.run(_fetch_registry_refs())
    else:
        version, refs = _existing_refs(args.output_dir)

    selected = _select(sorted(refs), difficulties)
    partitions = _allocate(selected)
    strata = {task["name"]: task["stratum"] for task in selected}

    for partition, names in partitions.items():
        if len(names) != TARGET_COUNTS[partition]:
            raise RuntimeError(f"{partition} holds {len(names)}, expected {TARGET_COUNTS[partition]}")
    flattened = [name for names in partitions.values() for name in names]
    if len(set(flattened)) != len(flattened):
        raise RuntimeError("a task appears in more than one partition")

    manifest = {
        "schema_version": 1,
        "task_source": TASK_SOURCE,
        "dataset_name": DATASET_NAME,
        "dataset_version": version,
        "seed": SEED,
        "ratios": {p: float(RATIOS[p]) for p in PARTITIONS},
        "difficulty_quotas": QUOTAS,
        "partition_counts": TARGET_COUNTS,
        "difficulty_counts": {
            partition: dict(Counter(strata[name] for name in names))
            for partition, names in partitions.items()
        },
        "tasks": [
            {"name": name, "ref": refs[name], "partition": partition, "difficulty": strata[name]}
            for partition in PARTITIONS
            for name in partitions[partition]
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    documents = {"manifest.json": manifest} | {
        f"{partition}.json": partitions[partition] for partition in PARTITIONS
    }
    for filename, document in documents.items():
        rendered = json.dumps(document, indent=2, sort_keys=False) + "\n"
        path = args.output_dir / filename
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
                raise SystemExit(f"{path} does not match the regenerated split")
        else:
            path.write_text(rendered, encoding="utf-8")

    print(("verified" if args.check else "wrote") + f" {args.output_dir}")
    for partition in PARTITIONS:
        print(f"  {partition:<12}{TARGET_COUNTS[partition]:>4}  {manifest['difficulty_counts'][partition]}")


if __name__ == "__main__":
    main()
