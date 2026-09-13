#!/usr/bin/env python3
"""Create and verify the committed SWE-bench-Pro development/validation/test split.

The task source is the ``swebenchpro`` dataset in the default Harbor registry
(731 instances, version 1.0). Unlike swe-atlas-qna, it is a *registry* dataset
rather than an ``<org>/<name>@sha256:<digest>`` package: its tasks resolve to
``GitTaskId``s under ``laude-institute/harbor-datasets`` at a pinned commit, so
the version pin here is ``swebenchpro@1.0`` and the recorded ref is the pinned
git commit plus the task's path within that repository.

The deterministic, sha256-keyed stratified split and the ``--check`` mode mirror
the GAIA script exactly, so the committed split is reproducible and verifiable.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

DATASET_NAME = "swebenchpro"
DATASET_VERSION = "1.0"
TASK_SOURCE = f"{DATASET_NAME}@{DATASET_VERSION}"
SEED = "vero-swe-bench-pro-v1"
PARTITIONS = ("development", "validation", "test")
RATIOS = {
    "development": Fraction(1, 5),
    "validation": Fraction(2, 5),
    "test": Fraction(2, 5),
}

# 731 instances split 20/40/40. The exact ratios are 146.2/292.4/292.4, so the
# leftover case goes by largest remainder; the .4/.4 tie between validation and
# test breaks towards test, matching how the sibling benchmarks resolved the
# same tie (officeqa 246 -> 49/98/99, swe-atlas-qna 124 -> 25/49/50).
TOTAL_TASKS = 731
TARGET_COUNTS = {"development": 146, "validation": 292, "test": 293}

# `--sample` writes a second, smaller split into partitions/sample/ for the
# optimizer variant (baseline/build.sample.yaml). The canonical split above is
# never touched by it. 33/66/66 matches gaia and browsecomp-plus, the suite's
# tightest split, which is the right reference for its most expensive benchmark:
# a case here builds a real repository and runs its test suite, so the full-split
# held-out baseline alone cost ~7.8h of wall clock for three rounds.
SAMPLE_COUNTS = {"development": 33, "validation": 66, "test": 66}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        help=(
            "Directory containing the exported SWE-bench-Pro task directories. "
            "Omit to re-derive the split from the committed manifest, which "
            "already records every instance's name and stratum."
        ),
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help=(
            "Write the smaller SAMPLE_COUNTS split to partitions/sample/ for the "
            "optimizer variant. Leaves the canonical split untouched."
        ),
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

    SWE-bench-Pro's ``task.toml`` carries no ``[task].name``: Harbor derives a
    task's canonical name from its directory (``instance_<owner>__<repo>-<sha>``),
    which is what ``-i`` matches against. The stratification key is the upstream
    project, recorded verbatim as ``repo`` in each task's ``tests/config.json``.
    """
    tasks: list[dict[str, str]] = []
    for task_dir in sorted(_task_root(path).iterdir()):
        if not task_dir.is_dir():
            continue
        config = json.loads(
            (task_dir / "tests" / "config.json").read_text(encoding="utf-8")
        )
        repository = config.get("repo")
        if not repository:
            raise ValueError(f"{task_dir.name} has no tests/config.json repo")
        tasks.append({"name": task_dir.name, "repository": repository})
    if len(tasks) != TOTAL_TASKS:
        raise ValueError(f"expected {TOTAL_TASKS} tasks, found {len(tasks)}")
    return tasks


def _tasks_from_manifest(output_dir: Path) -> list[dict[str, str]]:
    """Re-read the (name, repository) pairs the committed manifest already pins.

    Deriving the sample should not require re-exporting 731 task directories:
    the manifest records every instance and its stratum verbatim, which is the
    entire input `_allocate` needs. `--tasks-dir` remains the path for re-pinning
    against a moved dataset.
    """
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("no existing manifest; pass --tasks-dir")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = [
        {"name": item["name"], "repository": item["repository"]}
        for item in manifest["tasks"]
    ]
    if len(tasks) != TOTAL_TASKS:
        raise ValueError(f"expected {TOTAL_TASKS} tasks, manifest has {len(tasks)}")
    return tasks


async def _fetch_registry_refs() -> tuple[str, dict[str, str]]:
    try:
        from harbor.registry.client.factory import RegistryClientFactory
    except ImportError as error:
        raise RuntimeError(
            "--fetch-registry requires the exactly pinned Harbor package"
        ) from error

    metadata = await RegistryClientFactory.create().get_dataset_metadata(TASK_SOURCE)
    # Registry tasks are git-backed: pin the commit and the in-repo path, which
    # together identify the exact task content the way a package digest would.
    refs = {
        task.get_name(): f"{task.git_commit_id}:{task.path.as_posix()}"
        for task in metadata.task_ids
    }
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


def _subsample(
    full: dict[str, list[str]], tasks: list[dict[str, str]]
) -> dict[str, list[str]]:
    """Shrink each partition to SAMPLE_COUNTS without letting tasks cross partitions.

    Sampling happens strictly inside a partition, so every sampled set is a
    subset of the corresponding full set. Two properties follow, and both are
    why it is done this way rather than by re-running the split at a smaller
    size: an instance can never move from test into development (a flat
    re-split does move them, which would silently destroy the held-out
    guarantee), and a reward already measured over the full partition can be
    recomputed over the sample from the existing per-case trials.

    Within a partition the same repository stratification and sha256 ordering
    apply, so the sample keeps the partition's repo mix and stays reproducible.
    """
    repository = {task["name"]: task["repository"] for task in tasks}
    sampled: dict[str, list[str]] = {}
    for partition in PARTITIONS:
        target = SAMPLE_COUNTS[partition]
        members = full[partition]
        if target > len(members):
            raise ValueError(
                f"{partition} sample of {target} exceeds its {len(members)} tasks"
            )
        strata: dict[str, list[str]] = defaultdict(list)
        for name in members:
            strata[repository[name]].append(name)

        quota = {
            stratum: target * len(names) // len(members)
            for stratum, names in strata.items()
        }
        remainders = sorted(
            strata,
            key=lambda stratum: (
                -(target * len(strata[stratum]) % len(members)),
                stratum,
            ),
        )
        cursor = 0
        while sum(quota.values()) < target:
            stratum = remainders[cursor % len(remainders)]
            if quota[stratum] < len(strata[stratum]):
                quota[stratum] += 1
            cursor += 1

        chosen: list[str] = []
        for stratum in sorted(strata):
            chosen.extend(sorted(strata[stratum], key=_stable_key)[: quota[stratum]])
        chosen.sort()
        if len(chosen) != target:
            raise RuntimeError(f"wrong sampled {partition} size: {len(chosen)}")
        sampled[partition] = chosen
    return sampled


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


def _render_sample(partitions: dict[str, list[str]]) -> dict[str, str]:
    """Just the three partition files. The manifest is shared with the full split.

    Nothing benchmark-level changes for the sample: the same manifest pins the
    same 731 refs, and the build loader reads only task names out of it. Writing
    a second manifest would duplicate that pinning and let the two drift.
    """
    return {
        f"{partition}.json": json.dumps(partitions[partition], indent=2) + "\n"
        for partition in PARTITIONS
    }


def main() -> None:
    args = _parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    tasks = (
        _read_tasks(args.tasks_dir)
        if args.tasks_dir is not None
        else _tasks_from_manifest(args.output_dir)
    )
    if args.sample:
        # Derived from the canonical split, then narrowed inside each partition.
        rendered = _render_sample(_subsample(_allocate(tasks), tasks))
        args.output_dir = args.output_dir / "sample"
    else:
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
