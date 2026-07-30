#!/usr/bin/env python3
"""Create and verify deterministic splits for Harbor-native benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

PARTITIONS = ("development", "validation", "test")
RATIOS = {
    "development": Fraction(1, 5),
    "validation": Fraction(2, 5),
    "test": Fraction(2, 5),
}


@dataclass(frozen=True)
class BenchmarkSpec:
    dataset_name: str
    dataset_digest: str
    seed: str
    task_count: int
    target_counts: dict[str, int]
    export_dir_name: str
    stratified_by: tuple[str, ...]
    #: Registry version to pin instead of a content digest. Some hub datasets are
    #: addressed by an incrementing version rather than a sha256, and the loader
    #: only requires the reference to be explicit -- not that it be a digest.
    #: Prefer a digest when one is published: a version number is a mutable
    #: pointer in principle, so record which form was used.
    dataset_version_ref: str | None = None

    @property
    def task_source(self) -> str:
        if self.dataset_version_ref is not None:
            return f"{self.dataset_name}@{self.dataset_version_ref}"
        return f"{self.dataset_name}@sha256:{self.dataset_digest}"


SPECS = {
    "swe-atlas-qna": BenchmarkSpec(
        dataset_name="scale-ai/swe-atlas-qna",
        dataset_digest=(
            "0e26bc0313ae2fc6f912b67b928e648c7f20d17d91f765f702a93042ce5be0e4"
        ),
        seed="vero-swe-atlas-qna-v1",
        task_count=124,
        target_counts={"development": 25, "validation": 49, "test": 50},
        export_dir_name="swe-atlas-qna",
        stratified_by=("repository",),
    ),
    "tau3": BenchmarkSpec(
        dataset_name="sierra-research/tau3-bench",
        dataset_digest=(
            "a57304f682894ac061090769af771a3617664f3ff6e5417d4eadf8e30433e4d9"
        ),
        seed="vero-tau3-v1",
        task_count=375,
        target_counts={"development": 75, "validation": 150, "test": 150},
        export_dir_name="tau3-bench",
        stratified_by=("domain",),
    ),
    # Terminal-Bench 2.1: the whole 89-task set, unfiltered. Filtering to a
    # SWE/ML subset was considered and rejected -- it drops the set to ~50 tasks
    # (10/20/20), and choosing which tasks count as in-domain is a selection knob
    # we would then have to defend. Taking everything removes that argument.
    #
    # Stratified by category AND difficulty. Category alone looks sufficient --
    # 16 categories over 89 tasks, and crossing with difficulty leaves many
    # singleton strata -- but it is not, and the measured spread says so:
    #
    #   stratified by            dev hard   val hard   test hard
    #   category only               29%        28%        42%
    #   category + difficulty       35%        31%        36%   (dataset: 34%)
    #
    # Category alone hands the optimizer an easier search set than the set it is
    # scored on, which biases every measured improvement downward and confounds
    # it with difficulty. The singleton strata are harmless: allocation floors
    # per stratum and then fills to the exact target by largest remainder, so a
    # stratum of one simply lands wherever the remainder ordering puts it.
    "terminal-bench": BenchmarkSpec(
        dataset_name="terminal-bench/terminal-bench-2-1",
        dataset_digest=(
            "7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
        ),
        seed="vero-terminal-bench-2-1-v1",
        task_count=89,
        # 89 does not divide 1:2:2. Validation and test hold the exact 2/5 each
        # and development absorbs the rounding loss, because development is the
        # optimizer's own full-disclosure search set while test is the measurement.
        target_counts={"development": 17, "validation": 36, "test": 36},
        export_dir_name="terminal-bench-2-1",
        stratified_by=("category", "difficulty"),
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=sorted(SPECS))
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="Directory containing an exported copy of the pinned Harbor dataset",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fetch-registry", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def _task_root(path: Path, spec: BenchmarkSpec) -> Path:
    path = path.expanduser().resolve()
    candidates = (path, path / spec.export_dir_name)
    for candidate in candidates:
        if len(list(candidate.glob("*/task.toml"))) == spec.task_count:
            return candidate
    raise ValueError(
        f"{path} does not contain exactly {spec.task_count} exported tasks"
    )


def _task_details(
    benchmark: str, task_dir: Path, config: dict[str, Any]
) -> dict[str, str]:
    metadata = config.get("metadata", {})
    if benchmark == "swe-atlas-qna":
        details = {
            "repository": str(metadata.get("repository") or ""),
            "language": str(metadata.get("language") or ""),
            "category": str(metadata.get("category") or ""),
        }
        if not all(details.values()):
            raise ValueError(f"{task_dir.name} has incomplete SWE-Atlas metadata")
        return details

    if benchmark == "terminal-bench":
        details = {
            "category": str(metadata.get("category") or ""),
            "difficulty": str(metadata.get("difficulty") or ""),
        }
        # Both come from the task's own metadata, so a task that stops declaring
        # them would silently join an empty-string stratum and skew the split.
        # Fail instead.
        missing = [key for key, value in details.items() if not value]
        if missing:
            raise ValueError(
                f"{task_dir.name} is missing task metadata: {', '.join(missing)}"
            )
        return details

    keywords = config.get("task", {}).get("keywords", [])
    domains = {
        keyword
        for keyword in keywords
        if keyword in {"airline", "banking_knowledge", "retail", "telecom"}
    }
    if len(domains) != 1:
        raise ValueError(f"{task_dir.name} does not identify one tau3 domain")
    return {
        "domain": domains.pop(),
        "difficulty": str(metadata.get("difficulty") or "unknown"),
    }


def _read_tasks(
    benchmark: str, path: Path, spec: BenchmarkSpec
) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    for task_dir in sorted(_task_root(path, spec).iterdir()):
        if not task_dir.is_dir():
            continue
        config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        canonical_name = config.get("task", {}).get("name")
        if not isinstance(canonical_name, str) or not canonical_name.startswith(
            spec.dataset_name.split("/", 1)[0] + "/"
        ):
            raise ValueError(
                f"{task_dir.name} has unexpected canonical name {canonical_name!r}"
            )
        details = _task_details(benchmark, task_dir, config)
        stratum = ":".join(details[field] for field in spec.stratified_by)
        tasks.append({"name": canonical_name, "stratum": stratum, **details})
    if len(tasks) != spec.task_count:
        raise ValueError(f"expected {spec.task_count} tasks, found {len(tasks)}")
    if len({task["name"] for task in tasks}) != spec.task_count:
        raise ValueError("dataset contains duplicate canonical task names")
    return tasks


async def _fetch_registry_refs(
    spec: BenchmarkSpec,
) -> tuple[str, dict[str, str]]:
    try:
        from harbor.registry.client.package import PackageDatasetClient
    except ImportError as error:
        raise RuntimeError(
            "--fetch-registry requires the exactly pinned Harbor package"
        ) from error

    metadata = await PackageDatasetClient().get_dataset_metadata(spec.task_source)
    refs = {task.get_name(): str(task.ref) for task in metadata.task_ids}
    return str(metadata.version), refs


def _existing_refs(output_dir: Path) -> tuple[str, dict[str, str]]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("no existing manifest; rerun with --fetch-registry")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    refs = {item["name"]: item["ref"] for item in manifest["tasks"]}
    return manifest["dataset_version"], refs


def _stable_key(seed: str, task_name: str) -> str:
    return hashlib.sha256(f"{seed}:{task_name}".encode()).hexdigest()


def _allocate(tasks: list[dict[str, str]], spec: BenchmarkSpec) -> dict[str, list[str]]:
    strata: dict[str, list[dict[str, str]]] = defaultdict(list)
    for task in tasks:
        strata[task["stratum"]].append(task)

    allocation: dict[str, dict[str, int]] = {}
    remaining_by_stratum: dict[str, int] = {}
    totals = Counter()
    for stratum, members in strata.items():
        row = {
            partition: int(len(members) * RATIOS[partition]) for partition in PARTITIONS
        }
        allocation[stratum] = row
        totals.update(row)
        remaining_by_stratum[stratum] = len(members) - sum(row.values())

    deficits = {
        partition: spec.target_counts[partition] - totals[partition]
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
                choices.append(
                    (ideal - allocation[stratum][partition], stratum, partition)
                )
        if not choices:
            raise RuntimeError("could not satisfy exact partition sizes")
        _, stratum, partition = max(
            choices,
            key=lambda item: (item[0], -PARTITIONS.index(item[2]), item[1]),
        )
        allocation[stratum][partition] += 1
        remaining_by_stratum[stratum] -= 1
        deficits[partition] -= 1

    result = {partition: [] for partition in PARTITIONS}
    for stratum in sorted(strata):
        members = sorted(
            strata[stratum], key=lambda task: _stable_key(spec.seed, task["name"])
        )
        cursor = 0
        for partition in PARTITIONS:
            count = allocation[stratum][partition]
            result[partition].extend(
                task["name"] for task in members[cursor : cursor + count]
            )
            cursor += count
    for partition in PARTITIONS:
        result[partition].sort()
        if len(result[partition]) != spec.target_counts[partition]:
            raise RuntimeError(f"wrong {partition} size: {len(result[partition])}")
    return result


def _render(
    tasks: list[dict[str, str]],
    partitions: dict[str, list[str]],
    dataset_version: str,
    refs: dict[str, str],
    spec: BenchmarkSpec,
) -> dict[str, str]:
    names = {task["name"] for task in tasks}
    if names != set(refs):
        raise ValueError("exported task names do not match the pinned registry dataset")
    membership = {
        name: partition
        for partition, partition_tasks in partitions.items()
        for name in partition_tasks
    }
    partition_digest = hashlib.sha256(
        json.dumps(partitions, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task_source": spec.task_source,
        "dataset_name": spec.dataset_name,
        "dataset_version": dataset_version,
        "seed": spec.seed,
        "ratios": {partition: float(RATIOS[partition]) for partition in PARTITIONS},
        "stratified_by": list(spec.stratified_by),
        "partition_counts": spec.target_counts,
        "partition_digest": f"sha256:{partition_digest}",
        "stratum_counts": dict(
            sorted(Counter(task["stratum"] for task in tasks).items())
        ),
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
    args = _parse_args()
    spec = SPECS[args.benchmark]
    tasks = _read_tasks(args.benchmark, args.tasks_dir, spec)
    output_dir = args.output_dir.expanduser().resolve()
    if args.fetch_registry:
        dataset_version, refs = asyncio.run(_fetch_registry_refs(spec))
    else:
        dataset_version, refs = _existing_refs(output_dir)
    rendered = _render(tasks, _allocate(tasks, spec), dataset_version, refs, spec)

    if args.check:
        changed = [
            filename
            for filename, content in rendered.items()
            if not (output_dir / filename).is_file()
            or (output_dir / filename).read_text(encoding="utf-8") != content
        ]
        if changed:
            raise SystemExit("partition files are stale: " + ", ".join(changed))
        print(f"{args.benchmark} partitions match the pinned dataset")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in rendered.items():
        (output_dir / filename).write_text(content, encoding="utf-8")
    print(f"wrote {len(rendered)} files to {output_dir}")


if __name__ == "__main__":
    main()
