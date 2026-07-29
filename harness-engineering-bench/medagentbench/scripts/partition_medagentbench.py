#!/usr/bin/env python3
"""Create and verify the deterministic medagentbench split.

Unlike the other benchmarks this needs no local export of the dataset: the
stratum is carried by the canonical name itself (`stanford/task<N>_<M>`, where
`task<N>` is one of the ten MedAgentBench task categories), so the registry alone
supplies both the names and the stratification key.

The ten categories divide into retrieval (GET-only) and action (issues a POST),
and the published per-category scores differ sharply between them, so the mix is
the parameter that sets where the seed lands. Taking an equal 15 per category
keeps that mix fixed and balanced rather than letting a uniform draw decide it.

    python scripts/partition_medagentbench.py --fetch-registry
    python scripts/partition_medagentbench.py --check
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

DATASET_NAME = "stanford/medagentbench"
DATASET_DIGEST = "c52d82b3462fb26417707682095e43f224a31f8f785eb7da615c1ab6adc20bf0"
TASK_SOURCE = f"{DATASET_NAME}@sha256:{DATASET_DIGEST}"
SEED = "vero-medagentbench-v1"

# 15 of each category, so every partition holds a balanced 3/6/6 per category.
# 150 cases holds the finalize wall at 8 waves (ceil(60 x 3 / 24)); the full 300
# would cost 15 and put this benchmark in swe-bench-pro's runtime class.
PER_CATEGORY = 15
PARTITIONS = ("development", "validation", "test")
PER_CATEGORY_SPLIT = {"development": 3, "validation": 6, "test": 6}
TARGET_COUNTS = {"development": 30, "validation": 60, "test": 60}
EXPECTED_TASKS = 300
EXPECTED_CATEGORIES = 10


def _stable_key(task_name: str) -> str:
    return hashlib.sha256(f"{SEED}:{task_name}".encode()).hexdigest()


def _category(task_name: str) -> str:
    # "stanford/task5_20" -> "task5"
    return task_name.split("/", 1)[1].split("_", 1)[0]


async def _fetch_registry_refs() -> tuple[str, dict[str, str]]:
    from harbor.registry.client.package import PackageDatasetClient

    metadata = await PackageDatasetClient().get_dataset_metadata(TASK_SOURCE)
    refs = {task.get_name(): str(task.ref) for task in metadata.task_ids}
    if len(refs) != EXPECTED_TASKS:
        raise ValueError(f"registry returned {len(refs)} tasks, expected {EXPECTED_TASKS}")
    return str(metadata.version), refs


def _existing_refs(output_dir: Path) -> tuple[str, dict[str, str]]:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest["dataset_version"], {t["name"]: t["ref"] for t in manifest["tasks"]}


def _allocate(refs: dict[str, str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for name in refs:
        by_category[_category(name)].append(name)
    if len(by_category) != EXPECTED_CATEGORIES:
        raise ValueError(f"found {len(by_category)} categories, expected {EXPECTED_CATEGORIES}")

    partitions: dict[str, list[str]] = {p: [] for p in PARTITIONS}
    strata: dict[str, str] = {}
    for category in sorted(by_category, key=lambda name: int(name.removeprefix("task"))):
        members = sorted(by_category[category], key=_stable_key)
        if len(members) < PER_CATEGORY:
            raise ValueError(f"{category} has {len(members)} tasks, need {PER_CATEGORY}")
        chosen = members[:PER_CATEGORY]
        cursor = 0
        for partition in PARTITIONS:
            count = PER_CATEGORY_SPLIT[partition]
            for name in chosen[cursor : cursor + count]:
                partitions[partition].append(name)
                strata[name] = category
            cursor += count
    return {p: sorted(names) for p, names in partitions.items()}, strata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent.parent / "partitions"
    )
    parser.add_argument("--fetch-registry", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.fetch_registry:
        version, refs = asyncio.run(_fetch_registry_refs())
    else:
        version, refs = _existing_refs(args.output_dir)

    partitions, strata = _allocate(refs)
    for partition, names in partitions.items():
        if len(names) != TARGET_COUNTS[partition]:
            raise RuntimeError(
                f"{partition} holds {len(names)}, expected {TARGET_COUNTS[partition]}"
            )
    flattened = [name for names in partitions.values() for name in names]
    if len(set(flattened)) != len(flattened):
        raise RuntimeError("a task appears in more than one partition")

    manifest = {
        "schema_version": 1,
        "task_source": TASK_SOURCE,
        "dataset_name": DATASET_NAME,
        "dataset_version": version,
        "seed": SEED,
        "per_category": PER_CATEGORY,
        "partition_counts": TARGET_COUNTS,
        "category_counts": {
            partition: dict(sorted(Counter(strata[n] for n in names).items()))
            for partition, names in partitions.items()
        },
        "tasks": [
            {
                "name": name,
                "ref": refs[name],
                "partition": partition,
                "category": strata[name],
            }
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
        counts = manifest["category_counts"][partition]
        print(f"  {partition:<12}{TARGET_COUNTS[partition]:>4}  {len(counts)} categories x {sorted(set(counts.values()))}")


if __name__ == "__main__":
    main()
