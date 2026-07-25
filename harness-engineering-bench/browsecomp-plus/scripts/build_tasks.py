#!/usr/bin/env python3
"""Build reproducible Harbor tasks from the pinned BrowseComp-Plus dataset."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

UPSTREAM_COMMIT = "046949032b0328319cc9a02663a759ec601d9402"
DATASET_NAME = "Tevatron/browsecomp-plus"
DATASET_REVISION = "144cff8e35b5eaef7e526346aa60774a9deb941f"
INDEX_DATASET = "Tevatron/browsecomp-plus-indexes"
INDEX_REVISION = "b3f37f70c33829eb09d04784a54277a31871fd63"
EXPECTED_TASKS = 830
SEED = "vero-browsecomp-plus-v1"
PARTITION_COUNTS = {"development": 33, "validation": 66, "test": 66}

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = BENCHMARK_DIR / "upstream"
TEMPLATE_DIR = BENCHMARK_DIR / "task-template"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "tasks"
DEFAULT_PARTITIONS_DIR = BENCHMARK_DIR / "partitions"


@dataclass(frozen=True)
class TaskRecord:
    query_id: str
    query: str
    answer: str
    gold_docids: tuple[str, ...]
    evidence_docids: tuple[str, ...]

    @property
    def task_id(self) -> str:
        suffix = self.query_id.zfill(4) if self.query_id.isdigit() else self.query_id
        if re.fullmatch(r"[A-Za-z0-9_.-]+", suffix) is None:
            raise ValueError(f"unsafe query id: {self.query_id!r}")
        return f"browsecomp-plus-q{suffix}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--partitions-dir", type=Path, default=DEFAULT_PARTITIONS_DIR)
    parser.add_argument(
        "--force", action="store_true", help="replace an existing generated task set"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify generated tasks and partitions without changing them",
    )
    return parser.parse_args()


def _upstream_commit() -> str:
    if not (UPSTREAM_DIR / ".git").exists():
        raise RuntimeError(
            "BrowseComp-Plus submodule is missing; run "
            "`git submodule update --init harness-engineering-bench/"
            "browsecomp-plus/upstream`"
        )
    result = subprocess.run(
        ["git", "-C", str(UPSTREAM_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"BrowseComp-Plus submodule is at {commit}, expected {UPSTREAM_COMMIT}"
        )
    return commit


def _load_decrypter() -> tuple[Callable[[Any, str, set[str]], Any], str]:
    source = UPSTREAM_DIR / "scripts_build_index" / "decrypt_dataset.py"
    spec = importlib.util.spec_from_file_location("browsecomp_plus_decrypt", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load upstream decrypter from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    typed_module: ModuleType = module
    transform = typed_module.transform_decrypt
    canary = typed_module.DEFAULT_CANARY
    if not callable(transform) or not isinstance(canary, str):
        raise TypeError("upstream decrypter has an unexpected interface")
    return transform, canary


def _document_ids(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("docid"), str):
            raise TypeError(f"{field} contains an invalid document")
        result.append(item["docid"])
    return result


def load_records() -> list[TaskRecord]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "the builder requires `datasets==4.0.0`; run it with "
            "`uv run --no-project --python 3.12 --with datasets==4.0.0 -- "
            "python scripts/build_tasks.py`"
        ) from error

    _upstream_commit()
    decrypt, canary = _load_decrypter()
    dataset = load_dataset(
        DATASET_NAME,
        split="test",
        revision=DATASET_REVISION,
    )
    if len(dataset) != EXPECTED_TASKS:
        raise ValueError(f"expected {EXPECTED_TASKS} rows, found {len(dataset)}")

    records: list[TaskRecord] = []
    for raw in dataset:
        encrypted = {
            "query": raw["query"],
            "answer": raw["answer"],
            "gold_docids": _document_ids(raw["gold_docs"], "gold_docs"),
            "evidence_docids": _document_ids(raw["evidence_docs"], "evidence_docs"),
        }
        row = decrypt(encrypted, canary, set())
        record = TaskRecord(
            query_id=str(raw["query_id"]),
            query=str(row["query"]).strip(),
            answer=str(row["answer"]).strip(),
            gold_docids=tuple(str(item) for item in row["gold_docids"]),
            evidence_docids=tuple(str(item) for item in row["evidence_docids"]),
        )
        if not record.query or not record.answer:
            raise ValueError(f"query {record.query_id} is missing text or an answer")
        records.append(record)

    if len({record.query_id for record in records}) != EXPECTED_TASKS:
        raise ValueError("dataset contains duplicate query ids")
    if len({record.task_id for record in records}) != EXPECTED_TASKS:
        raise ValueError("dataset produces duplicate Harbor task ids")
    return sorted(records, key=lambda record: record.task_id)


def _partition(records: list[TaskRecord]) -> dict[str, list[str]]:
    # A deterministic seeded shuffle picks the first sum(PARTITION_COUNTS)
    # tasks and splits them; the remaining dataset rows are rendered as task
    # dirs but left unassigned to any partition (a size reduction that keeps
    # the split stratified and reproducible).
    if len(records) < sum(PARTITION_COUNTS.values()):
        raise ValueError("partition counts exceed the dataset size")
    ordered = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{SEED}:{record.task_id}".encode()
        ).hexdigest(),
    )
    result: dict[str, list[str]] = {}
    cursor = 0
    for name, count in PARTITION_COUNTS.items():
        result[name] = sorted(
            record.task_id for record in ordered[cursor : cursor + count]
        )
        cursor += count
    return result


def _task_toml(record: TaskRecord) -> str:
    return f'''version = "1.0"

[task]
name = "browsecomp-plus/{record.task_id}"

[metadata]
author_name = "BrowseComp-Plus authors"
author_email = "s42chen@uwaterloo.ca"
benchmark = "BrowseComp-Plus"
source = "{DATASET_NAME}"
source_id = {json.dumps(record.query_id)}
upstream_commit = "{UPSTREAM_COMMIT}"
dataset_revision = "{DATASET_REVISION}"
retriever = "BM25"
tags = ["qa", "deep-research", "retrieval", "browsecomp-plus"]

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = 3600.0

[environment]
build_timeout_sec = 7200.0
cpus = 2
memory_mb = 8192
storage_mb = 10240
allow_internet = true
'''


def _instruction(record: TaskRecord) -> str:
    return f"""You are a deep-research agent working against the fixed BrowseComp-Plus corpus.

Use the provided `search` and `get_document` tools to find and cross-check the answer. The retrieval corpus and BM25 index are fixed; do not use the live web.

Question: {record.query}

Write the final response to `/app/answer.txt` in this exact shape:

```text
Explanation: <your reasoning, citing supporting document ids in square brackets>
Exact Answer: <a succinct final answer>
Confidence: <a number from 0% to 100%>
```
"""


def _solution(record: TaskRecord) -> str:
    response = (
        "Explanation: Oracle answer from the benchmark ground truth.\n"
        f"Exact Answer: {record.answer}\n"
        "Confidence: 100%\n"
    )
    encoded = base64.b64encode(response.encode()).decode("ascii")
    return f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s' '{encoded}' | base64 --decode > /app/answer.txt
"""


def _config(record: TaskRecord) -> str:
    return (
        json.dumps(
            {
                "query_id": record.query_id,
                "question": record.query,
                "expected_answer": record.answer,
                "gold_docids": list(record.gold_docids),
                "evidence_docids": list(record.evidence_docids),
                "dataset_revision": DATASET_REVISION,
                "upstream_commit": UPSTREAM_COMMIT,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def _write_task(root: Path, record: TaskRecord) -> None:
    task_dir = root / record.task_id
    shutil.copytree(
        TEMPLATE_DIR,
        task_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.py[oc]"),
    )
    (task_dir / "task.toml").write_text(_task_toml(record), encoding="utf-8")
    (task_dir / "instruction.md").write_text(_instruction(record), encoding="utf-8")
    solution = task_dir / "solution" / "solve.sh"
    solution.parent.mkdir()
    solution.write_text(_solution(record), encoding="utf-8")
    solution.chmod(0o755)
    (task_dir / "tests" / "config.json").write_text(_config(record), encoding="utf-8")
    (task_dir / "tests" / "test.sh").chmod(0o755)


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _render_tasks(root: Path, records: list[TaskRecord]) -> None:
    root.mkdir(parents=True)
    for record in records:
        _write_task(root, record)
    build_manifest = {
        "schema_version": 1,
        "upstream_repository": "https://github.com/texttron/BrowseComp-Plus.git",
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "index_dataset": INDEX_DATASET,
        "index_revision": INDEX_REVISION,
        "task_count": len(records),
        "tasks_digest": f"sha256:{_tree_digest(root)}",
    }
    (root / ".build-manifest.json").write_text(
        json.dumps(build_manifest, indent=2) + "\n", encoding="utf-8"
    )


def _partition_files(
    records: list[TaskRecord], partitions: dict[str, list[str]]
) -> dict[str, str]:
    membership = {
        task_id: partition
        for partition, task_ids in partitions.items()
        for task_id in task_ids
    }
    manifest = {
        "schema_version": 1,
        "task_source": "../tasks",
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "upstream_repository": "https://github.com/texttron/BrowseComp-Plus.git",
        "upstream_commit": UPSTREAM_COMMIT,
        "index_dataset": INDEX_DATASET,
        "index_revision": INDEX_REVISION,
        "seed": SEED,
        "ratios": {"development": 0.2, "validation": 0.4, "test": 0.4},
        "partition_counts": PARTITION_COUNTS,
        "tasks": [
            {
                "name": record.task_id,
                "source_id": record.query_id,
                "partition": membership[record.task_id],
            }
            for record in records
            if record.task_id in membership
        ],
    }
    files = {
        f"{name}.json": json.dumps(task_ids, indent=2) + "\n"
        for name, task_ids in partitions.items()
    }
    files["manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    return files


def _check_partitions(path: Path, expected: dict[str, str]) -> list[str]:
    return [
        name
        for name, content in expected.items()
        if not (path / name).is_file()
        or (path / name).read_text(encoding="utf-8") != content
    ]


def main() -> None:
    args = _parse_args()
    records = load_records()
    partitions = _partition(records)
    partition_files = _partition_files(records, partitions)
    output_dir = args.output_dir.expanduser().resolve()
    partitions_dir = args.partitions_dir.expanduser().resolve()

    with tempfile.TemporaryDirectory(
        prefix="browsecomp-plus-tasks-", dir=output_dir.parent
    ) as temporary:
        rendered = Path(temporary) / "tasks"
        _render_tasks(rendered, records)
        if args.check:
            stale = _check_partitions(partitions_dir, partition_files)
            if not output_dir.is_dir() or _tree_digest(output_dir) != _tree_digest(
                rendered
            ):
                stale.append(str(output_dir))
            if stale:
                raise SystemExit(
                    "generated BrowseComp-Plus files are stale: " + ", ".join(stale)
                )
            print(f"verified {len(records)} BrowseComp-Plus Harbor tasks")
            return

        if output_dir.exists():
            if not args.force:
                raise SystemExit(
                    f"{output_dir} already exists; pass --force to replace it"
                )
            if output_dir == Path(output_dir.anchor) or len(output_dir.parts) < 3:
                raise SystemExit(
                    f"refusing to replace unsafe output path: {output_dir}"
                )
            shutil.rmtree(output_dir)
        os.replace(rendered, output_dir)

    partitions_dir.mkdir(parents=True, exist_ok=True)
    for name, content in partition_files.items():
        (partitions_dir / name).write_text(content, encoding="utf-8")
    print(f"wrote {len(records)} Harbor tasks to {output_dir}")


if __name__ == "__main__":
    main()
