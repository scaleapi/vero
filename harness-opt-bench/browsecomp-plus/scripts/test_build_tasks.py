from __future__ import annotations

import json

from build_tasks import (
    DATASET_REVISION,
    EXPECTED_TASKS,
    PARTITION_COUNTS,
    UPSTREAM_COMMIT,
    TaskRecord,
    _partition,
    _write_task,
)


def _record(index: int) -> TaskRecord:
    return TaskRecord(
        query_id=str(index),
        query=f"Question {index}?",
        answer=f"Answer {index}",
        gold_docids=(f"gold-{index}",),
        evidence_docids=(f"evidence-{index}",),
    )


def test_partition_is_complete_disjoint_and_deterministic():
    records = [_record(index) for index in range(EXPECTED_TASKS)]

    first = _partition(records)
    second = _partition(list(reversed(records)))

    assert first == second
    assert {name: len(tasks) for name, tasks in first.items()} == PARTITION_COUNTS
    flattened = [task for tasks in first.values() for task in tasks]
    assert len(flattened) == len(set(flattened)) == EXPECTED_TASKS


def test_generated_task_contains_pins_and_keeps_answer_in_tests(tmp_path):
    record = _record(7)
    _write_task(tmp_path, record)
    task = tmp_path / "browsecomp-plus-q0007"

    assert UPSTREAM_COMMIT in (task / "task.toml").read_text(encoding="utf-8")
    assert DATASET_REVISION in (task / "task.toml").read_text(encoding="utf-8")
    assert record.answer not in (task / "instruction.md").read_text(encoding="utf-8")
    config = json.loads((task / "tests" / "config.json").read_text(encoding="utf-8"))
    assert config["expected_answer"] == record.answer
    assert (task / "solution" / "solve.sh").stat().st_mode & 0o111
