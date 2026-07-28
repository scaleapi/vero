"""Tests for the post-hoc per-trial token aggregator.

Run with: uv run --python 3.12 python -m pytest \
    harness-engineering-bench/scripts/test_per_trial_tokens.py
"""

from __future__ import annotations

import json
from pathlib import Path

import per_trial_tokens as p


def _trial(root: Path, evaluation: str, index: int, task: str, question: str) -> None:
    trial_dir = (
        root / "evaluations" / evaluation / "artifacts" / "harbor" / "jobs" / "j"
        / f"t{index}"
    )
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "trial_name": f"t{index}",
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:05:00Z",
                "agent_result": {
                    "n_input_tokens": 100 + index,
                    "n_cache_tokens": 0,
                    "n_output_tokens": 5,
                },
            }
        )
    )
    (trial_dir / "agent" / "trace.jsonl").write_text(question)


def _write_log(root: Path, records: list[dict]) -> Path:
    requests = root / "artifacts" / "inference" / "requests"
    requests.mkdir(parents=True)
    (requests / "requests-00001.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    return requests


def test_stamped_log_attributes_stateful_chains_fully(tmp_path):
    evaluation = "eval-1"
    _trial(tmp_path, evaluation, 0, "org/task-A", "Question about alpha widgets")
    _trial(tmp_path, evaluation, 1, "org/task-B", "Question about beta gadgets")

    records = []
    for thread, question in (
        ("thA", "Question about alpha widgets"),
        ("thB", "Question about beta gadgets"),
    ):
        records.append(
            {
                "scope": "evaluation",
                "attribution": evaluation,
                "thread_id": thread,
                "root_snippet": question,
                "input_tokens": 1000,
                "cached_input_tokens": 400,
                "output_tokens": 50,
                "total_tokens": 1050,
                "latency_ms": 100,
                "ts": "2026-01-01T00:01:00Z",
            }
        )
        # two stateful follow-ups: thread_id only, no recoverable body
        for _ in range(2):
            records.append(
                {
                    "scope": "evaluation",
                    "attribution": evaluation,
                    "thread_id": thread,
                    "input_tokens": 500,
                    "cached_input_tokens": 200,
                    "output_tokens": 20,
                    "total_tokens": 520,
                    "latency_ms": 50,
                    "ts": "2026-01-01T00:02:00Z",
                }
            )
    requests = _write_log(tmp_path, records)

    threads = p.load_threads(requests)[evaluation]
    trials = p.load_trials(tmp_path)[evaluation]
    attributed = {}
    residual = 0
    for thread in threads.values():
        trial = p.assign_thread(thread, trials, [])
        if trial is None:
            residual += thread["total_tokens"]
        else:
            attributed[trial["task_name"]] = (
                attributed.get(trial["task_name"], 0) + thread["total_tokens"]
            )

    # every turn — root and both chained follow-ups — attributed to its trial
    assert attributed == {"org/task-A": 2090, "org/task-B": 2090}
    assert residual == 0


def test_legacy_log_labels_roots_via_tasks_dir_and_residualizes_chains(tmp_path):
    # Two concurrent trials (overlapping windows) so the time-window fallback
    # cannot disambiguate empty chained follow-ups — the legacy limitation.
    evaluation = "eval-1"
    _trial(tmp_path, evaluation, 0, "officeqa/uid0001", "unused-a")
    _trial(tmp_path, evaluation, 1, "officeqa/uid0002", "unused-b")
    for uid, question in (
        ("uid0001", "Compute the treasury yield for 1948"),
        ("uid0002", "Compute the bond spread for 1952"),
    ):
        task_dir = tmp_path / "tasks" / uid / "tests"
        task_dir.mkdir(parents=True)
        (task_dir / "prompt.txt").write_text(question)

    records = []
    for question in (
        "Compute the treasury yield for 1948",
        "Compute the bond spread for 1952",
    ):
        # root turn: question recoverable from the body
        records.append(
            {
                "scope": "evaluation",
                "attribution": evaluation,
                "request": {"text": json.dumps({"input": question})},
                "input_tokens": 5000,
                "cached_input_tokens": 0,
                "output_tokens": 100,
                "total_tokens": 5100,
                "latency_ms": 100,
                "ts": "2026-01-01T00:01:00Z",
            }
        )
    # empty-bodied stateful follow-ups from both trials, interleaved windows:
    # no unique covering trial → unrecoverable in legacy mode
    for _ in range(3):
        records.append(
            {
                "scope": "evaluation",
                "attribution": evaluation,
                "request": {"text": '{"previous_response_id":"resp_x","input":[]}'},
                "input_tokens": 4000,
                "cached_input_tokens": 0,
                "output_tokens": 80,
                "total_tokens": 4080,
                "latency_ms": 80,
                "ts": "2026-01-01T00:02:00Z",
            }
        )
    requests = _write_log(tmp_path, records)

    threads = p.load_threads(requests)[evaluation]
    trials = p.load_trials(tmp_path)[evaluation]
    task_texts = p.load_task_texts(tmp_path / "tasks")

    attributed = {}
    residual = 0
    for thread in threads.values():
        trial = p.assign_thread(thread, trials, task_texts)
        if trial is None:
            residual += thread["total_tokens"]
        else:
            attributed[trial["task_name"]] = (
                attributed.get(trial["task_name"], 0) + thread["total_tokens"]
            )

    # both root turns labeled to their tasks via the instruction files
    assert attributed == {"officeqa/uid0001": 5100, "officeqa/uid0002": 5100}
    # the empty-bodied follow-ups (ambiguous across two live trials) are the
    # honest residual — exactly the gap gateway-side stamping closes
    assert residual == 3 * 4080


def test_analyze_session_stamped_full_coverage_wall_and_reconciliation(tmp_path):
    evaluation = "eval-1"
    _trial(tmp_path, evaluation, 0, "org/task-A", "Question about alpha widgets")
    _trial(tmp_path, evaluation, 1, "org/task-B", "Question about beta gadgets")
    records = []
    for thread, question in (
        ("thA", "Question about alpha widgets"),
        ("thB", "Question about beta gadgets"),
    ):
        records.append(
            {
                "scope": "evaluation", "attribution": evaluation,
                "thread_id": thread, "root_snippet": question,
                "input_tokens": 1000, "cached_input_tokens": 400,
                "output_tokens": 50, "total_tokens": 1050,
                "latency_ms": 100, "ts": "2026-01-01T00:01:00Z",
            }
        )
        for _ in range(2):
            records.append(
                {
                    "scope": "evaluation", "attribution": evaluation,
                    "thread_id": thread,
                    "input_tokens": 500, "cached_input_tokens": 200,
                    "output_tokens": 20, "total_tokens": 520,
                    "latency_ms": 50, "ts": "2026-01-01T00:02:00Z",
                }
            )
    requests = _write_log(tmp_path, records)

    report = p.analyze_session(tmp_path, requests, [])
    data = report[evaluation]

    # report schema
    assert set(data) == {
        "trials", "distribution", "gateway_total", "attributed",
        "agent_reported_total", "residual", "coverage_pct",
    }
    # stamped chains attribute fully
    assert data["coverage_pct"] == 100.0
    assert data["residual"]["total_tokens"] == 0
    assert data["gateway_total"] == data["attributed"]

    a = data["trials"]["org/task-A"]
    # per-trial token triple: root + two follow-ups
    assert a["input_tokens"] == 1000 + 2 * 500
    assert a["cached_input_tokens"] == 400 + 2 * 200
    assert a["output_tokens"] == 50 + 2 * 20
    assert a["total_tokens"] == 2090
    assert a["requests"] == 3
    assert a["latency_ms"] == 100 + 2 * 50
    assert a["wall_s"] == 300.0  # one trial, 5-minute window, counted once
    # agent-reported carried per trial and summed across the eval for reconciliation
    assert a["agent_reported"]["n_input_tokens"] == 100
    assert data["agent_reported_total"]["n_input_tokens"] == 100 + 101


def test_csv_rows_schema_and_residual_row(tmp_path):
    evaluation = "eval-1"
    _trial(tmp_path, evaluation, 0, "org/task-A", "alpha")  # window 00:00-00:05
    records = [
        {
            "scope": "evaluation", "attribution": evaluation, "thread_id": "thA",
            "input_tokens": 10, "cached_input_tokens": 2, "output_tokens": 3,
            "total_tokens": 13, "latency_ms": 5, "ts": "2026-01-01T00:01:00Z",
        },
        # outside the only trial's window and unlabelled -> honest residual
        {
            "scope": "evaluation", "attribution": evaluation, "thread_id": "ghost",
            "input_tokens": 7, "cached_input_tokens": 0, "output_tokens": 1,
            "total_tokens": 8, "latency_ms": 4, "ts": "2026-01-01T00:10:00Z",
        },
    ]
    requests = _write_log(tmp_path, records)

    report = p.analyze_session(tmp_path, requests, [])
    rows = list(p.csv_rows("run-1", report))

    assert all(set(row) == set(p._CSV_FIELDS) for row in rows)
    trial_row = next(r for r in rows if r["task"] == "org/task-A")
    assert trial_row["run"] == "run-1"
    assert trial_row["total_tokens"] == 13
    residual_row = next(r for r in rows if r["task"] == "(unattributed)")
    assert residual_row["total_tokens"] == 8


def test_distribution_reports_mean_median_max_for_skewed_trials(tmp_path):
    # Heavy tail: one trial dominates. mean alone hides the shape, so the
    # aggregator reports median beside it and max names the tail.
    evaluation = "eval-1"
    sizes = {"a": 100, "b": 200, "c": 3000}
    for index, (suffix, tokens) in enumerate(sizes.items()):
        _trial(
            tmp_path, evaluation, index, f"org/task-{suffix}",
            f"Question about {suffix} widgets in the treasury bulletin corpus",
        )
    records = [
        {
            "scope": "evaluation", "attribution": evaluation,
            "thread_id": f"th{suffix}",
            "root_snippet": (
                f"Question about {suffix} widgets in the treasury bulletin corpus"
            ),
            "input_tokens": tokens, "cached_input_tokens": tokens // 10,
            "output_tokens": tokens // 20, "total_tokens": tokens,
            "latency_ms": tokens, "ts": "2026-01-01T00:01:00Z",
        }
        for suffix, tokens in sizes.items()
    ]
    requests = _write_log(tmp_path, records)

    report = p.analyze_session(tmp_path, requests, [])
    dist = report[evaluation]["distribution"]

    assert dist["input_tokens"]["mean"] == 1100.0     # dragged up by the tail
    assert dist["input_tokens"]["median"] == 200.0    # unmoved by it
    assert dist["input_tokens"]["max"] == 3000.0
    assert dist["input_tokens"]["n"] == 3
    assert dist["output_tokens"]["median"] == 10.0
    assert dist["latency_ms"]["median"] == 200.0
    # every trial shares one 5-minute window in the fixture
    assert dist["wall_s"]["median"] == 300.0
