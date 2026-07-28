#!/usr/bin/env python3
"""Post-hoc per-trial token/latency accounting from the gateway request log.

Reads a VeRO session directory (extracted ``session.tar.gz``, a salvage copy,
or the admin volume) and attributes the gateway's request-log records to the
Harbor trials they served, without any harness cooperation:

1. Trials are inventoried from ``evaluations/*/artifacts/harbor/**/result.json``
   (task name, wall-clock window, agent-reported tokens) together with their
   agent trace files, which serve as per-trial content ground truth.
2. Records are grouped into conversation threads — by the gateway's stamped
   ``thread_id`` when present (``request_log.attribution`` builds), otherwise
   by a first-user-message snippet recovered from the truncated body.
3. Threads map to trials by snippet containment in the trial's own traces or
   (with --tasks-dir) task instruction files, falling back to a unique trial
   time-window overlap. Anything else is an explicit unattributed residual —
   per-trial numbers are lower bounds; the gateway per-evaluation totals are
   the envelope.

Coverage note: on **stamped** logs (``request_log.attribution: true``) every
turn inherits its conversation's thread_id, so stateful APIs (OpenAI
responses' ``previous_response_id``) attribute fully. On **legacy** logs the
fallback recovers only each conversation's root turn — chained follow-ups
with empty bodies are unrecoverable and land in the residual. Enable
attribution at build time for complete coverage.

Pass one session dir for a single run, or several to roll up a grid; ``--csv``
writes a flat per-trial table across all of them (one row per
run/evaluation/task) for spreadsheet analysis. Tokens are the reported unit;
dollars are a downstream linear function of the (input, cached, output) triple
with a per-model rate vector, so they are deliberately not computed here.

Usage: per_trial_tokens.py SESSION_DIR [SESSION_DIR ...] [--requests-dir DIR]
       [--tasks-dir DIR] [--json] [--csv OUT.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

TOKEN_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")
_INPUT_PATTERN = re.compile(r'"input"\s*:\s*"((?:[^"\\]|\\.){20,600})')
_USER_CONTENT_PATTERN = re.compile(
    r'"role"\s*:\s*"user"[^{}\[\]]*?"(?:content|text)"\s*:\s*"((?:[^"\\]|\\.){20,600})'
)
_ESCAPES = [("\\n", " "), ("\\t", " "), ('\\"', '"'), ("\\\\", "\\")]


def normalize(text: str) -> str:
    return "".join(character for character in text.lower() if character.isalnum())


def parse_time(value: str | None) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_trials(session: Path) -> dict[str, list[dict]]:
    """Per evaluation id: trials with window, tokens, and normalized trace."""
    by_evaluation: dict[str, list[dict]] = defaultdict(list)
    for result_path in session.glob("evaluations/*/artifacts/harbor/**/result.json"):
        try:
            value = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or "task_name" not in value:
            continue
        evaluation_id = result_path.relative_to(session / "evaluations").parts[0]
        trace_chunks = []
        agent_dir = result_path.parent / "agent"
        if agent_dir.is_dir():
            for path in sorted(agent_dir.rglob("*")):
                if path.is_file() and path.stat().st_size < 20_000_000:
                    try:
                        trace_chunks.append(
                            path.read_text(encoding="utf-8", errors="replace")
                        )
                    except OSError:
                        continue
        agent_result = value.get("agent_result") or {}
        by_evaluation[evaluation_id].append(
            {
                "task_name": value.get("task_name"),
                "trial_name": value.get("trial_name"),
                "started": parse_time(value.get("started_at")),
                "finished": parse_time(value.get("finished_at")),
                "agent_reported": {
                    key: agent_result.get(key)
                    for key in ("n_input_tokens", "n_cache_tokens", "n_output_tokens")
                    if isinstance(agent_result.get(key), (int, float))
                },
                "trace": normalize("\n".join(trace_chunks)),
            }
        )
    return by_evaluation


def recover_snippet(record: dict) -> str | None:
    """First-user-message snippet for legacy (unstamped) records."""
    text = (record.get("request") or {}).get("text") or ""
    match = _INPUT_PATTERN.search(text) or _USER_CONTENT_PATTERN.search(text)
    if not match:
        return None
    snippet = match.group(1)
    for escaped, plain in _ESCAPES:
        snippet = snippet.replace(escaped, plain)
    return snippet[:200]


def load_threads(requests_dir: Path) -> dict[str, dict[str, dict]]:
    """Per evaluation id: threads with summed usage, snippet, and time span."""
    per_evaluation: dict[str, dict[str, dict]] = defaultdict(dict)
    for log_path in sorted(requests_dir.glob("requests-*.jsonl")):
        with open(log_path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("scope") not in ("evaluation", "finalization"):
                    continue
                # Salvage copies can lack attribution; a None key would sort
                # against the real ones and crash the report.
                evaluation_id = record.get("attribution") or "unattributed"
                snippet = record.get("root_snippet") or recover_snippet(record)
                thread_key = record.get("thread_id") or (
                    "snippet:" + normalize(snippet or "")[:160] or "unattributed"
                )
                thread = per_evaluation[evaluation_id].setdefault(
                    thread_key,
                    {
                        "requests": 0,
                        "latency_ms": 0.0,
                        "snippet": None,
                        "first": None,
                        "last": None,
                        **{key: 0 for key in TOKEN_KEYS},
                    },
                )
                thread["requests"] += 1
                thread["latency_ms"] += record.get("latency_ms") or 0.0
                for key in TOKEN_KEYS:
                    value = record.get(key)
                    if isinstance(value, (int, float)):
                        thread[key] += int(value)
                if snippet and not thread["snippet"]:
                    thread["snippet"] = snippet
                timestamp = parse_time(record.get("ts"))
                if timestamp is not None:
                    if thread["first"] is None or timestamp < thread["first"]:
                        thread["first"] = timestamp
                    if thread["last"] is None or timestamp > thread["last"]:
                        thread["last"] = timestamp
    return per_evaluation


def load_task_texts(tasks_dir: Path | None) -> list[tuple[str, str]]:
    """(task_name, normalized instruction) for snippet→task labeling."""
    if tasks_dir is None:
        return []
    texts = []
    for task_dir in sorted(tasks_dir.glob("*")):
        if not task_dir.is_dir():
            continue
        chunks = []
        for name in ("tests/prompt.txt", "instruction.md", "task.toml"):
            path = task_dir / name
            if path.is_file():
                try:
                    chunks.append(path.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    continue
        if chunks:
            texts.append((task_dir.name, normalize("\n".join(chunks))))
    return texts


def assign_thread(
    thread: dict, trials: list[dict], task_texts: list[tuple[str, str]]
) -> dict | None:
    snippet = normalize(thread["snippet"] or "")
    if len(snippet) >= 24:
        matches = [trial for trial in trials if snippet in trial["trace"]]
        if len(matches) == 1:
            return matches[0]
        # Label via task instruction files, then resolve to the trial(s) that
        # ran that task (>1 with repeated passes → tokens split evenly).
        labeled = [name for name, text in task_texts if snippet[:120] in text]
        if len(labeled) == 1:
            by_task = [t for t in trials if str(t["task_name"]).endswith(labeled[0])]
            if by_task:
                return by_task[0]
    if thread["first"] is not None and thread["last"] is not None:
        slack = timedelta(seconds=10)
        covering = [
            trial
            for trial in trials
            if trial["started"] is not None
            and trial["finished"] is not None
            and trial["started"] - slack <= thread["first"]
            and thread["last"] <= trial["finished"] + slack
        ]
        if len(covering) == 1:
            return covering[0]
    return None


_DISTRIBUTION_KEYS = (*TOKEN_KEYS, "requests", "latency_ms", "wall_s")


def distribution(entries: list[dict]) -> dict[str, dict[str, float]]:
    """mean/median/max per-trial, for each token and latency measure.

    Token and latency distributions are unbounded and heavy-tailed — a few
    trials carry much of an evaluation's total — so the median is reported
    beside the mean and the max names the tail. (Accuracy, being bounded, is
    summarized by its mean alone.)
    """
    summary: dict[str, dict[str, float]] = {}
    for key in _DISTRIBUTION_KEYS:
        values = [
            entry[key]
            for entry in entries
            if isinstance(entry.get(key), (int, float))
        ]
        if not values:
            continue
        summary[key] = {
            "mean": round(sum(values) / len(values), 1),
            "median": round(statistics.median(values), 1),
            "max": round(float(max(values)), 1),
            "n": len(values),
        }
    return summary


def trial_wall_seconds(trial: dict) -> float | None:
    started, finished = trial.get("started"), trial.get("finished")
    if started is None or finished is None:
        return None
    return round((finished - started).total_seconds(), 1)


def analyze_session(
    session: Path, requests_dir: Path, task_texts: list[tuple[str, str]]
) -> dict:
    """Per evaluation id: per-trial token triples + latency + wall, with the
    trusted gateway envelope, the attributed sum, the independent agent-reported
    sum (for reconciliation), and the unattributed residual."""
    trials_by_evaluation = load_trials(session)
    threads_by_evaluation = load_threads(requests_dir)

    report: dict[str, dict] = {}
    for evaluation_id, threads in sorted(threads_by_evaluation.items()):
        trials = trials_by_evaluation.get(evaluation_id, [])
        per_trial: dict[str, dict] = {}
        seen: dict[str, set] = defaultdict(set)
        residual = {"requests": 0, **{key: 0 for key in TOKEN_KEYS}}
        for thread in threads.values():
            trial = assign_thread(thread, trials, task_texts)
            if trial is None:
                residual["requests"] += thread["requests"]
                for key in TOKEN_KEYS:
                    residual[key] += thread[key]
                continue
            task_name = trial["task_name"]
            entry = per_trial.setdefault(
                task_name,
                {
                    "requests": 0,
                    "latency_ms": 0.0,
                    "wall_s": 0.0,
                    "agent_reported": trial["agent_reported"],
                    **{key: 0 for key in TOKEN_KEYS},
                },
            )
            entry["requests"] += thread["requests"]
            entry["latency_ms"] += thread["latency_ms"]
            for key in TOKEN_KEYS:
                entry[key] += thread[key]
            # Wall is a per-trial property, so add each contributing trial once:
            # repeated passes of one task sum, but a trial's many threads don't
            # double-count its wall.
            trial_name = trial.get("trial_name")
            if trial_name not in seen[task_name]:
                seen[task_name].add(trial_name)
                wall = trial_wall_seconds(trial)
                if wall is not None:
                    entry["wall_s"] = round(entry["wall_s"] + wall, 1)
        gateway_total = {
            key: sum(thread[key] for thread in threads.values()) for key in TOKEN_KEYS
        }
        attributed = {
            key: sum(entry[key] for entry in per_trial.values()) for key in TOKEN_KEYS
        }
        agent_reported_total = {
            key: sum(
                entry["agent_reported"].get(key, 0) for entry in per_trial.values()
            )
            for key in ("n_input_tokens", "n_cache_tokens", "n_output_tokens")
        }
        report[evaluation_id] = {
            "trials": per_trial,
            "distribution": distribution(list(per_trial.values())),
            "gateway_total": gateway_total,
            "attributed": attributed,
            "agent_reported_total": agent_reported_total,
            "residual": residual,
            "coverage_pct": (
                round(
                    100.0 * attributed["total_tokens"] / gateway_total["total_tokens"],
                    1,
                )
                if gateway_total["total_tokens"]
                else None
            ),
        }
    return report


_CSV_FIELDS = [
    "run",
    "evaluation",
    "task",
    "requests",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "total_tokens",
    "latency_ms",
    "wall_s",
    "agent_reported_input_tokens",
    "agent_reported_cache_tokens",
    "agent_reported_output_tokens",
]


def csv_rows(run_label: str, report: dict):
    """Flat per-trial rows (plus one residual row per evaluation) for the grid."""
    for evaluation_id, data in report.items():
        for task_name, entry in sorted(data["trials"].items()):
            reported = entry["agent_reported"]
            yield {
                "run": run_label,
                "evaluation": evaluation_id,
                "task": task_name,
                "requests": entry["requests"],
                "input_tokens": entry["input_tokens"],
                "cached_input_tokens": entry["cached_input_tokens"],
                "output_tokens": entry["output_tokens"],
                "total_tokens": entry["total_tokens"],
                "latency_ms": round(entry["latency_ms"], 1),
                "wall_s": entry["wall_s"],
                "agent_reported_input_tokens": reported.get("n_input_tokens"),
                "agent_reported_cache_tokens": reported.get("n_cache_tokens"),
                "agent_reported_output_tokens": reported.get("n_output_tokens"),
            }
        residual = data["residual"]
        if residual["requests"]:
            yield {
                "run": run_label,
                "evaluation": evaluation_id,
                "task": "(unattributed)",
                "requests": residual["requests"],
                "input_tokens": residual["input_tokens"],
                "cached_input_tokens": residual["cached_input_tokens"],
                "output_tokens": residual["output_tokens"],
                "total_tokens": residual["total_tokens"],
                "latency_ms": "",
                "wall_s": "",
                "agent_reported_input_tokens": "",
                "agent_reported_cache_tokens": "",
                "agent_reported_output_tokens": "",
            }


def print_report(run_label: str, report: dict) -> None:
    for evaluation_id, data in report.items():
        print(f"\n=== {run_label} / evaluation {evaluation_id} ===")
        print(
            f"{'task':<40} {'req':>5} {'input':>10} {'cached':>10} "
            f"{'output':>8} {'wall_s':>8} {'agent in/cache/out':>24}"
        )
        for task_name, entry in sorted(data["trials"].items()):
            reported = entry["agent_reported"]
            reported_text = "/".join(
                str(reported.get(key, "-"))
                for key in ("n_input_tokens", "n_cache_tokens", "n_output_tokens")
            )
            print(
                f"{str(task_name)[:40]:<40} {entry['requests']:>5} "
                f"{entry['input_tokens']:>10} {entry['cached_input_tokens']:>10} "
                f"{entry['output_tokens']:>8} {entry['wall_s']:>8} {reported_text:>24}"
            )
        residual = data["residual"]
        print(
            f"{'(unattributed)':<40} {residual['requests']:>5} "
            f"{residual['input_tokens']:>10} {residual['cached_input_tokens']:>10} "
            f"{residual['output_tokens']:>8}"
        )
        gateway, attributed = data["gateway_total"], data["attributed"]
        agent = data["agent_reported_total"]
        print(
            f"coverage: {data['coverage_pct']}% of {gateway['total_tokens']} "
            f"gateway-metered tokens attributed to trials"
        )
        if data["distribution"]:
            print(f"per-trial distribution  {'mean':>12} {'median':>12} {'max':>12}")
            for key, stats in data["distribution"].items():
                print(
                    f"  {key:<20} {stats['mean']:>12,.1f} "
                    f"{stats['median']:>12,.1f} {stats['max']:>12,.1f}"
                )
        # Reconcile the trusted envelope against the two independent token sources.
        print(
            f"  input tokens  gateway {gateway['input_tokens']} | "
            f"attributed {attributed['input_tokens']} | "
            f"agent-reported {agent['n_input_tokens']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", type=Path, nargs="+", help="one or more session dirs")
    parser.add_argument(
        "--requests-dir", type=Path, default=None, help="single session only"
    )
    parser.add_argument("--tasks-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--csv", type=Path, default=None, help="write a flat per-trial CSV")
    arguments = parser.parse_args()

    task_texts = load_task_texts(arguments.tasks_dir)
    grid: dict[str, dict] = {}
    for session in arguments.sessions:
        if arguments.requests_dir is not None and len(arguments.sessions) == 1:
            requests_dir = arguments.requests_dir
        else:
            requests_dir = session / "artifacts" / "inference" / "requests"
        if not requests_dir.is_dir():
            print(f"no request log at {requests_dir}", file=sys.stderr)
            continue
        # "." resolves to an empty name, which would label every row blank.
        run_label = session.resolve().name or str(session)
        grid[run_label] = analyze_session(session, requests_dir, task_texts)

    if not grid:
        return 1

    if arguments.csv is not None:
        with open(arguments.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for run_label, report in grid.items():
                writer.writerows(csv_rows(run_label, report))
        print(f"wrote {arguments.csv}", file=sys.stderr)

    if arguments.as_json:
        # One session keeps the original flat {evaluation: ...} schema; several
        # nest under their run label.
        payload = next(iter(grid.values())) if len(grid) == 1 else grid
        print(json.dumps(payload, indent=1, default=str))
        return 0

    for run_label, report in grid.items():
        print_report(run_label, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
