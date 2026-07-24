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

Usage: per_trial_tokens.py SESSION_DIR [--requests-dir DIR] [--tasks-dir DIR]
       [--json]
"""

from __future__ import annotations

import argparse
import json
import re
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
                evaluation_id = record.get("attribution")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--requests-dir", type=Path, default=None)
    parser.add_argument("--tasks-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    arguments = parser.parse_args()

    session = arguments.session
    requests_dir = (
        arguments.requests_dir or session / "artifacts" / "inference" / "requests"
    )
    if not requests_dir.is_dir():
        print(f"no request log at {requests_dir}", file=sys.stderr)
        return 1

    trials_by_evaluation = load_trials(session)
    threads_by_evaluation = load_threads(requests_dir)
    task_texts = load_task_texts(arguments.tasks_dir)

    report = {}
    for evaluation_id, threads in sorted(threads_by_evaluation.items()):
        trials = trials_by_evaluation.get(evaluation_id, [])
        per_trial: dict[str, dict] = {}
        residual = {"requests": 0, **{key: 0 for key in TOKEN_KEYS}}
        for thread in threads.values():
            trial = assign_thread(thread, trials, task_texts)
            if trial is None:
                residual["requests"] += thread["requests"]
                for key in TOKEN_KEYS:
                    residual[key] += thread[key]
                continue
            entry = per_trial.setdefault(
                trial["task_name"],
                {
                    "requests": 0,
                    "latency_ms": 0.0,
                    "agent_reported": trial["agent_reported"],
                    **{key: 0 for key in TOKEN_KEYS},
                },
            )
            entry["requests"] += thread["requests"]
            entry["latency_ms"] += thread["latency_ms"]
            for key in TOKEN_KEYS:
                entry[key] += thread[key]
        total = {
            key: sum(thread[key] for thread in threads.values()) for key in TOKEN_KEYS
        }
        attributed = {
            key: sum(entry[key] for entry in per_trial.values()) for key in TOKEN_KEYS
        }
        report[evaluation_id] = {
            "trials": per_trial,
            "gateway_total": total,
            "attributed": attributed,
            "residual": residual,
            "coverage_pct": (
                round(100.0 * attributed["total_tokens"] / total["total_tokens"], 1)
                if total["total_tokens"]
                else None
            ),
        }

    if arguments.as_json:
        print(json.dumps(report, indent=1, default=str))
        return 0

    for evaluation_id, data in report.items():
        print(f"\n=== evaluation {evaluation_id} ===")
        header = f"{'task':<44} {'req':>5} {'input':>10} {'cached':>10} {'output':>8} {'agent-reported in/cache/out':>28}"
        print(header)
        for task_name, entry in sorted(data["trials"].items()):
            reported = entry["agent_reported"]
            reported_text = "/".join(
                str(reported.get(key, "-"))
                for key in ("n_input_tokens", "n_cache_tokens", "n_output_tokens")
            )
            print(
                f"{str(task_name)[:44]:<44} {entry['requests']:>5} "
                f"{entry['input_tokens']:>10} {entry['cached_input_tokens']:>10} "
                f"{entry['output_tokens']:>8} {reported_text:>28}"
            )
        residual = data["residual"]
        print(
            f"{'(unattributed)':<44} {residual['requests']:>5} "
            f"{residual['input_tokens']:>10} {residual['cached_input_tokens']:>10} "
            f"{residual['output_tokens']:>8}"
        )
        print(
            f"coverage: {data['coverage_pct']}% of "
            f"{data['gateway_total']['total_tokens']} gateway-metered tokens attributed"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
