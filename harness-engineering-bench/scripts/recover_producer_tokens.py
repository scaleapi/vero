#!/usr/bin/env python3
"""Recompute producer-scope input tokens for runs metered before the cache fix.

Until the gateway learned Anthropic's usage shape, it read `usage.input_tokens`
and ignored the `cache_read_input_tokens` / `cache_creation_input_tokens`
siblings. Anthropic counts only the slice of the prompt that was neither read
from nor written to the cache as `input_tokens`, so a cached optimizer turn
metered as 2 tokens. Output was captured correctly, which is why the numbers
looked plausible. Measured undercount on live runs: 15,000x to 47,000x.

Nothing needs re-running. The request log captures each response body, so the
true figures are recoverable after the fact:

    python3 harness-engineering-bench/scripts/recover_producer_tokens.py \
        runs/officeqa/opencode-sonnet-5/salvaged-session

Accepts a session directory, a directory holding `artifacts/inference/requests/`,
or a `requests-*.jsonl` file. Read-only: it reports, it does not rewrite
`usage.json`.

Reads the high-water mark per record rather than summing matches, mirroring the
fixed gateway: a streaming response carries input in `message_start` and a
cumulative output in `message_delta`, so summing would double count.

`coverage` is the share of successful requests whose captured body still held a
usage block. The request log keeps only the head and tail of each body, so a
coverage below 100% makes the recovered total a lower bound.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

FIELDS = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
)
PATTERNS = {name: re.compile(rf'"{name}":\s*(\d+)') for name in FIELDS}


def find_logs(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    for candidate in (
        target / "artifacts" / "inference" / "requests",
        target / "session" / "artifacts" / "inference" / "requests",
        target / "inference" / "requests",
        target / "requests",
        target,
    ):
        logs = sorted(candidate.glob("requests-*.jsonl"))
        if logs:
            return logs
    return sorted(target.rglob("requests-*.jsonl"))


def body_text(record: dict) -> str:
    response = record.get("response")
    if isinstance(response, dict):
        return response.get("text") or ""
    return response if isinstance(response, str) else ""


def report(target: Path) -> int:
    logs = find_logs(target)
    if not logs:
        print(f"{target}: no requests-*.jsonl found")
        return 1

    scopes: dict[str, dict[str, int]] = {}
    for log in logs:
        with log.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # a partial final line while a run is still live
                if record.get("status") != 200:
                    continue
                scope = scopes.setdefault(
                    record.get("scope") or "?",
                    dict.fromkeys(
                        (
                            "requests",
                            "covered",
                            "metered_input",
                            "metered_cached",
                            "uncached",
                            "read",
                            "written",
                        ),
                        0,
                    ),
                )
                scope["requests"] += 1
                scope["metered_input"] += record.get("input_tokens") or 0
                scope["metered_cached"] += record.get("cached_input_tokens") or 0
                text = body_text(record)
                if not text:
                    continue
                found = {
                    name: max(
                        (int(m) for m in pattern.findall(text)),
                        default=None,
                    )
                    for name, pattern in PATTERNS.items()
                }
                if found["cache_read_input_tokens"] is None and (
                    found["cache_creation_input_tokens"] is None
                ):
                    continue  # not an Anthropic body; already metered correctly
                scope["covered"] += 1
                scope["uncached"] += found["input_tokens"] or 0
                scope["read"] += found["cache_read_input_tokens"] or 0
                scope["written"] += found["cache_creation_input_tokens"] or 0

    print(f"{target}")
    for name, s in sorted(scopes.items()):
        recovered = s["uncached"] + s["read"] + s["written"]
        if not s["covered"]:
            print(
                f"  {name:13s} n={s['requests']:5d} "
                f"input={s['metered_input']:>13,}  (no Anthropic bodies; "
                "metering already correct)"
            )
            continue
        factor = recovered / s["metered_input"] if s["metered_input"] else float("inf")
        coverage = 100.0 * s["covered"] / s["requests"]
        print(
            f"  {name:13s} n={s['requests']:5d} coverage={coverage:5.1f}%\n"
            f"    metered   input={s['metered_input']:>13,} "
            f"cached={s['metered_cached']:>13,}\n"
            f"    recovered input={recovered:>13,} "
            f"cached={s['read']:>13,}  "
            f"(uncached {s['uncached']:,} + read {s['read']:,} "
            f"+ written {s['written']:,})\n"
            f"    undercounted by {factor:,.0f}x"
        )
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    return max(report(Path(a)) for a in sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
