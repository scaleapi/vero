#!/usr/bin/env python3
"""Report which benchmarks have their task data, and how to fetch what is missing.

    python3 scripts/task_data.py            # status for every benchmark
    python3 scripts/task_data.py --check    # same, but exit 1 if anything is missing

Vendored task data is gitignored on purpose -- officeqa's 246 and
browsecomp-plus's 830 task directories are hundreds of megabytes and thousands of
files, and committing them once bloated the repository and timed out the
pre-commit secret scan. The cost is that a fresh checkout cannot run those two
benchmarks until the data is fetched, and nothing tells you so: the failure
surfaces from deep inside config validation.

Each benchmark already has its own fetcher, but they differ in language,
interface and location, so there was no single place to ask "what do I need?".
That is what this script is. It fetches nothing itself -- it reports state and
names the exact command -- because the fetchers are slow, network-bound and
occasionally destructive, and that is a decision for whoever is at the keyboard.

Counts come from CONFIGURATION.md, which records them as a property of each
dataset ("246 officeqa tasks all at 1800, 830 browsecomp tasks all at 3600"). A
count that disagrees means a partial or interrupted fetch, which is worth
knowing: a half-vendored directory validates fine and then scores a subset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]


class Benchmark:
    def __init__(self, name: str, glob: str, expected: int | None, how: str, note: str):
        self.name = name
        self.glob = glob
        self.expected = expected
        self.how = how
        self.note = note

    @property
    def tasks_dir(self) -> Path:
        return BENCH_DIR / self.name / "tasks"

    def count(self) -> int:
        if not self.tasks_dir.is_dir():
            return 0
        return sum(1 for _ in self.tasks_dir.glob(self.glob))


BENCHMARKS = [
    Benchmark(
        "officeqa",
        "officeqa-uid*",
        246,
        "bash harness-engineering-bench/officeqa/scripts/vendor_tasks.sh",
        "sparse-checkout of datasets/officeqa from harbor-datasets (~4GB repo, "
        "blobless); the script also patches each task.toml with a canonical "
        "[task].name, which vero's local task staging requires",
    ),
    Benchmark(
        "browsecomp-plus",
        "browsecomp-plus-q*",
        830,
        "python3 harness-engineering-bench/browsecomp-plus/scripts/build_tasks.py",
        "builds from the pinned Tevatron/browsecomp-plus dataset revision plus the "
        "upstream submodule at harness-engineering-bench/browsecomp-plus/upstream "
        "(run `git submodule update --init` first); also needs the pinned BM25 "
        "index dataset",
    ),
    # No local task data: the build pins a registry digest and Harbor fetches it.
    Benchmark("gaia", "*", None, "", "registry task_source, nothing to vendor"),
    Benchmark("tau3", "*", None, "", "registry task_source, nothing to vendor"),
    Benchmark("swe-atlas-qna", "*", None, "", "registry task_source, nothing to vendor"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any benchmark's data is missing or the count disagrees",
    )
    args = parser.parse_args()

    problems = []
    for benchmark in BENCHMARKS:
        if benchmark.expected is None:
            print(f"{benchmark.name:18} n/a      {benchmark.note}")
            continue
        found = benchmark.count()
        if found == benchmark.expected:
            print(f"{benchmark.name:18} OK       {found} tasks")
            continue
        state = "MISSING " if found == 0 else "PARTIAL "
        print(f"{benchmark.name:18} {state} {found}/{benchmark.expected} tasks")
        print(f"{'':18}          fetch: {benchmark.how}")
        print(f"{'':18}          {benchmark.note}")
        problems.append(benchmark.name)

    if problems:
        print(
            f"\n{len(problems)} benchmark(s) cannot run until fetched: "
            + ", ".join(problems)
        )
        print(
            "Until then a launch fails during config validation, because "
            "task_source points at a path that does not resolve."
        )
    else:
        print("\nall benchmarks have their task data")
    return 1 if (problems and args.check) else 0


if __name__ == "__main__":
    sys.exit(main())
