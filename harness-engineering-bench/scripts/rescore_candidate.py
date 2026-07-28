#!/usr/bin/env python3
"""Re-score a candidate from a run whose verifier failed.

A verifier failure loses the reward but not the work: the exported
`session.tar.gz` carries `candidates/repository.git`, so every candidate commit
the optimizer produced is recoverable. This extracts one and scores it on the
held-out partition using *the same method that produced the pinned baselines* --
a plain `harbor run` over an explicit task list, no gateway, no sidecar, and
harbor's default timeout multiplier of 1.0 -- so the number is directly
comparable to `baseline_reward` rather than a second scoring pathway we would
have to argue is equivalent.

Deliberately NOT a sidecar restore. `vero harbor finalize` needs a live sidecar
and `vero harbor run` compiles to a temp dir, so its minted tokens are gone once
the run ends; recompiling would mint new ones. Re-using the baseline path avoids
all of that.

Run from the vero checkout so PyYAML and uv are available:

    cd vero && uv run python \\
      ../harness-engineering-bench/scripts/rescore_candidate.py \\
      --session ../runs/officeqa/claude-sonnet-5-run2/jobs/*/task__*/verifier/session.tar.gz \\
      --benchmark officeqa --cases 2 --rounds 1

Aggregation matches runs/recompute.py exactly: pooled mean over every scored
trial, with the stdev taken across round means.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent


def log(message: str) -> None:
    print(f"[rescore] {message}", flush=True)


def load_build(benchmark: str) -> dict:
    import yaml  # provided by the vero environment

    path = BENCH_ROOT / benchmark / "baseline" / "build.yaml"
    if not path.is_file():
        sys.exit(f"no build.yaml for benchmark {benchmark!r} at {path}")
    return yaml.safe_load(path.read_text()), path


def resolve_param(value: str) -> str:
    """Resolve a `${name:-default}` placeholder to its default."""
    match = re.fullmatch(r"\$\{[^:}]+:-([^}]*)\}", str(value))
    return match.group(1) if match else str(value)


def open_session(session: str, workdir: Path) -> Path:
    """Return a directory containing the session tree (unpacking if needed)."""
    path = Path(session)
    if path.is_dir():
        # Accept either the session root or its parent.
        if (path / "candidates").is_dir():
            return path
        inner = path / "session"
        if (inner / "candidates").is_dir():
            return inner
        sys.exit(f"{path} does not look like a session dir (no candidates/)")
    if not path.is_file():
        sys.exit(f"session not found: {path}")
    target = workdir / "session-extract"
    target.mkdir(parents=True, exist_ok=True)
    log(f"unpacking {path.name}")
    with tarfile.open(path) as archive:
        archive.extractall(target, filter="data")  # our own trusted export
    for candidate in (target / "session", target):
        if (candidate / "candidates").is_dir():
            return candidate
    sys.exit("no candidates/ inside the session archive")


def shipped_version(session_dir: Path, origin: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    # The verifier writes finalization.json into /logs/verifier/, i.e. *beside*
    # session.tar.gz rather than inside it.
    candidates = [origin.parent / "finalization.json", session_dir / "finalization.json"]
    for path in candidates:
        if path.is_file():
            data = json.loads(path.read_text())
            version = (data.get("candidate") or {}).get("version")
            if version:
                log(f"shipped candidate from {path.name}: {version[:12]}")
                return version
    # Fall back to the newest commit in the candidate repo.
    repo = session_dir / "candidates" / "repository.git"
    head = subprocess.run(
        ["git", f"--git-dir={repo}", "log", "--all", "-1", "--format=%H"],
        capture_output=True, text=True, check=False,
    )
    if head.returncode == 0 and head.stdout.strip():
        version = head.stdout.strip()
        log(f"no finalization.json; using newest candidate commit {version[:12]}")
        return version
    sys.exit("could not determine the candidate; pass --version SHA")


def extract_candidate(session_dir: Path, version: str, dest: Path) -> None:
    repo = session_dir / "candidates" / "repository.git"
    if not repo.is_dir():
        sys.exit(f"no candidate repository at {repo}")
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", f"--git-dir={repo}", "archive", version],
        capture_output=True, check=False,
    )
    if archive.returncode != 0:
        sys.exit(f"git archive failed: {archive.stderr.decode()[:400]}")
    with tarfile.open(fileobj=__import__("io").BytesIO(archive.stdout)) as tar:
        tar.extractall(dest, filter="data")  # our own candidate repo
    log(f"extracted candidate {version[:12]} -> {dest}")


def harbor_command(
    *,
    build: dict,
    build_path: Path,
    workspace: Path,
    tasks: list[str],
    jobs_dir: Path,
    attempts: int,
    concurrency: int,
) -> list[str]:
    """Mirror vero/src/vero/harbor/backend.py::_command and the baseline runs.

    Notably absent: --agent-timeout-multiplier. Omitting it leaves harbor at its
    default 1.0, which is exactly what the pinned baselines ran at.
    """
    # Same flags vero uses (harbor/backend.py::_source_args): -p for a local task
    # directory, -d for a hub dataset ref.
    source = str(build["task_source"])
    if "@" in source or source.startswith("http"):
        source_args = ["-d", source]
    else:
        source_args = ["-p", str((build_path.parent / source).resolve())]

    command = [
        "uv", "run",
        "--python", str(build.get("harbor_python_version", "3.12")),
        "--no-config", "--no-env-file",
        "--project", str(workspace),
        "--with", build.get("harbor_requirement", "harbor[modal]==0.20.0"),
        "harbor", "run", "--yes",
        *source_args,
        "--agent-import-path", build["agent_import_path"],
        "-e", resolve_param(build.get("environment_name", "modal")),
        "-m", str(build["model"]),
        "-n", str(concurrency),
        "--n-attempts", str(attempts),
        "--jobs-dir", str(jobs_dir),
    ]
    for task in tasks:
        command.extend(["-i", task])
    command.extend(str(a) for a in build.get("extra_harbor_args", []))
    return command


def trial_rewards(round_dir: Path) -> list[float]:
    """Same extraction as runs/recompute.py, so numbers are comparable."""
    rewards = []
    for path in glob.glob(f"{round_dir}/**/result.json", recursive=True):
        if "/verifier/" in path:
            continue
        try:
            data = json.loads(Path(path).read_text())
        except Exception:
            continue
        if "task_name" not in data:
            continue
        verifier = data.get("verifier_result") or {}
        block = verifier.get("rewards")
        reward = block.get("reward") if isinstance(block, dict) else verifier.get("reward")
        if reward is not None:
            rewards.append(float(reward))
    return rewards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--session",
                        help="session.tar.gz, or an extracted session dir")
    source.add_argument(
        "--seed", action="store_true",
        help=(
            "score the benchmark's own seed harness instead of a candidate, to "
            "re-pin baseline_reward. Uses the same path and aggregation as a "
            "candidate rescore, so the two stay comparable."
        ),
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--version", help="candidate sha (default: the shipped one)")
    parser.add_argument("--partition", default="test")
    parser.add_argument("--cases", type=int,
                        help="score only the first N cases (for a cheap smoke)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="independent rounds, pooled (default 3, as the baselines)")
    parser.add_argument("--attempts", type=int, default=1,
                        help="attempts per case within a round (default 1)")
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--output", help="output dir (default: a temp dir)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    build, build_path = load_build(args.benchmark)
    outdir = Path(args.output).resolve() if args.output else Path(
        tempfile.mkdtemp(prefix=f"rescore-{args.benchmark}-"))
    outdir.mkdir(parents=True, exist_ok=True)

    if args.seed:
        # The seed harness lives beside the build config, at the path build.yaml
        # names in agent_repo. Copy it so the run cannot mutate the checkout.
        origin = (build_path.parent / str(build.get("agent_repo", "target"))).resolve()
        if not origin.is_dir():
            sys.exit(f"no seed harness at {origin}")
        workspace = outdir / "seed"
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.copytree(origin, workspace, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".venv", ".git"))
        version = "seed"
        log(f"seed harness from {origin}")
    else:
        session_dir = open_session(args.session, outdir)
        version = shipped_version(
            session_dir, Path(args.session).resolve(), args.version
        )
        workspace = outdir / "candidate"
        if workspace.exists():
            shutil.rmtree(workspace)
        extract_candidate(session_dir, version, workspace)

    partition_file = build["partition_files"][args.partition]
    tasks = json.loads((build_path.parent / partition_file).read_text())
    if args.cases:
        tasks = tasks[: args.cases]
    log(f"{len(tasks)} {args.partition} case(s), {args.rounds} round(s), "
        f"{args.attempts} attempt(s)/case, concurrency {args.concurrency}")

    if "OPENAI_API_KEY" not in os.environ and not args.dry_run:
        sys.exit("OPENAI_API_KEY is not set. Source the run's secrets.env first: "
                 "the target agent talks to the upstream directly here, exactly "
                 "as it did when the pinned baselines were measured.")

    round_means: list[float] = []
    pooled: list[float] = []
    for index in range(args.rounds):
        jobs_dir = outdir / "jobs" / f"round-{index + 1}"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        command = harbor_command(
            build=build, build_path=build_path, workspace=workspace, tasks=tasks,
            jobs_dir=jobs_dir, attempts=args.attempts, concurrency=args.concurrency,
        )
        if args.dry_run:
            print(" ".join(command))
            continue
        log(f"round {index + 1}/{args.rounds} -> {jobs_dir}")
        result = subprocess.run(command, cwd=build_path.parent, check=False)
        if result.returncode != 0:
            log(f"round {index + 1} exited {result.returncode}; scoring what landed")
        rewards = trial_rewards(jobs_dir)
        if rewards:
            mean = sum(rewards) / len(rewards)
            round_means.append(mean)
            pooled += rewards
            log(f"round {index + 1}: n={len(rewards)} mean={mean:.4f}")
        else:
            log(f"round {index + 1}: no scored trials")

    if args.dry_run:
        return 0
    if not pooled:
        log("no scored trials in any round — nothing to report")
        return 1

    reward = sum(pooled) / len(pooled)
    sd = statistics.pstdev(round_means) if len(round_means) >= 2 else 0.0
    pinned = None
    for target in build.get("targets", []):
        if target.get("partition") == args.partition:
            pinned = target.get("baseline_reward")
    print()
    label = "seed harness" if version == "seed" else f"candidate {version[:12]}"
    print(f"  {label}")
    print(f"  {args.partition:15s} n={len(pooled)} reward={reward:.4f} sd={sd:.4f}")
    print(f"  rounds          {' / '.join(f'{m:.3f}' for m in round_means)}")
    if pinned is not None:
        print(f"  pinned baseline {pinned:.4f}   delta={reward - pinned:+.4f}")
    print(f"  artifacts       {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
