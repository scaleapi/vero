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


def load_build(benchmark: str) -> tuple[dict, Path]:
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
    model: str,
    agent: str | None = None,
    setup_timeout_multiplier: float | None = None,
    agent_env: list[str] | None = None,
    extra_requirements: list[str] | None = None,
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
        # Harbor-native harnesses import their framework in the ORCHESTRATOR, not the
        # task container, and harbor does not declare those imports: dspy-rlm dies on
        # ModuleNotFoundError: No module named 'dspy' before the agent ever starts.
        *[arg for req in (extra_requirements or []) for arg in ("--with", req)],
        "harbor", "run", "--yes",
        *source_args,
        # A reference run swaps the seed program for an installed harness
        # (claude-code, opencode, ...) and changes NOTHING else -- same dataset
        # ref, same partition, same rounds, same aggregation -- so the number
        # stays comparable to baseline_reward and to every contestant. The target
        # model stays pinned, so the harness is the only variable.
        *(["--agent", agent] if agent
          else ["--agent-import-path", build["agent_import_path"]]),
        "-e", resolve_param(build.get("environment_name", "modal")),
        "-m", str(model),
        "-n", str(concurrency),
        "--n-attempts", str(attempts),
        "--jobs-dir", str(jobs_dir),
    ]
    # opencode's install (nvm -> node 22 -> npm -g) is far heavier than the seed's
    # zero setup and can exceed harbor's default. swe-atlas already gives its
    # OPTIMIZER agent a 4x multiplier for the same reason.
    if setup_timeout_multiplier:
        command += ["--agent-setup-timeout-multiplier", str(setup_timeout_multiplier)]
    for pair in agent_env or []:
        command += ["--ae", pair]
    for task in tasks:
        command.extend(["-i", task])
    command.extend(str(a) for a in build.get("extra_harbor_args", []))
    return command


# Convention (2026-07-31, adopted on Greptile's PR #75 catch) -- IDENTICAL to
# runs/recompute.py, which produced the pinned baselines. A trial the HARNESS killed
# scores 0: the harness owns its own install, its context management and its step
# budget, so a reference harness that cannot install or cannot finish must pay for it
# exactly as a candidate does at finalization, which zero-fills dead attempts. A trial
# the PLATFORM killed is dropped: a retry cannot score a trial that never ran, and
# zero-filling infra bakes outage luck into the number.
#
# Dropping them instead -- which this script did until 2026-08-02 -- silently inflates
# every reference score by scoring only the trials that survived. It cost 73 zeroes
# across the reference grid, worst on swe-atlas x mini-swe-agent (n=122 of 150).
HARNESS_EXCEPTIONS = {
    "RuntimeError",                # swe-atlas seed: empty-completion fail-fast
    "UnicodeDecodeError",          # terminal-bench seed: undecodable command output
    "AgentTimeoutError",           # wall-clock exhaustion: the step budget is harness-owned
    "BadRequestError",             # gpt-oss 128k overflow: context management is harness-owned
    "NonZeroAgentExitCodeError",   # the harness crashed or never installed its binary
    "AgentSetupTimeoutError",      # the harness owns how long its own install takes
    "AgentAuthenticationError",    # subclasses NonZeroAgentExitCodeError: the CLI reports
                                   # no login, i.e. the harness did not read a credential
                                   # surface we set (goose reads OPENAI_HOST, not _BASE_URL)
    "AdapterParseError",           # dspy's own output parser gave up: harness-owned
}
INFRA_EXCEPTIONS = {
    "RateLimitError",
    "ApiRateLimitError",
    "NetworkConnectionError",
    "VerifierTimeoutError",
    "EnvironmentStartTimeoutError",
    "SandboxFilesystemNotFoundError",
    "AddTestsDirError",            # harbor could not stage the tests: platform, not agent
    "ConnectionError",
    "UnknownApiError",             # harbor's ApiError subclass for a provider error it
                                   # could not classify: upstream, like the two RateLimits
    "RewardFileNotFoundError",     # the verifier ran and produced no reward file
    "CancelledError",              # harbor cancelled the trial (job.py CANCELLED_ERROR_TYPE)
}
# ValueError is deliberately in NEITHER set. Both instances on disk are harbor's
# "ContextVar ... was created in a different Context" orchestration bug, which is
# platform-side, but the name is generic enough that an agent-side ValueError would
# land here too. A rescore that meets one should stop and be looked at, not guess.


def trial_rewards(round_dir: Path) -> list[float]:
    """Same extraction AND the same failure convention as runs/recompute.py."""
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
        exception = (data.get("exception_info") or {}).get("exception_type")
        if reward is not None:
            rewards.append(float(reward))
        elif exception in HARNESS_EXCEPTIONS:
            rewards.append(0.0)  # the harness killed it: price the defect
        elif exception in INFRA_EXCEPTIONS:
            continue             # the platform killed it: drop, do not bake in outage luck
        elif exception:
            message = (data.get("exception_info") or {}).get("exception_message") or ""
            sys.exit(f"unclassified exception {exception!r} in {round_dir}: add it to "
                     "HARNESS_EXCEPTIONS or INFRA_EXCEPTIONS before quoting a number"
                     f"\n  {message.splitlines()[0][:200] if message else '(no message)'}")
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
    parser.add_argument(
        "--model",
        help="Target model to score with, overriding the build's own. Use when "
             "choosing a target for a new benchmark: the pinned baseline is "
             "model-specific, so a model change invalidates it.",
    )
    parser.add_argument("--cases", type=int,
                        help="score only the first N cases (for a cheap smoke)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="independent rounds, pooled (default 3, as the baselines)")
    parser.add_argument("--attempts", type=int, default=1,
                        help="attempts per case within a round (default 1)")
    parser.add_argument("--agent",
                        help="run an INSTALLED harbor harness (claude-code, opencode, "
                             "...) instead of the seed program, at the benchmark's "
                             "pinned model. Measures a SOTA reference, not a bound.")
    parser.add_argument("--harbor-requirement",
                        help="override build.yaml's harbor_requirement for this run "
                             "only. Also relaxes the copied workspace's own harbor pin, "
                             "which build.yaml and target/pyproject.toml must otherwise "
                             "keep in lockstep (see a70c572) or uv cannot resolve.")
    parser.add_argument("--setup-timeout-multiplier", type=float,
                        help="scale harbor's agent-setup timeout (installed harnesses "
                             "with heavy toolchains need this)")
    parser.add_argument("--agent-env", action="append", metavar="KEY=VALUE",
                        default=[],
                        help="extra env var for the agent (repeatable). goose reads "
                             "OPENAI_HOST/OPENAI_BASE_PATH rather than OPENAI_BASE_URL, "
                             "so it needs the proxy pointed at explicitly.")
    parser.add_argument("--with-requirement", action="append", metavar="SPEC",
                        default=[], dest="extra_requirements",
                        help="extra package for the orchestrator uv env (repeatable). "
                             "Harbor-native harnesses import their framework here and "
                             "harbor does not declare it -- dspy-rlm needs 'dspy'.")
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--output", help="output dir (default: a temp dir)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    build, build_path = load_build(args.benchmark)
    if args.harbor_requirement:
        build["harbor_requirement"] = args.harbor_requirement
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
        if args.harbor_requirement:
            pyproject = workspace / "pyproject.toml"
            if pyproject.is_file():
                import re as _re
                text = pyproject.read_text()
                relaxed = _re.sub(r'"harbor==[^"]+"', '"harbor"', text)
                if relaxed != text:
                    pyproject.write_text(relaxed)
                    log("relaxed the copied workspace's harbor pin so the override resolves")
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

    # Some tasks reference the base URL under litellm's alias rather than
    # OPENAI_BASE_URL -- swe-atlas-qna's rubric judge declares
    # `EVAL_BASE_URL = "${OPENAI_API_BASE}"` in [verifier.env], and harbor aborts
    # the whole job with "Missing Environment Variables" before running a single
    # trial if it is unset. Mirror it so a benchmark's own judge can start.
    if os.environ.get("OPENAI_BASE_URL") and not os.environ.get("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"]
        log("mirrored OPENAI_BASE_URL -> OPENAI_API_BASE for task verifier env")

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
            model=args.model or build["model"], agent=args.agent,
            setup_timeout_multiplier=args.setup_timeout_multiplier,
            agent_env=args.agent_env,
            extra_requirements=args.extra_requirements,
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
