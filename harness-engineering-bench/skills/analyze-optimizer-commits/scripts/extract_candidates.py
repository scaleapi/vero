#!/usr/bin/env python3
"""Extract every optimizer candidate from a benchmark's cells as structured facts.

    python3 extract_candidates.py --benchmark tau3                 # -> stdout table
    python3 extract_candidates.py --benchmark tau3 --json out.json  # -> machine readable
    python3 extract_candidates.py --runs-dir /path/to/runs --benchmark gaia

Why a script rather than asking an agent to read the repos: counts are the part of
this analysis most likely to be wrong and least likely to be checked. "15% of
candidates were reverts" is a claim a reader will trust and nobody will recompute.
An agent counting commits by eye across 12 cells will miscount, and the error is
invisible in the final prose. So every number that reaches the write-up comes from
here, and agents are asked to interpret rather than to tally.

What it reads, per cell:
  verifier/finalization.json      the shipped candidate, its reward, validity fields
  verifier/session.tar.gz         candidates/repository.git -- every candidate commit
                                  evaluations/*/evaluation.json -- per-candidate scores

What it deliberately does NOT do: classify behaviour. Whether a commit is a
"revert" or "inert" or "structural" is a judgement that depends on reading the diff,
and a regex on the commit message gets it wrong often enough to matter (a commit
saying "revert speculative KB tweak" also adds 19 lines of new logic). The script
surfaces the evidence -- message, diffstat, files touched, scores before and after --
and leaves the call to the analysis, which has to cite what it saw.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "--git-dir", str(repo), *args],
                         capture_output=True, text=True)
    return out.stdout


def extract_session(tgz: Path, dest: Path) -> Path | None:
    """Pull just the candidate repo and evaluation records out of a session archive."""
    try:
        with tarfile.open(tgz) as tar:
            wanted = [m for m in tar.getmembers()
                      if "/candidates/repository.git/" in m.name
                      or m.name.endswith("/evaluation.json")]
            if not wanted:
                return None
            # filter= landed in 3.12 and is a TypeError before that. These
            # archives are our own verifier's output, so the guard is about
            # running on the box's 3.10 rather than about untrusted input.
            if sys.version_info >= (3, 12):
                tar.extractall(dest, members=wanted, filter="data")
            else:
                tar.extractall(dest, members=wanted)
    except Exception as exc:
        print(f"    ! could not read {tgz}: {exc}", file=sys.stderr)
        return None
    hits = list(dest.glob("**/candidates/repository.git"))
    return hits[0] if hits else None


def per_candidate_scores(session_root: Path) -> dict[str, dict[str, float]]:
    """candidate id -> {partition: score}, from the sidecar's evaluation records."""
    scores: dict[str, dict[str, float]] = {}
    for path in session_root.glob("**/evaluations/*/evaluation.json"):
        try:
            doc = json.loads(path.read_text())
        except Exception:
            continue
        request, report = doc.get("request") or {}, doc.get("report") or {}
        cand = ((request.get("candidate") or {}).get("id") or "")[:12]
        part = (request.get("evaluation_set") or {}).get("partition")
        score = (report.get("metrics") or {}).get("score")
        if cand and part and score is not None:
            scores.setdefault(cand, {})[part] = score
    return scores


def read_cell(cell_dir: Path, tmp: Path) -> dict | None:
    finals = sorted(cell_dir.glob("jobs/*/task__*/verifier/finalization.json"))
    if not finals:
        return None
    final = json.loads(finals[-1].read_text())
    metrics = (final.get("reward_metrics") or {}).get("reward", {}) or {}
    tgz = finals[-1].parent / "session.tar.gz"

    row = {
        "cell": cell_dir.name,
        "benchmark": cell_dir.parent.name,
        "shipped": final.get("shipped"),
        "reward": (final.get("rewards") or {}).get("reward"),
        "baseline_reward": (final.get("baseline_rewards") or {}).get("reward"),
        "error_rate": metrics.get("error_rate"),
        "total_tokens": metrics.get("inference_total_tokens"),
        "mean_case_wall_seconds": metrics.get("mean_case_wall_seconds"),
        "shipped_candidate_id": ((final.get("candidate") or {}).get("id") or "")[:12],
        "shipped_candidate_desc": (final.get("candidate") or {}).get("description", ""),
        "candidates": [],
    }
    # A cell is only reportable if it shipped, scored every case, and metered
    # tokens. swe-atlas produced cells reporting shipped/error_rate 0.0 while every
    # case had been dropped by infrastructure and no tokens were spent, so the token
    # check is what distinguishes a real run from a hollow one.
    row["reportable"] = bool(
        row["shipped"] and row["error_rate"] in (0.0, 0) and row["total_tokens"])

    if not tgz.is_file():
        return row
    dest = tmp / cell_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    repo = extract_session(tgz, dest)
    if repo is None:
        return row
    scores = per_candidate_scores(dest)

    log = [l.split("\t", 2) for l in
           git(repo, "log", "--all", "--format=%H\t%at\t%s").splitlines() if l.strip()]
    log.reverse()  # oldest first: the seed is index 0
    for position, parts in enumerate(log):
        sha, when, subject = (parts + ["", "", ""])[:3]
        short = sha[:12]
        shortstat = git(repo, "show", "--shortstat", "--format=", sha).strip().splitlines()
        stat = shortstat[-1].strip() if shortstat else ""
        files = [f for f in git(repo, "show", "--name-only", "--format=", sha).splitlines()
                 if f.strip() and "__pycache__" not in f]
        ins = int(m.group(1)) if (m := re.search(r"(\d+) insertion", stat)) else 0
        dele = int(m.group(1)) if (m := re.search(r"(\d+) deletion", stat)) else 0
        body = git(repo, "show", "--format=%b", "--no-patch", sha).strip()
        row["candidates"].append({
            "position": position,           # 0 = seed
            "id": short,
            "subject": subject,
            "body": body[:2000],            # optimizers explain their reasoning here
            "insertions": ins,
            "deletions": dele,
            "files": files,
            "is_seed": position == 0,
            "is_shipped": short == row["shipped_candidate_id"],
            "scores": scores.get(short, {}),
        })
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--runs-dir", default=None,
                    help="defaults to <repo>/runs")
    ap.add_argument("--json", default=None, help="write structured output here")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    runs = Path(args.runs_dir) if args.runs_dir else repo_root / "runs"
    bench_dir = runs / args.benchmark
    if not bench_dir.is_dir():
        sys.exit(f"no such benchmark directory: {bench_dir}")

    rows = []
    with tempfile.TemporaryDirectory(prefix="optcommits-") as tmpname:
        tmp = Path(tmpname)
        for cell in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
            row = read_cell(cell, tmp)
            if row:
                rows.append(row)

    if not rows:
        sys.exit(f"no cells with a finalization.json under {bench_dir}")

    reportable = [r for r in rows if r["reportable"]]
    # Exclude the seed from every count: it is position 0 and it is what all
    # cells started from, so counting it inflates each denominator by one.
    total_cands = sum(max(len(r["candidates"]) - 1, 0) for r in reportable)
    print(f"benchmark {args.benchmark}: {len(rows)} cells, "
          f"{len(reportable)} reportable, {total_cands} candidates (seed excluded)\n")
    print(f"{'cell':34} {'reward':>7} {'cands':>5} {'shipped pos':>11}  shipped subject")
    for r in rows:
        flag = "" if r["reportable"] else "  [NOT REPORTABLE]"
        pos = next((c["position"] for c in r["candidates"] if c["is_shipped"]), None)
        n = max(len(r["candidates"]) - 1, 0)
        rew = f"{r['reward']:.4f}" if isinstance(r["reward"], (int, float)) else "-"
        print(f"{r['cell'][:34]:34} {rew:>7} {n:>5} "
              f"{(str(pos)+' of '+str(n)) if pos is not None else '-':>11}  "
              f"{r['shipped_candidate_desc'][:44]}{flag}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
