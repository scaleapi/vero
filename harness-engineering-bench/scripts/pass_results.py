#!/usr/bin/env python3
"""Extract one grid pass's results, refusing to report a damaged cell as a number.

    python3 scripts/pass_results.py runs/officeqa r1
    python3 scripts/pass_results.py runs/browsecomp-plus r1 --baseline 0.4619

A cell's reward is only a measurement when the verifier finished with an empty
`errors` block. On 2026-07-29 an officeqa cell reported reward 0.0 with
`shipped: true` because held-out scoring was aborted by a misclassified provider
rate limit -- indistinguishable, in a results table, from a candidate that
genuinely scored nothing. This script exists so that can never be read as data
again: any cell whose verifier recorded an error is printed as INVALID with the
reason, and is excluded from the summary statistics.

It also counts SEARCH damage, which the reward alone cannot show. A terminated
evaluation mid-search tells the optimizer "the run cannot continue and must not
be retried", so it stops exploring and ships early. Such a cell can complete with
a plausible-looking reward while measuring a truncated search, and its candidate
is NOT recoverable by rescoring -- only by re-running. W&B's `diagnostics` field
is last-value and cannot count these, so they are counted here from the agent
log, which records each terminated evaluation.

Caveat on in-flight runs: the agent's transcript is tee'd inside the sandbox and
only syncs when artifacts are collected at the end, so a cell still running always
reports 0 kills. That is missing data, not a clean bill of health. For a live view
of the same condition use W&B's `diagnostics` field, which shows whether at least
one kill has happened (though not how many).

Exit status is 1 if any cell is INVALID or shows search damage, so this can gate
a publish step.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
import tarfile

# The verifier's own record of why a reward is not trustworthy.
ERRORS_BLOCK = re.compile(r'"errors":\s*\{(.*?)\}', re.S)
SHIPPED = re.compile(r'"shipped":\s*(\w+)')
# The session database is the authoritative record of every evaluation the
# optimizer ran: its partition, its principal (agent = the optimizer's own
# search, admin = held-out scoring), its status, and its diagnostics. Counting
# from the agent transcript instead was unreliable in both directions -- it
# over-reported by matching the same event twice and under-reported entirely on
# some cells -- so it is not used here.
SESSION_DB = "session/database.json"


def session_evaluations(job: str) -> list[dict] | None:
    """Every evaluation record from one job's exported session, or None.

    Scoped to a single job directory, not the whole cell. `infrastructure_max_attempts`
    can leave several jobs behind, each with its own archive, and the reward is read
    from one specific job -- so reading evidence from any other would describe a
    different run than the number it is reported beside. An earlier abandoned attempt
    that saw 2 of 37 cases would report near-zero kills for a cell whose real
    attempt was cut to pieces. `glob` returns filesystem order, so which one you got
    was not even stable between invocations.
    """
    archives = sorted(glob.glob(f"{job}/task__*/verifier/session.tar.gz"))
    if not archives:
        return None
    try:
        with tarfile.open(archives[-1], "r:gz") as tar:
            handle = tar.extractfile(SESSION_DB)
            if handle is None:
                return None
            database = json.load(handle)
    except (tarfile.TarError, OSError, ValueError, KeyError):
        return None
    records = []
    for record in (database.get("evaluations") or {}).values():
        report = record.get("report") or {}
        request = record.get("request") or {}
        records.append(
            {
                "partition": ((request.get("evaluation_set") or {}).get("partition")) or "?",
                "principal": record.get("principal"),
                "status": report.get("status"),
                "diagnostics": [d.get("code") for d in (report.get("diagnostics") or [])],
                "created_at": record.get("created_at") or "",
            }
        )
    return sorted(records, key=lambda r: r["created_at"])


def cell_result(directory: str) -> dict:
    """Reward, validity and search damage for one contestant directory."""
    out = {
        "reward": None,
        "valid": True,
        "reason": "",
        "shipped": None,
        "kills": 0,
        "kill_codes": set(),
        "running": False,
        "success": {},
        "killed": {},
        "other_invalid": {},
        "recovered": set(),
        "evidence_unknown": False,
    }
    results = sorted(glob.glob(f"{directory}/jobs/*/result.json"))
    if not results:
        out["valid"] = False
        out["reason"] = "no result.json (run never produced a trial)"
        return out

    # Everything below is read from this one job, so the reward, the verifier's
    # error block, and the search evidence all describe the same run.
    job = os.path.dirname(results[-1])
    record = json.load(open(results[-1]))
    stats = record.get("stats") or {}
    if record.get("finished_at") is None:
        # Still in flight. Reporting this as INVALID would be worse than useless:
        # it invites writing off a healthy run that simply has not finished.
        out["running"] = True
        out["reason"] = "still running"
        return out
    if not stats.get("n_completed_trials"):
        out["valid"] = False
        out["reason"] = (
            f"0 completed trials ({stats.get('n_errored_trials', 0)} errored)"
        )
    for evaluation in (stats.get("evals") or {}).values():
        metrics = (evaluation.get("metrics") or [{}])[0]
        if metrics.get("mean") is not None:
            out["reward"] = metrics["mean"]

    # Same job as the reward, for the same reason: this loop's last write wins, so
    # ranging over every job could invalidate a cell for an error another attempt
    # raised, or overwrite a real error with a clean earlier one.
    for stdout in sorted(glob.glob(f"{job}/task__*/verifier/*stdout*")):
        text = open(stdout, errors="replace").read()
        shipped = SHIPPED.search(text)
        if shipped:
            out["shipped"] = shipped.group(1) == "true"
        for match in ERRORS_BLOCK.finditer(text):
            body = match.group(1).strip()
            if body:  # a non-empty errors block invalidates the reward
                out["valid"] = False
                out["reason"] = re.sub(r"\s+", " ", body)[:150]

    # Search evidence and damage, from the session database.
    records = session_evaluations(job)
    if records is None:
        out["evidence_unknown"] = True
        return out
    for record in records:
        if record["principal"] == "admin":
            continue  # held-out scoring, judged via the verifier errors block
        partition = record["partition"]
        if record["status"] == "success":
            out["success"][partition] = out["success"].get(partition, 0) + 1
        elif "inference_budget_exhausted" in record["diagnostics"]:
            out["killed"][partition] = out["killed"].get(partition, 0) + 1
            out["kills"] += 1
            out["kill_codes"].add("inference_budget_exhausted")
        else:
            out["other_invalid"][partition] = out["other_invalid"].get(partition, 0) + 1
    # Recovery: after the last kill on a partition, did a successful evaluation on
    # that partition still follow? Where it did, the kill cost wall-clock rather
    # than information, which is the difference between a cell needing a re-run
    # and a cell that merely ran slower.
    for partition in list(out["killed"]):
        ordered = [r for r in records if r["partition"] == partition
                   and r["principal"] != "admin"]
        last_kill = max(
            (i for i, r in enumerate(ordered)
             if "inference_budget_exhausted" in r["diagnostics"]),
            default=None,
        )
        if last_kill is not None and any(
            r["status"] == "success" for r in ordered[last_kill + 1:]
        ):
            out["recovered"].add(partition)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pass_dir", help="e.g. runs/officeqa")
    parser.add_argument("suffix", nargs="?", default="r1", help="cell suffix, default r1")
    parser.add_argument("--baseline", type=float, help="pinned baseline_reward for deltas")
    parser.add_argument(
        "--thin-validation",
        type=int,
        default=2,
        help="flag a cell whose successful validation evaluations are <= this "
        "(validation is the selection partition, so it is the evidence the "
        "optimizer chose its candidate on); default 2",
    )
    args = parser.parse_args()

    cells = sorted(
        d for d in glob.glob(f"{args.pass_dir}/*{args.suffix}")
        if os.path.isdir(d) and "shed" not in d and "infrafail" not in d
    )
    if not cells:
        print(f"no cells matching {args.pass_dir}/*{args.suffix}", file=sys.stderr)
        return 2

    rows = [(os.path.basename(d).removesuffix(f"-{args.suffix}"), cell_result(d)) for d in cells]
    width = max(len(name) for name, _ in rows)

    print(
        f"{'contestant':{width}}  {'reward':>8} {'delta':>8}  "
        f"{'dev':>4} {'val':>7} {'killed':>6}  verdict"
    )
    clean = []
    for name, r in sorted(rows, key=lambda kv: -(kv[1]["reward"] or -1)):
        reward = f"{r['reward']:.4f}" if r["reward"] is not None else "-"
        delta = (
            f"{r['reward'] - args.baseline:+.3f}"
            if (args.baseline and r["reward"] is not None and r["valid"])
            else "-"
        )
        dev = r["success"].get("development", 0)
        val = r["success"].get("validation", 0)
        killed = r["kills"]
        recovered = ", ".join(sorted(r["recovered"])) if r["recovered"] else ""

        if r["running"]:
            verdict = "running"
        elif not r["valid"]:
            # The held-out score itself did not survive, so the reward is an
            # artefact. Rescoring recovers it only if the SEARCH was sound.
            verdict = f"INVALID: {r['reason']}"
            if r["shipped"] and not killed:
                verdict += "  [shipped, search intact -- RESCORABLE]"
            elif r["shipped"]:
                verdict += f"  [shipped but {killed} search evaluation(s) lost -- RE-RUN]"
        elif r["evidence_unknown"]:
            verdict = "ok (reward valid; no session database, search evidence unknown)"
            clean.append(r["reward"])
        elif val <= args.thin_validation:
            # The measured damage was never corrupted numbers -- it was thin
            # selection evidence. Validation is the selection partition, so this
            # count is how much the optimizer had to choose its candidate on.
            verdict = (
                f"THIN SELECTION: {val} successful validation evaluation(s)"
                + (f", {killed} killed" if killed else "")
            )
        elif killed:
            verdict = f"ok, {killed} killed but recovered on {recovered}" if recovered \
                else f"ok, {killed} killed and not recovered"
            clean.append(r["reward"])
        else:
            verdict = "ok"
            clean.append(r["reward"])
        print(
            f"{name:{width}}  {reward:>8} {delta:>8}  {dev:>4} {val:>7} {killed:>6}  {verdict}"
        )

    finished = [r for _, r in rows if not r["running"]]
    running = len(rows) - len(finished)
    print(f"\n{len(clean)}/{len(finished)} finished cells are clean measurements"
          + (f"; {running} still running" if running else ""))
    if len(clean) >= 2:
        print(f"clean mean {statistics.mean(clean):.4f}  sd {statistics.stdev(clean):.4f}")
    if args.baseline and clean:
        beat = sum(1 for value in clean if value > args.baseline)
        print(f"{beat}/{len(clean)} clean cells beat the {args.baseline} baseline")
    # Non-zero when anything finished is unusable, so this can gate a publish
    # step. Runs still in flight are not a failure.
    return 0 if len(clean) == len(finished) else 1


if __name__ == "__main__":
    sys.exit(main())
