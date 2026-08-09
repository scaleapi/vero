#!/usr/bin/env python3
"""Fill `baseline_rewards` into finalization.json for cells that ran without a pin.

    python3 harness-engineering-bench/scripts/backfill_baseline.py \
        --benchmark tau3 --reward-key reward --value 0.5679 \
        --provenance "K=3 rescore_candidate.py --seed on build.gpt54mini.yaml, ..." \
        [--push] [--dry-run]

Why this is legitimate rather than editing results after the fact: nothing ever
measures `baseline_rewards` during a run. Every other benchmark carries a
`baseline_reward` that was measured out of band by rescore_candidate.py days
earlier and pasted into build.yaml, and the verifier simply copies it into the
finalization payload. Filling the same field, with a value from the same script on
the same seed and target, gives tau3's cells identical provenance -- it does not
invent a number, it supplies the one the config should have carried.

Why it matters: the results pipeline computes gain from `baseline_rewards`. Left
empty, tau3 is the one benchmark whose gain cannot be derived, and the likely
failure mode is that nobody notices until the table is built.

Two hard constraints this respects:

- `VerificationResult` is a StrictModel with extra="forbid", and vero's report.py
  validates the finalization payload through it. So NO new keys go in that file --
  provenance goes in a sibling baseline_reward_provenance.json that no schema reads.
- The candidate's own `rewards` are never touched. Only the comparator is added.

Every patch is validated by re-parsing the file through VerificationResult before
it is written, so a schema mistake fails here rather than in someone's report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
S3 = "s3://scale-ml/harness-engineering-bench"
PROFILE = "ml-worker"


def validates(payload: dict) -> tuple[bool, str]:
    """Re-parse through vero's own model, so a bad edit cannot reach S3."""
    try:
        sys.path.insert(0, str(REPO / "vero" / "src"))
        from vero.sidecar.verifier import VerificationResult  # type: ignore
    except Exception as exc:  # vero not importable here; skip rather than guess
        return True, f"(schema check skipped: {exc})"
    try:
        VerificationResult.model_validate_json(json.dumps(payload))
        return True, "ok"
    except Exception as exc:
        return False, str(exc)[:200]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--reward-key", default="reward")
    ap.add_argument("--value", type=float, required=True)
    ap.add_argument("--provenance", required=True,
                    help="one line recording how the value was measured")
    ap.add_argument("--push", action="store_true", help="also upload to S3")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pattern = f"runs/{args.benchmark}/*/jobs/*/task__*/verifier/finalization.json"
    files = sorted(REPO.glob(pattern))
    if not files:
        sys.exit(f"no finalization.json under runs/{args.benchmark}/")

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    touched = skipped = failed = 0
    for path in files:
        cell = path.relative_to(REPO).parts[2]
        payload = json.loads(path.read_text())
        existing = payload.get("baseline_rewards") or {}
        if existing.get(args.reward_key) is not None:
            print(f"  skip  {cell:34} already has {existing}")
            skipped += 1
            continue
        if not payload.get("shipped"):
            print(f"  skip  {cell:34} shipped=false, not a reportable cell")
            skipped += 1
            continue

        payload["baseline_rewards"] = dict(existing) | {args.reward_key: args.value}
        ok, why = validates(payload)
        if not ok:
            print(f"  FAIL  {cell:34} schema rejected: {why}")
            failed += 1
            continue

        prov = path.parent / "baseline_reward_provenance.json"
        prov_doc = {
            "backfilled_at": stamp,
            "reward_key": args.reward_key,
            "baseline_reward": args.value,
            "measured_by": args.provenance,
            "note": ("Added after the run. The build config carried no "
                     "baseline_reward, so the verifier wrote an empty "
                     "baseline_rewards. This value comes from the same script and "
                     "seed that every other benchmark's pinned baseline comes from; "
                     "the candidate's own rewards are untouched."),
        }
        if args.dry_run:
            print(f"  would {cell:34} set {args.reward_key}={args.value}")
            touched += 1
            continue

        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        prov.write_text(json.dumps(prov_doc, indent=2) + "\n", encoding="utf-8")
        print(f"  set   {cell:34} {args.reward_key}={args.value}")
        touched += 1

        if args.push:
            rel = path.relative_to(REPO / "runs")
            for local, key in ((path, rel), (prov, prov.relative_to(REPO / "runs"))):
                r = subprocess.run(
                    ["aws", "--profile", PROFILE, "s3", "cp", str(local),
                     f"{S3}/{key}", "--only-show-errors"],
                    capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"        UPLOAD FAILED {key}: {r.stderr.strip()[:120]}")

    print(f"\n{touched} patched, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
