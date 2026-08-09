#!/usr/bin/env python3
"""Cross-cell stats over extract_candidates.py's JSON. Replaces the six blind
lens agents from the skill's first draft: correlations, knob-touch counts, and a
keyword sweep, computed exactly instead of eyeballed across 16 per-cell reports.

    python3 cross_cell_stats.py --json /tmp/<benchmark>-candidates.json

Keyword hits are candidates FOR VERIFICATION, not a count to quote directly --
step 3 still has to check each one against the real diff before it ships.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st


def corr(a: list[float], b: list[float]) -> float:
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    rows = [r for r in json.load(open(args.json)) if r["reportable"]]
    cands = lambda r: [c for c in r["candidates"] if not c["is_seed"]]

    print(f"{len(rows)} reportable cells\n")
    print(f"{'cell':34} {'reward':>7} {'n':>3} {'shipPos':>8} {'ship +/-':>10}")
    for r in sorted(rows, key=lambda x: -x["reward"]):
        cs = cands(r)
        sh = next((c for c in r["candidates"] if c["is_shipped"]), None)
        pos = sh["position"] if sh else None
        print(f"{r['cell']:34} {r['reward']:.4f} {len(cs):3} "
              f"{(str(pos) + '/' + str(len(cs))) if pos is not None else '-':>8} "
              f"{('+' + str(sh['insertions']) + '/-' + str(sh['deletions'])) if sh else '-':>10}")

    n = [len(cands(r)) for r in rows]
    rew = [r["reward"] for r in rows]
    shipsize = [next(c for c in r["candidates"] if c["is_shipped"])["insertions"]
                + next(c for c in r["candidates"] if c["is_shipped"])["deletions"]
                for r in rows]
    print(f"\ncandidates: total {sum(n)}, median {st.median(n)}, range {min(n)}-{max(n)}")
    print(f"corr(candidate count, reward)  = {corr(n, rew):+.3f}  (n={len(rows)} cells)")
    print(f"corr(shipped diff size, reward) = {corr(shipsize, rew):+.3f}")

    print("\n=== keyword sweep (candidates for verification, NOT a verified count) ===")
    allc = [(r["cell"], c) for r in rows for c in cands(r)]
    print(f"total candidates: {len(allc)}")
    for label, pat in [
        ("revert-ish", r"\brevert|\bback out|\brestore\b|\bundo\b"),
        ("cites a number", r"\d+\s*(of|/)\s*\d+|0\.\d{2,}|\d+%"),
        ("reasoning_effort", r"reasoning[_ ]effort"),
        ("MAX_TURNS", r"max[_ ]turns"),
        ("timeout", r"timeout"),
        ("retry", r"retr(y|ies|ying)"),
    ]:
        hits = [(cell, c["id"], c["subject"][:52]) for cell, c in allc
                if re.search(pat, c["subject"] + " " + c["body"], re.I)]
        cells = len({h[0] for h in hits})
        print(f"\n{label}: {len(hits)} candidates across {cells} cells")
        for h in hits[:6]:
            print(f"   {h[0][:28]:28} {h[1]} {h[2]}")
        if len(hits) > 6:
            print(f"   ... and {len(hits) - 6} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
