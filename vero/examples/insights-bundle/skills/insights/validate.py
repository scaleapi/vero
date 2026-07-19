#!/usr/bin/env python3
"""validate.py — enforce the insights finding contract.

Reproduces the gate IG's `submit_finding` applies: a finding is rejected unless a
non-empty cohort of affected trace_ids was persisted, those ids exist in the
corpus, and the evidence is real (verbatim quotes from distinct traces). Run it
before a finding is considered submitted:

    python validate.py findings.json index.tsv

index.tsv is the corpus index built per SKILL.md: tab-separated, one row per
trial, columns ``trace_id <TAB> case_id <TAB> eval <TAB> reward <TAB> trace_path``.
Only column 1 (trace_id) and column 5 (trace_path, used for the best-effort
verbatim-quote check) are read. Exit code is non-zero if any finding fails, and
every problem is printed so the agent can fix and re-run.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MIN_EVIDENCE = 8
VALID_STATUS = {"confirmed", "refuted", "inconclusive"}
COHORT_CORPUS_RATIO_WARN = 0.25


def load_corpus_index(index_path: Path) -> dict[str, str]:
    """Return {trace_id: trace_path} from the corpus index (col1 -> col5)."""
    index: dict[str, str] = {}
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        trace_id = cols[0].strip()
        path = cols[4].strip() if len(cols) >= 5 else ""
        index[trace_id] = path
    return index


_TRIAL_TEXT_CACHE: dict[str, str] = {}


def _canon(text: str) -> str:
    """Lowercase, alphanumeric-only. Collapses JSON escaping/spacing/punctuation
    differences (``{"answer":"X"}`` vs ``\\"answer\\": \\"X\\"``) so a genuine
    excerpt matches while paraphrase/fabrication (different words) does not."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _quote_is_verbatim(quote: str, trace_path: str) -> bool | None:
    """True/False if the quote appears in the trial; None if unknown.

    Best-effort. A legitimate excerpt may come from the trace JSONL, the agent's
    ``answer.txt``, or the verifier's ``test-stdout.txt`` — and a tool-call
    argument is stored JSON-escaped — so the whole trial directory is searched
    and both sides are canonicalized to alphanumerics before comparing. Returns
    None when nothing can be read (caller neither warns nor endorses).
    """
    if not trace_path:
        return None
    tp = Path(trace_path)
    # Guard: a bogus/short path whose .parent.parent is a high-level dir (e.g. "/")
    # would make rglob walk the whole filesystem. Only proceed for a real trace.
    if not tp.is_file():
        return None
    trial_dir = tp.parent.parent  # trial dir = parent of the agent/ dir
    if not trial_dir.is_dir() or trial_dir == trial_dir.parent:
        return None
    key = str(trial_dir)
    text = _TRIAL_TEXT_CACHE.get(key)
    if text is None:
        parts: list[str] = []
        try:
            files = [p for p in trial_dir.rglob("*") if p.is_file()]
        except OSError:
            files = []
        for p in files:
            try:
                if p.stat().st_size > 5_000_000:
                    continue
                parts.append(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
        text = _canon(" ".join(parts))
        _TRIAL_TEXT_CACHE[key] = text
    if not text:
        return None
    needle = _canon(quote)
    return bool(needle) and needle in text


def validate_finding(
    idx: int, f: dict, corpus_index: dict[str, str], base: Path
) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one finding."""
    corpus_ids = corpus_index.keys()
    errs: list[str] = []
    warns: list[str] = []
    name = f.get("hypothesis_name") or f"#{idx}"

    # --- required scalar fields ---
    for field in ("hypothesis_name", "status", "summary", "prevalence"):
        if not f.get(field):
            errs.append(f"[{name}] missing required field: {field}")
    status = f.get("status")
    if status and status not in VALID_STATUS:
        errs.append(f"[{name}] status '{status}' not in {sorted(VALID_STATUS)}")

    # --- cohort: the IG "no saved cohort -> reject" rule ---
    cohort_file = f.get("cohort_file")
    cohort_ids: list[str] = []
    if not cohort_file:
        errs.append(
            f"[{name}] no cohort_file: you must persist the affected trace_ids "
            f"(cohorts/<slug>.txt) before submitting — a finding without a cohort "
            f"is rejected"
        )
    else:
        cpath = (base / cohort_file).resolve()
        if not cpath.exists():
            errs.append(f"[{name}] cohort_file does not exist: {cohort_file}")
        else:
            cohort_ids = [
                ln.strip() for ln in cpath.read_text().splitlines() if ln.strip()
            ]
            if not cohort_ids:
                errs.append(f"[{name}] cohort_file is empty: {cohort_file}")
            unknown = [t for t in cohort_ids if t not in corpus_ids]
            if unknown:
                errs.append(
                    f"[{name}] cohort has {len(unknown)} trace_id(s) not in the "
                    f"corpus (e.g. {unknown[:3]}) — cohort ids must be real"
                )
            if corpus_ids and len(cohort_ids) / len(corpus_ids) > COHORT_CORPUS_RATIO_WARN:
                warns.append(
                    f"[{name}] cohort is {len(cohort_ids)}/{len(corpus_ids)} "
                    f"(>{COHORT_CORPUS_RATIO_WARN:.0%}) of the corpus — confirm this "
                    f"is a real population-level pattern, not the analyzed pool"
                )

    # --- evidence: verbatim quotes from distinct traces ---
    evidence = f.get("evidence") or []
    if len(evidence) < MIN_EVIDENCE:
        errs.append(
            f"[{name}] {len(evidence)} evidence item(s); need >= {MIN_EVIDENCE}"
        )
    ev_trace_ids: list[str] = []
    for j, ev in enumerate(evidence):
        loc = f"[{name}] evidence[{j}]"
        tid = ev.get("trace_id")
        quote = (ev.get("quote") or "").strip()
        if not tid:
            errs.append(f"{loc}: missing trace_id")
        else:
            ev_trace_ids.append(tid)
            if corpus_ids and tid not in corpus_ids:
                errs.append(f"{loc}: trace_id '{tid}' not in the corpus")
            elif quote:
                verbatim = _quote_is_verbatim(quote, corpus_index.get(tid, ""))
                if verbatim is False:
                    warns.append(
                        f"{loc}: quote not found verbatim in trace '{tid}' — "
                        f"use a literal excerpt, not a paraphrase"
                    )
        if not quote:
            errs.append(f"{loc}: missing verbatim quote")
        if len((ev.get("explanation") or "").strip()) < 40:
            errs.append(
                f"{loc}: explanation too thin — name the concrete mechanism in "
                f"this trace (tool/exception/turn), don't restate the finding"
            )
    # Evidence must span >= MIN_EVIDENCE distinct traces (counterexamples, which
    # legitimately sit outside the cohort, still count as distinct traces).
    distinct = len(set(ev_trace_ids))
    if len(evidence) >= MIN_EVIDENCE and distinct < MIN_EVIDENCE:
        errs.append(
            f"[{name}] evidence spans only {distinct} distinct trace(s); need "
            f">= {MIN_EVIDENCE} — quote from different traces, not the same 2-3"
        )

    return errs, warns


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python validate.py findings.json index.tsv", file=sys.stderr)
        return 2
    findings_path = Path(argv[1])
    index_path = Path(argv[2])
    base = findings_path.resolve().parent

    try:
        doc = json.loads(findings_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"cannot read findings.json: {e}", file=sys.stderr)
        return 2
    corpus_index = load_corpus_index(index_path) if index_path.exists() else {}
    if not corpus_index:
        print(f"warning: no corpus ids loaded from {index_path}", file=sys.stderr)

    findings = doc.get("findings") or []
    if not findings:
        print("REJECTED: findings.json has no findings", file=sys.stderr)
        return 1

    all_errs: list[str] = []
    all_warns: list[str] = []
    for i, f in enumerate(findings):
        e, w = validate_finding(i, f, corpus_index, base)
        all_errs += e
        all_warns += w

    for w in all_warns:
        print(f"WARN  {w}")
    for e in all_errs:
        print(f"ERROR {e}")

    if all_errs:
        print(f"\nREJECTED: {len(all_errs)} error(s) across {len(findings)} finding(s).")
        return 1
    print(
        f"OK: {len(findings)} finding(s) validated"
        + (f" ({len(all_warns)} warning(s))" if all_warns else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
