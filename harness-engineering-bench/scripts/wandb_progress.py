#!/usr/bin/env python3
"""Live evaluation progress for in-flight grid cells, read from W&B.

    python3 scripts/wandb_progress.py officeqa r2

A run's local artefacts are useless while it is in flight: `agent/` and
`verifier/` are written only at the end, and `run.log` holds nothing but the
outer harbor spinner. W&B is the only mid-run view of whether the optimizer is
actually evaluating candidates or merely burning clock.

Reads credentials from a gitignored `*.secrets.env` in vero/ and prints none of
them.

Scope names matter and must not be collapsed. Gateway metrics are logged under
paths like `inference/<scope>/<metric>`; keying on the last path segment alone
makes `evaluation` and `finalization` overwrite each other, which previously
produced a badly wrong error-rate figure. Key on the full path, and report the
scopes separately -- a clean search alongside a throttled finalization is a
different situation from the reverse, and a single total hides which you have.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PKG = os.path.join(REPO, "vero")
PROJECT = "harness-engineering-bench"


def load_credentials(filename="heb.secrets.env"):
    """W&B settings out of a gitignored env file.

    ``WANDB_ENTITY`` is read here rather than left to the account default: if the
    file names an entity and we ignore it, the query silently targets whatever
    org the local default points at and reports "no runs matching", which reads
    as "nothing has started yet" rather than "you asked the wrong org".
    """
    path = os.path.join(PKG, filename)
    if not os.path.exists(path):
        sys.exit(f"no credential file: vero/{filename}")
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            match = re.match(
                r"\s*(WANDB_API_KEY|WANDB_BASE_URL|WANDB_ENTITY)\s*=\s*(.+)", line
            )
            if match:
                os.environ[match.group(1)] = match.group(2).strip().strip("'\"")


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else "officeqa"
    suffix = sys.argv[2] if len(sys.argv) > 2 else "r2"
    load_credentials()
    try:
        import wandb
    except ImportError:
        sys.exit("wandb not importable; run via `uv run --project vero python ...`")

    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    runs = api.runs(f"{entity}/{PROJECT}", order="-created_at", per_page=100)

    # vero appends a `--<hash>` uniquifier to the wandb_run it is given, so the
    # suffix is not anchored to end-of-string.
    want = re.compile(rf"^{re.escape(bench)}__.*__{re.escape(suffix)}(--\w+)?$")
    rows = []
    for run in runs:
        if not want.match(run.name or ""):
            continue
        summary = dict(run.summary)
        # Keys are inference/<scope>/<metric>, and the scope must be kept: keying
        # on the last path segment alone collapses `evaluation` into
        # `finalization`, which once produced a badly wrong error figure.
        scopes: dict[str, dict[str, float]] = {}
        for key, value in summary.items():
            match = re.match(r"inference/([^/]+)/(requests|upstream_errors)$", str(key))
            if match and isinstance(value, (int, float)):
                scopes.setdefault(match.group(1), {})[match.group(2)] = value
        rows.append((run.name, run.state, summary, scopes, run.created_at, run.id))

    if not rows:
        print(f"no W&B runs matching {bench}__*__{suffix} in {entity}/{PROJECT}")
        return

    def total(summary, tail):
        """Sum `<partition>/agent/evaluations/<tail>` across partitions."""
        found = [v for k, v in summary.items()
                 if str(k).endswith(f"/agent/evaluations/{tail}")
                 and isinstance(v, (int, float))]
        return int(sum(found)) if found else None

    print(f"{'run':46} {'state':8} {'ok':>3} {'kill':>4} {'err':>5} {'reqs':>7} "
          f"{'gen tok':>10} {'tok/min':>8}  per-scope err/reqs")
    flagged = []
    for name, state, summary, scopes, created, _rid in sorted(rows):
        ok = total(summary, "success_total")
        killed = total(summary, "terminated_total")
        errors = int(sum(v.get("upstream_errors", 0) for v in scopes.values()))
        reqs = int(sum(v.get("requests", 0) for v in scopes.values()))
        gen = int(sum(v for k, v in summary.items()
                      if re.match(r"inference/.+/output_tokens$", str(k))
                      and isinstance(v, (int, float))))
        minutes = max((now_utc() - parse_created(created)).total_seconds() / 60.0, 1.0)
        # Per scope, because a clean search alongside a throttled finalization is
        # a different situation from the reverse, and the totals hide which.
        detail = "  ".join(
            f"{scope}={int(v.get('upstream_errors', 0))}/{int(v.get('requests', 0))}"
            for scope, v in sorted(scopes.items())) or "-"
        print(f"{name:46} {state:8} {ok or 0!s:>3} {killed or 0!s:>4} "
              f"{errors:>5} {reqs:>7} {gen:>10,} {gen / minutes:>8,.0f}  {detail}")
        if killed:
            flagged.append(f"{name}: {killed} evaluation(s) TERMINATED")

    # terminated_total > 0 is the PR #65 regression signal. Never let it be a
    # number the reader has to notice on their own.
    for line in flagged:
        print(f"\n  *** {line} -- taxonomy regression, investigate before ramping")


def now_utc():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc)


def parse_created(value):
    """W&B hands back created_at as a naive UTC ISO string, not a datetime."""
    import datetime
    if isinstance(value, datetime.datetime):
        stamp = value
    else:
        stamp = datetime.datetime.fromisoformat(str(value).replace("Z", ""))
    return (stamp.replace(tzinfo=datetime.timezone.utc)
            if stamp.tzinfo is None else stamp)


if __name__ == "__main__":
    main()
