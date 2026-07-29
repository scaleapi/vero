#!/usr/bin/env bash
# Tear down orphaned benchmark infrastructure after a killed `vero harbor run`.
#
# Killing the outer harbor process orphans everything below it: the docker
# compose topology (daemon-owned, survives its creator) and any in-flight Modal
# sandboxes (server-side, default 24h timeout). Confirmed on 2026-07-29 -- three
# browsecomp-plus runs killed at 07:00 and 07:53 kept working for over six hours
# after their local processes died, one advancing from 7,632 to 18,608 evaluation
# requests. So killing a run does not stop it; it only discards the local
# collector and makes the work unrecoverable. Load shedding is not available as a
# mitigation in this architecture, and an intentional stop needs BOTH halves.
#
# Usage:
#   bash scripts/cleanup_orphans.sh --dry-run              # always start here
#   bash scripts/cleanup_orphans.sh --app NAME [--yes]
#   bash scripts/cleanup_orphans.sh --keep sb-A,sb-B --yes # spare specific ones
#
#   --app NAME     Modal app to clean. Default harness-engineering-bench.
#   --keep IDS     Comma-separated sandbox ids to spare.
#   --yes          Actually terminate. Without it this only reports.
#
# THE SAFETY PROBLEM THIS DOES NOT FULLY SOLVE. Every grid run currently shares
# one Modal app (`--ek app_name=harness-engineering-bench` in each build.yaml, set
# deliberately so the whole suite is visible in one place), and harbor creates the
# sandboxes, so we cannot tag them per run -- Modal supports Sandbox.list(tags=)
# and set_tags(), but only for sandboxes we create ourselves. A dry run at 14:13Z
# on 2026-07-29 listed 223 sandboxes, mixing 8 healthy runs with the orphans, and
# terminating all of them would have destroyed the pass.
#
# Until sandboxes can be attributed to runs, there are two safe patterns:
#   1. Run this only BETWEEN passes, when nothing should be alive.
#   2. Give each run its own Modal app at launch: pass `--ek app_name=<cell>`,
#      which overrides the build.yaml value (harbor takes the last value per key),
#      then clean one cell with `--app <cell>`. This fragments the Modal UI view,
#      which is the reason it is not the default, but it is the only way to
#      target a single run today.
# Otherwise: orphans expire on Modal's own 24h timeout, so doing nothing is a
# valid choice and is safer than terminating indiscriminately.
#
# Requires MODAL_TOKEN_ID/MODAL_TOKEN_SECRET in the environment (or a dotenv
# sourced beforehand); skips Modal cleanup if they are absent.
set -euo pipefail

app_name=harness-engineering-bench
keep=""
confirmed=false
dry_run=false
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry_run=true ;;
        --yes) confirmed=true ;;
        --app) app_name="${2:?--app needs a name}"; shift ;;
        --keep) keep="${2:?--keep needs sandbox ids}"; shift ;;
        *) echo "unknown argument: $1" >&2; sed -n '11,20p' "$0" >&2; exit 2 ;;
    esac
    shift
done
# Reporting is the default. Terminating requires --yes, so a bare invocation can
# never destroy a live pass by accident.
$confirmed || dry_run=true

containers=$(docker ps --format '{{.Names}}' | grep -i '^task__' || true)
if [ -n "$containers" ]; then
    echo "orphaned containers:"; echo "$containers"
    if ! $dry_run; then
        echo "$containers" | xargs docker rm -f
    fi
else
    echo "no orphaned task__* containers"
fi

if [ -z "${MODAL_TOKEN_ID:-}" ]; then
    echo "MODAL_TOKEN_ID not set; skipping Modal sandbox cleanup"
    exit 0
fi

uv run --quiet --python 3.12 --with modal python - "$dry_run" "$app_name" "$keep" <<'PY'
import sys
import modal

dry_run = sys.argv[1] == "true"
app_name = sys.argv[2]
keep = {value for value in sys.argv[3].split(",") if value}

try:
    app = modal.App.lookup(app_name, create_if_missing=False)
except Exception as error:
    print(f"no {app_name!r} app: {error}")
    raise SystemExit(0)

terminated = spared = 0
for sandbox in modal.Sandbox.list(app_id=app.app_id):
    if sandbox.object_id in keep:
        print(f"  sparing      {sandbox.object_id}")
        spared += 1
        continue
    print(f"  {'would terminate' if dry_run else 'terminating   '} {sandbox.object_id}")
    if not dry_run:
        sandbox.terminate()
    terminated += 1

total = terminated + spared
print(f"{total} running sandbox(es) in {app_name!r}")
if dry_run and terminated:
    print(
        f"nothing was terminated. Re-run with --yes to remove {terminated}, and "
        "make sure no pass is in flight -- these cannot be attributed to a run, "
        "so a healthy run's sandboxes look identical to an orphan's."
    )
PY
