#!/usr/bin/env bash
# Tear down orphaned benchmark infrastructure after a killed `vero harbor run`.
#
# Killing the outer harbor process orphans everything below it: the docker
# compose topology (daemon-owned, survives its creator) and any in-flight
# Modal sandboxes (server-side, default 24h timeout). This removes both.
# Scope is strictly ours: `task__*` compose containers locally, and sandboxes
# in the dedicated `harness-engineering-bench` Modal app.
#
# Usage: bash scripts/cleanup_orphans.sh [--dry-run]
# Requires MODAL_TOKEN_ID/MODAL_TOKEN_SECRET in the environment (or a dotenv
# sourced beforehand); skips Modal cleanup if they are absent.
set -euo pipefail

dry_run=false
[ "${1:-}" = "--dry-run" ] && dry_run=true

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

uv run --quiet --python 3.12 --with modal python - "$dry_run" <<'PY'
import sys
import modal

dry_run = sys.argv[1] == "true"
try:
    app = modal.App.lookup("harness-engineering-bench", create_if_missing=False)
except Exception as error:
    print(f"no harness-engineering-bench app: {error}")
    raise SystemExit(0)
count = 0
for sandbox in modal.Sandbox.list(app_id=app.app_id):
    print(("would terminate" if dry_run else "terminating"), sandbox.object_id)
    if not dry_run:
        sandbox.terminate()
    count += 1
print(f"{count} running sandbox(es) in harness-engineering-bench")
PY
