#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p /logs/verifier
python3 /tests/evaluate.py || true
test -f /logs/verifier/reward.txt || printf '0' > /logs/verifier/reward.txt
exit 0
