#!/bin/bash
# Harbor reads the score from /logs/verifier/reward.txt. Always leave a value
# there and always exit 0, so a verifier crash reads as 0.0 rather than as
# infrastructure error -- an unanswered task is a legitimate zero.
set -Eeuo pipefail

mkdir -p /logs/verifier
python3 /tests/verify.py || echo 0 > /logs/verifier/reward.txt
cat /logs/verifier/reward.txt
exit 0
