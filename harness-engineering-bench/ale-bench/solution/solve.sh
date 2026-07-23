#!/bin/sh
set -eu
cd /work/agent
git config user.name optimizer; git config user.email o@v.test
vero harbor eval --backend cmd --evaluation-set ale-bench --partition validation --start 0 --stop 1
vero harbor status
