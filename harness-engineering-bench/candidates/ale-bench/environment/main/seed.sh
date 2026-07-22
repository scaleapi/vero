#!/bin/sh
set -eu
if [ ! -d /work/agent/.git ]; then
  cp -a /opt/agent-seed/. /work/agent/
  cd /work/agent
  git init -q
  git add -A
  git -c user.email=seed@vero.test -c user.name=seed commit -qm "baseline"
fi
find /work/agent -exec chown agent:agent {} +
git config --system --add safe.directory /work/agent
exec sleep infinity
