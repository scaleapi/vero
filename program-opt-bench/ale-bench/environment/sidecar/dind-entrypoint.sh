#!/bin/sh
# Start an in-container Docker daemon (DinD) so ALE-Bench's judge + Rust-tool
# containers are SIBLINGS on a daemon whose filesystem IS this container's — the
# only way their bind mounts resolve. (A host/VM socket mount fails: the sibling
# containers resolve mount sources against the daemon host, not this container.)
# Requires the container to run privileged. Then serve (the passed command).
set -eu

dockerd >/var/log/dockerd.log 2>&1 &
for i in $(seq 1 60); do
  docker info >/dev/null 2>&1 && break
  sleep 1
done
if ! docker info >/dev/null 2>&1; then
  echo "dockerd failed to start" >&2
  cat /var/log/dockerd.log >&2 || true
  exit 1
fi

# Bootstrap the ALE-Bench C++ judge image. ALE-Bench expects the tag
# ale-bench:cpp20-202301; upstream builds it as `FROM yimjk/ale-bench:cpp20-202301`
# plus `chown UID:GID /workdir` — a no-op as root — so a pull + retag suffices.
docker pull yimjk/ale-bench:cpp20-202301
docker tag yimjk/ale-bench:cpp20-202301 ale-bench:cpp20-202301
# rust:1.79.0-buster (tool build) and httpd (vis) are pulled on demand by ale_bench.

exec "$@"
