# Trusted evaluation sidecar. It owns scoring, budget, disclosure, and final
# selection. The inner loop is a plain CommandBackend (harness/score.py), which
# calls ALE-Bench's deterministic judge. ALE-Bench spawns the C++ judge + its
# Rust-tool builder as sibling Docker containers with bind mounts, so this
# container runs its OWN Docker daemon (DinD) — the sibling containers must share
# this container's filesystem for their mounts to resolve. Requires privileged.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git ca-certificates curl \
        docker.io iptables \
        libcairo2 libffi8 \
    && rm -rf /var/lib/apt/lists/*

# VeRO (not on public PyPI): vendor into the build context, then install.
COPY vero /opt/vero
RUN uv pip install --system "/opt/vero[harbor]"

# ALE-Bench toolkit (the judge orchestration + scoring).
RUN uv pip install --system "git+https://github.com/SakanaAI/ALE-Bench.git"

COPY agent-baseline /opt/agent-baseline
COPY sidecar/ale_factory.py /opt/ale_factory.py
COPY sidecar/harness /opt/harness
COPY sidecar/serve.json /opt/serve.json
COPY sidecar/dind-entrypoint.sh /opt/dind-entrypoint.sh
RUN chmod +x /opt/dind-entrypoint.sh

RUN cd /opt/agent-baseline \
    && git init -q \
    && git add -A \
    && git -c user.email=baseline@vero.test -c user.name=baseline commit -qm "baseline" \
    && git config --system --add safe.directory /opt/agent-baseline \
    && git config --system --add safe.directory /work/agent \
    && git config --system --add safe.directory /work/agent/.git

ENV PYTHONPATH=/opt
WORKDIR /opt
# Start the in-container Docker daemon + bootstrap the judge image, then exec the
# serve command passed by docker-compose.
ENTRYPOINT ["/opt/dind-entrypoint.sh"]
