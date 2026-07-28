# Trusted evaluation sidecar. It owns scoring, budget, disclosure, and final
# candidate selection. The inner loop is a plain CommandBackend (circle-packing's
# evaluate.py) — a SIMPLE inner loop, NOT a nested `harbor run`.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# VeRO is NOT on public PyPI (the `scale-vero` name there is an unrelated
# placeholder), so vendor the package into this build context first — e.g.
# `cp -r <repo>/vero environment/vero` — then install it:
COPY vero /opt/vero
RUN uv pip install --system "/opt/vero[harbor]"

COPY agent-baseline /opt/agent-baseline
COPY sidecar/circle_factory.py /opt/circle_factory.py
COPY sidecar/harness /opt/harness
COPY sidecar/serve.json /opt/serve.json

# The sidecar holds the trusted baseline as a git repo (source of the baseline
# candidate; also the transport root for importing the agent's commits).
RUN cd /opt/agent-baseline \
    && git init -q \
    && git add -A \
    && git -c user.email=baseline@vero.test -c user.name=baseline commit -qm "baseline" \
    && git config --system --add safe.directory /opt/agent-baseline \
    && git config --system --add safe.directory /work/agent \
    && git config --system --add safe.directory /work/agent/.git

ENV PYTHONPATH=/opt
WORKDIR /opt
