# Main service: the optimizer workbench a coding agent works in.
# Holds the editable target program plus the VeRO CLI so the agent can call
# `evals run` / `evals status` against the trusted sidecar.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install VeRO's harbor client. VeRO is NOT on public PyPI (the `scale-vero`
# name there is an unrelated placeholder), so vendor the package into this build
# context first — e.g. `cp -r <repo>/vero environment/vero` — then install it:
COPY vero /opt/vero
RUN uv pip install --system "/opt/vero[harbor]"

COPY agent-seed /opt/agent-seed
COPY main/seed.sh /opt/seed.sh
RUN chmod +x /opt/seed.sh && useradd -m -u 1001 agent

WORKDIR /work/agent
