# Main service: the optimizer workbench a coding agent works in.
# Holds the editable target program plus the VeRO CLI so the agent can call
# `vero harbor eval` / `vero harbor status` against the trusted sidecar.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install VeRO's harbor client. From a local checkout, replace with:
#   COPY vero /opt/vero && uv pip install --system "/opt/vero[harbor]"
RUN uv pip install --system "scale-vero[harbor]"

COPY agent-seed /opt/agent-seed
COPY main/seed.sh /opt/seed.sh
RUN chmod +x /opt/seed.sh && useradd -m -u 1001 agent

WORKDIR /work/agent
