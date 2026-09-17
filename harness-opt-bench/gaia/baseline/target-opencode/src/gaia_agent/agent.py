"""GAIA target: the opencode coding harness, run from vendored source.

The editable surface is the whole opencode source tree under ``opencode/`` plus
this wrapper. Harbor's stock opencode runner is reused for the trajectory format
and the run command; what changes is where the binary comes from. Instead of
``npm i -g opencode-ai`` the wrapper compiles the vendored tree on the evaluation
host (see ``_build``) and uploads that binary into the task container, so an edit
anywhere in the source is what gets evaluated.

Inference goes through the OpenAI-compatible endpoint the environment provides
(``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``): the target model is registered under
opencode's ``openai`` provider with that base URL. Because the agent runs inside
the task container, the build sets ``task_services_use_upstream`` so that endpoint
is the public upstream proxy rather than the compose-internal metered gateway; the
target model is therefore fixed by this wrapper, and its use is checked after the
fact from the proxy's per-key request log.
"""

from __future__ import annotations

import asyncio
import os
import shlex
from typing import Any, override

from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from gaia_agent import _build

REMOTE_BINARY = "/usr/local/bin/opencode"

# The GAIA grader reads the exact final answer from this file; a general coding
# agent will not know that unless the task text says so.
ANSWER_FILE_INSTRUCTION = (
    "When you have the final answer, write it -- and nothing else -- to /app/answer.txt, "
    "then stop. Attached files for the question, if any, are in the working directory.\n\n"
)


def _bare_model(model_name: str) -> str:
    """``xai/grok-build-0.1`` and ``openai/grok-build-0.1`` both mean ``grok-build-0.1``."""
    return model_name.rsplit("/", 1)[-1]


def _base_url() -> str | None:
    # opencode runs inside the task container, so it needs an endpoint the container
    # can reach. Under `task_services_use_upstream` VeRO puts the public upstream in
    # OPENAI_BASE_URL; the VERO_AGENT_INFERENCE_* gateway is compose-internal and is
    # deliberately not used here.
    return os.environ.get("OPENAI_BASE_URL")


def _api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


class GaiaAgent(OpenCode):
    """opencode, built from this repository, as the GAIA agent."""

    _build_lock: asyncio.Lock | None = None

    @staticmethod
    @override
    def name() -> str:
        return "gaia-opencode"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        model_name = kwargs.get("model_name")
        if model_name is None:
            raise ValueError("GAIA agent requires a Harbor model")
        bare = _bare_model(model_name)
        # Register the model under the openai provider pointed at the gateway, so
        # opencode neither consults models.dev for it nor talks to a vendor directly.
        provider: dict[str, Any] = {"models": {bare: {}}}
        if _base_url():
            provider["options"] = {"baseURL": _base_url()}
        kwargs["model_name"] = f"openai/{bare}"
        kwargs.setdefault("opencode_config", {})
        kwargs["opencode_config"] = {
            **{"provider": {"openai": provider}},
            **kwargs["opencode_config"],
        }
        super().__init__(*args, **kwargs)

    @override
    def version(self) -> str | None:
        return self._version or "vendored"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        result = await environment.exec("mkdir -p /app", timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError(result.stderr or "could not prepare /app")
        await super().setup(environment)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        # The run pipeline pipes through `stdbuf` (coreutils); most images have it.
        # No node/npm: the binary is self-contained. harbor 0.20.0 (the sidecar's
        # version) has no ensure_system_dependencies, so do it by hand.
        await self.exec_as_root(
            environment,
            command=(
                "command -v stdbuf >/dev/null 2>&1 || "
                "(apt-get update -qq && apt-get install -y -qq coreutils) || "
                "apk add --no-cache coreutils || true"
            ),
        )
        if GaiaAgent._build_lock is None:
            GaiaAgent._build_lock = asyncio.Lock()
        async with GaiaAgent._build_lock:
            binary = await asyncio.to_thread(_build.build_binary)
        staged = "/tmp/opencode.upload"
        await environment.upload_file(binary, staged)
        await self.exec_as_root(
            environment,
            command=f"mv {shlex.quote(staged)} {shlex.quote(REMOTE_BINARY)} && chmod 755 {shlex.quote(REMOTE_BINARY)}",
        )

    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        # The stock runner copies OPENAI_API_KEY / OPENAI_BASE_URL from this process
        # into the container for the `openai` provider; nothing to remap.
        await super().run(ANSWER_FILE_INSTRUCTION + instruction, environment, context)
