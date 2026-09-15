"""Terminal-Bench target: the opencode coding harness, run from vendored source.

The editable surface is the whole opencode source tree under ``opencode/`` plus
this wrapper. Harbor's stock opencode runner is reused for the trajectory format
and the run command; what changes is where the binary comes from. Instead of
``npm i -g opencode-ai`` the wrapper compiles the vendored tree on the evaluation
host (see ``_build``) and uploads that binary into the task container, so an edit
anywhere in the source is what gets evaluated.

Inference goes through the OpenAI-compatible endpoint the environment provides
(``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``): the target model is registered under
opencode's ``openai`` provider with that base URL, and the bare model name is what
the gateway allow-lists.
"""

from __future__ import annotations

import asyncio
import os
import shlex
from typing import Any, override

from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from terminal_bench_agent import _build

REMOTE_BINARY = "/usr/local/bin/opencode"


def _bare_model(model_name: str) -> str:
    """``xai/grok-build-0.1`` and ``openai/grok-build-0.1`` both mean ``grok-build-0.1``."""
    return model_name.rsplit("/", 1)[-1]


def _base_url() -> str | None:
    return os.environ.get("VERO_AGENT_INFERENCE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")


def _api_key() -> str | None:
    return os.environ.get("VERO_AGENT_INFERENCE_API_KEY") or os.environ.get("OPENAI_API_KEY")


class TerminalBenchAgent(OpenCode):
    """opencode, built from this repository, as the Terminal-Bench agent."""

    _build_lock: asyncio.Lock | None = None

    @staticmethod
    @override
    def name() -> str:
        return "terminal-bench-opencode"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        model_name = kwargs.get("model_name")
        if model_name is None:
            raise ValueError("Terminal-Bench agent requires a Harbor model")
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
    async def install(self, environment: BaseEnvironment) -> None:
        # stdbuf for the run pipeline; no node/npm: the binary is self-contained.
        await self.ensure_system_dependencies(environment, ("bash", "coreutils"))
        if TerminalBenchAgent._build_lock is None:
            TerminalBenchAgent._build_lock = asyncio.Lock()
        async with TerminalBenchAgent._build_lock:
            binary = await asyncio.to_thread(_build.build_binary)
        await environment.upload_file(binary, REMOTE_BINARY)
        await self.exec_as_root(environment, command=f"chmod 755 {shlex.quote(REMOTE_BINARY)}")

    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        # The stock runner reads the key and base URL from the model connection;
        # make sure the gateway pair is what reaches the container even when the
        # host only exports the VERO_AGENT_INFERENCE_* names.
        if _api_key() and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = _api_key()  # type: ignore[arg-type]
        if _base_url() and not os.environ.get("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = _base_url()  # type: ignore[arg-type]
        await super().run(instruction, environment, context)
