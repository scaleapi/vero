"""Terminal-Bench target: the opencode coding harness, run from vendored source.

The editable surface is the whole opencode source tree under ``opencode/`` plus
this wrapper. Harbor's stock opencode runner is reused for the trajectory format
and the run command; what changes is where the binary comes from. Instead of
``npm i -g opencode-ai`` the wrapper compiles the vendored tree on the evaluation
host (see ``_build``) and uploads that binary into the task container, so an edit
anywhere in the source is what gets evaluated.

Inference goes through the OpenAI-compatible endpoint the environment provides
(``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``). The target model is registered under
a provider id opencode has no builtin loader for (see ``WIRE_PROVIDER_ID``) so it
gets a plain Chat Completions client instead of one of opencode's vendor-specific
loaders. Because the agent runs inside the task container, the build sets
``task_services_use_upstream`` so that endpoint is the public upstream litellm
proxy rather than the compose-internal metered gateway; the target model is
therefore fixed by this wrapper, and its use is checked after the fact from the
proxy's per-key request log.
"""

from __future__ import annotations

import asyncio
import os
import shlex
from typing import Any, override

from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment

from terminal_bench_agent import _build

REMOTE_BINARY = "/usr/local/bin/opencode"


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


def _wire_model_id(bare: str) -> str:
    """The model string actually sent to the upstream endpoint.

    Harbor passes the bare name (``grok-build-0.1``) as ``model_name``, but the
    litellm proxy's model_group for it is registered under the vendor-prefixed
    form (``xai/grok-build-0.1``); the bare name 401s ("key not allowed to
    access model") and litellm's error for it misleadingly claims the key can
    only reach a model_group named "default" - that group has no healthy
    deployment and is a red herring. opencode's config lets a model declare a
    separate wire id (``models.<key>.id``) while keeping ``bare`` as the
    routing key everywhere else (CLI --model, prompt selection, logging), so
    only the outgoing request string changes. Overridable for other envs.
    """
    return os.environ.get("VERO_OPENCODE_WIRE_MODEL_ID", f"xai/{bare}")


# A provider id opencode has no special-cased behaviour for. opencode's builtin
# "openai" provider loader unconditionally sends every request through the
# Responses API (`sdk.responses(modelID)`, see provider/provider.ts), which the
# upstream litellm proxy 400s on for this model/key ("no healthy deployments
# for this model" at /v1/responses). Registering under a provider id opencode
# doesn't recognize falls back to the plain @ai-sdk/openai-compatible client,
# which talks Chat Completions instead - the endpoint the proxy actually has a
# healthy deployment for.
WIRE_PROVIDER_ID = "opencode-target"


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
        wire_id = _wire_model_id(bare)
        # Register the model under a provider id opencode has no builtin loader for
        # (see WIRE_PROVIDER_ID), so opencode neither consults models.dev for it nor
        # talks to a vendor directly, and defaults to a plain Chat Completions client.
        model_config: dict[str, Any] = {"id": wire_id} if wire_id != bare else {}
        provider: dict[str, Any] = {"npm": "@ai-sdk/openai-compatible", "models": {bare: model_config}}
        options: dict[str, Any] = {}
        if _base_url():
            options["baseURL"] = _base_url()
        if _api_key():
            options["apiKey"] = _api_key()
        if options:
            provider["options"] = options
        kwargs["model_name"] = f"{WIRE_PROVIDER_ID}/{bare}"
        kwargs.setdefault("opencode_config", {})
        kwargs["opencode_config"] = {
            **{"provider": {WIRE_PROVIDER_ID: provider}},
            **kwargs["opencode_config"],
        }
        super().__init__(*args, **kwargs)

    @override
    def version(self) -> str | None:
        return self._version or "vendored"

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
        if TerminalBenchAgent._build_lock is None:
            TerminalBenchAgent._build_lock = asyncio.Lock()
        async with TerminalBenchAgent._build_lock:
            binary = await asyncio.to_thread(_build.build_binary)
        staged = "/tmp/opencode.upload"
        await environment.upload_file(binary, staged)
        await self.exec_as_root(
            environment,
            command=f"mv {shlex.quote(staged)} {shlex.quote(REMOTE_BINARY)} && chmod 755 {shlex.quote(REMOTE_BINARY)}",
        )
