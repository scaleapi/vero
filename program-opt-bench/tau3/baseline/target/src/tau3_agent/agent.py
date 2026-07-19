"""Harbor adapter for the tau3 runner that executes inside the task network."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

REMOTE_ROOT = "/tmp/vero-tau3-agent"


class Tau3Agent(BaseAgent):
    """Install and execute the editable MCP agent inside the task environment."""

    @staticmethod
    @override
    def name() -> str:
        return "tau3-responses-baseline"

    @override
    def version(self) -> str:
        return "0.1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            raise ValueError("tau3 agent requires a Harbor model")
        self._api_model = self.model_name.removeprefix("openai/")

    def _server_url(self) -> str:
        urls = [
            str(server.url)
            for server in self.mcp_servers
            if getattr(server, "transport", None) == "streamable-http"
            and getattr(server, "url", None)
        ]
        if len(urls) != 1:
            raise RuntimeError(
                "tau3 agent requires exactly one streamable-http MCP server"
            )
        return urls[0]

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        command = (
            f"mkdir -p {REMOTE_ROOT} /logs/agent && "
            "python3 -m pip install --disable-pip-version-check --no-cache-dir -q "
            "mcp==1.27.0 openai==2.46.0"
        )
        result = await environment.exec(command, timeout_sec=300)
        if result.return_code != 0:
            raise RuntimeError(result.stderr or "could not install tau3 runner")
        await environment.upload_file(
            Path(__file__).with_name("runner.py"), f"{REMOTE_ROOT}/runner.py"
        )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        instruction_path = self.logs_dir / "instruction.md"
        instruction_path.write_text(instruction, encoding="utf-8")
        await environment.upload_file(instruction_path, f"{REMOTE_ROOT}/instruction.md")

        # The runner is a fresh process inside the task environment, so forward the
        # inference credentials Harbor gave this agent into its exec — otherwise the
        # runner's AsyncOpenAI() has no key (only the host-side agent process does).
        creds = [
            f"{name}={shlex.quote(value)}"
            for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL")
            if (value := os.environ.get(name))
        ]
        command = " ".join(
            [
                *creds,
                "python3",
                f"{REMOTE_ROOT}/runner.py",
                "--instruction",
                f"{REMOTE_ROOT}/instruction.md",
                "--mcp-url",
                shlex.quote(self._server_url()),
                "--model",
                shlex.quote(self._api_model),
            ]
        )
        result = await environment.exec(command, timeout_sec=3500)
        if result.return_code != 0:
            raise RuntimeError(result.stderr or result.stdout or "tau3 runner failed")

        local_context = self.logs_dir / "tau3-context.json"
        await environment.download_file("/logs/agent/tau3-context.json", local_context)
        values = json.loads(local_context.read_text(encoding="utf-8"))
        context.n_input_tokens = int(values.get("input_tokens") or 0)
        context.n_output_tokens = int(values.get("output_tokens") or 0)
        context.n_cache_tokens = int(values.get("cached_tokens") or 0)
        context.metadata = {
            "turns": int(values.get("turns") or 0),
            "trace": "tau3-trace.jsonl",
        }
