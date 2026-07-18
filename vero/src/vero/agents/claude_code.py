"""ClaudeCodeAgent - Agent backend using Claude Agent SDK for optimization."""

from __future__ import annotations

import inspect
import logging
import typing
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    McpSdkServerConfig,
    ResultMessage,
    SdkMcpTool,
    create_sdk_mcp_server,
)
from claude_agent_sdk.types import (
    Message,
    SystemPromptPreset,
)
from pydantic import BaseModel, create_model

from vero.agents.events import AgentEvent
from vero.agents.protocol import AgentContext, AgentRequirements, AgentRunResult
from vero.tools.base import ToolSet
from vero.tools.evaluation import EvaluationTools
from vero.tools.utils import get_tools_from_class
from vero.utils.general import recursively_serialize

logger = logging.getLogger(__name__)


def default_tool_sets() -> list:
    """Default tool sets for ClaudeCodeAgent."""
    return [EvaluationTools()]


def _default_claude_options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model="claude-sonnet-4-5-20250929",
        permission_mode="bypassPermissions",
        allowed_tools=["WebSearch", "WebFetch", "Task", "Bash"],
    )


@dataclass
class ClaudeCodeAgent:
    """Claude Agent SDK coding agent for a scoped optimization proposal."""

    requirements = AgentRequirements(host_visible_workspace=True)

    options: ClaudeAgentOptions = field(default_factory=_default_claude_options)
    tool_sets: list[ToolSet] = field(default_factory=default_tool_sets)
    output_format: type[BaseModel] | None = None
    trace: list[Message] = field(default_factory=list, repr=False)
    state: dict[str, str] | None = field(default=None, repr=False)

    _context: AgentContext | None = field(default=None, repr=False)
    _tools: dict[str, McpSdkServerConfig] = field(default_factory=dict, repr=False)
    _allowed_tools: list[str] = field(default_factory=list, repr=False)

    async def run(
        self,
        *,
        context: AgentContext,
        prompt: str | None,
        max_turns: int,
        on_event: Callable[[Any], Any] | None = None,
    ) -> AgentRunResult:
        """Run Claude Code in the candidate workspace."""

        self._context = context
        self._tools, self._allowed_tools = self._create_tools()
        input = prompt or "Improve the program, using evaluation feedback when useful."

        async with self._create_client(max_turns=max_turns) as client:
            await client.query(input)
            async for msg in client.receive_response():
                self.trace.append(msg)
                if on_event is not None:
                    event_result = on_event(msg)
                    if inspect.isawaitable(event_result):
                        await event_result

        # Update state with session ID for resumption
        result = self.latest_result
        if result is not None and hasattr(result, "session_id"):
            self.state = {"session_id": result.session_id}

        metadata = {"usage": self.usage()}
        model = self.options.model
        if model is not None:
            metadata["model"] = str(model)
        return AgentRunResult(
            description="Apply Claude coding-agent changes",
            state=recursively_serialize(self.serialize_state()),
            trace=recursively_serialize(self.serialize_trace()),
            metadata=metadata,
        )

    def serialize_event(self, event: Any) -> AgentEvent | None:
        """Convert a Claude Agent SDK Message to a normalized AgentEvent."""
        if isinstance(event, ResultMessage):
            text = ""
            if hasattr(event, "result") and event.result:
                text = str(event.result)
            return {"kind": "result", "text": text}

        if not hasattr(event, "content"):
            return None

        # Claude SDK Message has .content (str) and .role
        content = getattr(event, "content", "")
        role = getattr(event, "role", "")

        if role == "system":
            return {"kind": "system", "text": str(content)[:200]}

        if role == "assistant":
            return {"kind": "message", "text": str(content) if content else ""}

        if role == "tool":
            name = getattr(event, "tool_name", "")
            return {
                "kind": "tool_result",
                "name": name,
                "output": str(content)[:500] if content else "",
                "is_error": bool(getattr(event, "is_error", False)),
            }

        # Fallback
        return {"kind": "message", "text": str(content) if content else ""}

    def serialize_trace(self) -> Any:
        """Serialize the execution trace (event log)."""
        if not self.trace:
            return None
        return [asdict(msg) for msg in self.trace]

    def serialize_state(self) -> dict[str, str] | None:
        """Serialize state for resumption — the Claude SDK session ID."""
        return self.state

    def deserialize_state(self, state: dict[str, str]) -> None:
        """Restore state by setting the session ID for resume."""
        self.state = state

    def usage(self) -> dict:
        """Return usage statistics from the latest result."""
        result = self.latest_result
        if result is None:
            return {}
        usage = recursively_serialize(result.usage)
        if not usage:
            return {}
        return usage

    def summary(self) -> dict:
        """Return structured output for wandb summary."""
        structured_output = None
        if self.latest_structured_output is not None:
            try:
                structured_output = self.latest_structured_output.model_dump()
            except AttributeError:
                structured_output = self.latest_structured_output
        return {"structured_output": structured_output}

    def dict(self) -> dict[str, Any]:
        """Return serializable Claude Code configuration."""
        return {
            "claude_agent_options": recursively_serialize(
                {
                    "allowed_tools": self.options.allowed_tools,
                    "permission_mode": self.options.permission_mode,
                    "model": self.options.model,
                    "cwd": str(self.options.cwd) if self.options.cwd else None,
                }  # type: ignore
            ),
        }

    @property
    def latest_result(self) -> ResultMessage | None:
        """Gets the latest result from the run result."""
        if not self.trace:
            return None
        for msg in reversed(self.trace):
            if isinstance(msg, ResultMessage):
                return msg
        return None

    @property
    def latest_structured_output(self) -> BaseModel | None:
        """Gets the latest structured output, parsed into the output_format model."""
        result = self.latest_result

        if result is None or result.structured_output is None:
            return None

        if self.output_format is None:
            return result.structured_output

        try:
            if isinstance(result.structured_output, dict):
                return self.output_format.model_validate(result.structured_output)
            elif isinstance(result.structured_output, str):
                return self.output_format.model_validate_json(result.structured_output)
            else:
                return result.structured_output
        except Exception as e:
            logger.warning(f"Failed to parse structured output: {e}")
            return result.structured_output

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _create_client(self, max_turns: int | None = None) -> ClaudeSDKClient:
        """Creates the ClaudeSDKClient for this step."""
        from copy import copy

        options = copy(self.options)

        if self._context is not None:
            options.cwd = self._context.project_path

        # System prompt
        options.system_prompt = self._build_system_prompt()

        # MCP servers
        options.mcp_servers = self._tools
        options.allowed_tools = list(set(self._allowed_tools + options.allowed_tools))

        # Resume from saved state
        if self.state and isinstance(self.state, dict) and "session_id" in self.state:
            options.resume = self.state["session_id"]
            options.continue_conversation = True
            logger.info(f"Resuming Claude Code session: {self.state['session_id']}")

        # Max turns
        if max_turns is not None:
            options.max_turns = max_turns

        # Output format
        if self.output_format is not None:
            options.output_format = {
                "type": "json_schema",
                "schema": self.output_format.model_json_schema(),
            }

        return ClaudeSDKClient(options=options)

    def _build_system_prompt(self) -> str | SystemPromptPreset:
        """Builds the system prompt from instructions and/or options.system_prompt."""

        assert self._context is not None, "Agent context is not set"

        instructions = self._context.instructions

        # If options already has a system_prompt (e.g. SystemPromptPreset), merge with instructions
        if (
            isinstance(self.options.system_prompt, dict)
            and "type" in self.options.system_prompt
        ):
            preset = self.options.system_prompt
            if instructions:
                current_append = preset.append or ""
                new_append = (
                    current_append + "\n\n" + instructions
                    if current_append
                    else instructions
                )
                return SystemPromptPreset(
                    type=preset.type,
                    preset=preset.preset,
                    append=new_append,
                )
            return preset

        configured = self.options.system_prompt
        if isinstance(configured, str) and configured:
            return (
                f"{configured}\n\n{instructions}"
                if instructions
                else configured
            )
        return instructions or ""

    def _create_tools(
        self,
    ) -> tuple[dict[str, McpSdkServerConfig], list[str]]:
        """Creates the tool set instances based on tool_sets."""

        assert self._context is not None, "Agent context is not set"
        tools: dict[str, McpSdkServerConfig] = {}
        internal_allowed_tools: list[str] = []

        for tool_set in self.tool_sets:
            if hasattr(tool_set, "bind"):
                tool_set.bind(self._context)

            tool_names, mcp_config = self._to_claude_sdk_server_config(tool_set)
            key = type(tool_set).__name__
            tools[key] = mcp_config
            for name in tool_names:
                internal_allowed_tools.append(f"mcp__{key}__{name}")

        return tools, internal_allowed_tools

    @staticmethod
    def _to_claude_sdk_server_config(
        instance: object,
    ) -> tuple[list[str], McpSdkServerConfig]:
        """Convert a tool instance to a Claude Agent SDK server config."""
        import inspect

        tool_methods = get_tools_from_class(instance)
        sdk_mcp_tools: list[SdkMcpTool] = []
        tool_names: list[str] = []

        for method in tool_methods:
            name = method.__name__
            description = method.__doc__ or ""
            # Use get_type_hints() to resolve string annotations (from __future__ import annotations)
            try:
                resolved_hints = typing.get_type_hints(method, include_extras=True)
            except Exception:
                resolved_hints = {}
            params = {
                param.name: (resolved_hints.get(param.name, param.annotation), ...)
                for param in inspect.signature(method).parameters.values()
                if param.name != "self"
            }
            input_schema = create_model(name, **params).model_json_schema()

            def make_handler(m: Callable) -> Callable:
                def format_content(content: str | Exception) -> list[dict]:
                    return [{"type": "text", "text": str(content)}]

                async def handler(args: dict) -> dict:
                    try:
                        content = m(**args)
                        if inspect.isawaitable(content):
                            content = await content
                        return {"content": format_content(content)}
                    except Exception as e:
                        return {"content": format_content(e), "is_error": True}

                return handler

            sdk_mcp_tools.append(
                SdkMcpTool(
                    name=name,
                    description=description,
                    input_schema=input_schema,
                    handler=make_handler(method),
                )
            )
            tool_names.append(name)

        return tool_names, create_sdk_mcp_server(
            name=instance.__class__.__name__, version="1.0.0", tools=sdk_mcp_tools
        )
