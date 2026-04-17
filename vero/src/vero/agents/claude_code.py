"""ClaudeCodeAgent - Agent backend using Claude Agent SDK for optimization."""

from __future__ import annotations

import logging
import typing
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ClaudeSDKError,
    McpSdkServerConfig,
    ResultMessage,
    SdkMcpTool,
    create_sdk_mcp_server,
)
from claude_agent_sdk.types import (
    HookContext,
    HookEvent,
    HookMatcher,
    Message,
    PostToolUseHookInput,
    PostToolUseHookSpecificOutput,
    PreToolUseHookInput,
    PreToolUseHookSpecificOutput,
    SystemPromptPreset,
)
from pydantic import BaseModel, create_model

from vero.agents.base import BaseAgent
from vero.agents.events import AgentEvent
from vero.filesystem import AccessType
from vero.session import BestVersion, Session
from vero.tools import (
    DatasetViewer,
    ExperimentRunnerTool,
    ExperimentViewer,
)
from vero.tools.base import ToolSet
from vero.tools.utils import get_tools_from_class
from vero.utils import recursively_serialize

logger = logging.getLogger(__name__)


def _raise_claude_sdk_error(msg: str):
    raise ClaudeSDKError(msg)


def default_tool_sets() -> list:
    """Default tool sets for ClaudeCodeAgent."""
    return [DatasetViewer(), ExperimentRunnerTool(on_fatal=_raise_claude_sdk_error), ExperimentViewer()]


class ClaudeCodeHookBuilder:
    """Builds hooks for the ClaudeCodeAgent."""

    def __init__(self, agent: ClaudeCodeAgent):
        self.agent = agent

    def get_hooks(self) -> dict[HookEvent, list[HookMatcher]]:
        return {
            "PostToolUse": [
                HookMatcher(matcher="Write|Edit", hooks=[self.on_write_edit]),
            ],
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[self.check_bash_command])
            ],
        }

    async def on_write_edit(
        self, input_data: PostToolUseHookInput, tool_use_id: str, context: HookContext
    ) -> PostToolUseHookSpecificOutput | dict:
        tool_name = input_data["tool_name"]
        tool_input = input_data["tool_input"]

        if tool_name not in ["Write", "Edit"]:
            return {}

        commit_message = f"Committing changes from command: {tool_name} to file: {tool_input.get('file_path', '')}"
        logger.info(commit_message)

        if not self.agent._session or not self.agent._session.workspace:
            return {}

        try:
            await self.agent._session.workspace.save(commit_message)
        except Exception as e:
            return PostToolUseHookSpecificOutput(
                hookEventName="PostToolUse",
                additionalContext=f"Error committing changes: {e}",
            )
        return PostToolUseHookSpecificOutput(
            hookEventName="PostToolUse",
            additionalContext=f"Committed changes: {commit_message}",
        )

    async def check_bash_command(
        self, input_data: PreToolUseHookInput, tool_use_id: str, context: HookContext
    ) -> PreToolUseHookSpecificOutput | dict:
        tool_name = input_data["tool_name"]
        tool_input = input_data["tool_input"]
        if tool_name != "Bash":
            return {}
        command = tool_input.get("command", "")

        def cannot_checkout_other_branches(cmd: str) -> bool:
            return "git" in cmd and "checkout" in cmd

        def cannot_do_recursive_deletes(cmd: str) -> bool:
            return "rm" in cmd and ("-rf" in cmd or "-r" in cmd)

        def cannot_echo_env_vars(cmd: str) -> bool:
            return "echo" in cmd and "$" in cmd

        for test in [
            cannot_checkout_other_branches,
            cannot_do_recursive_deletes,
            cannot_echo_env_vars,
        ]:
            if test(command):
                return PreToolUseHookSpecificOutput(
                    hookEventName="PreToolUse",
                    permissionDecision="deny",
                    permissionDecisionReason=f"Command failed forbidden pattern test: {test.__name__}",
                )
        return {}


def _default_claude_options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model="claude-sonnet-4-5-20250929",
        permission_mode="bypassPermissions",
        allowed_tools=["WebSearch", "WebFetch", "Task", "Bash"],
    )


@dataclass
class ClaudeCodeAgent(BaseAgent):
    """Agent backend using the Claude Agent SDK (Claude Code)."""

    options: ClaudeAgentOptions = field(default_factory=_default_claude_options)
    tool_sets: list[ToolSet] = field(default_factory=default_tool_sets)
    enable_hooks: bool = True
    output_format: type[BaseModel] | None = None
    trace: list[Message] = field(default_factory=list, repr=False)
    state: dict[str, str] | None = field(default=None, repr=False)

    _session: Session | None = field(default=None, repr=False)
    _tools: dict[str, McpSdkServerConfig] = field(default_factory=dict, repr=False)
    _allowed_tools: list[str] = field(default_factory=list, repr=False)

    def init(self, session: Session) -> None:
        """Initialize the agent with a Session context."""
        self._session = session
        for tool_set in self.tool_sets:
            if isinstance(tool_set, ExperimentRunnerTool):
                tool_set.on_fatal = _raise_claude_sdk_error
        self._tools, self._allowed_tools = self._create_tools()

    async def step(
        self, input: str, max_turns: int = 200, on_event: Any | None = None, **kwargs
    ) -> list[Message]:
        """Execute optimization steps using the Claude Agent SDK."""
        assert self._session, "Session is not set! Call init(session) first."

        async with self._create_client(max_turns=max_turns) as client:
            await client.query(input)
            async for msg in client.receive_response():
                self.trace.append(msg)
                if on_event is not None:
                    on_event(msg)

        # Update state with session ID for resumption
        result = self.latest_result
        if result is not None and hasattr(result, "session_id"):
            self.state = {"session_id": result.session_id}

        return self.trace

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
        """Return ClaudeCodeAgent-specific fields for Policy.as_dict()."""
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

    def get_best_version(self) -> BestVersion:
        """Extract best commit from structured output if available."""

        output = self.latest_structured_output
        if output is not None and hasattr(output, "best_commit") and output.best_commit:
            return BestVersion(
                commit=output.best_commit,
                score=getattr(output, "best_score", None),
            )
        return BestVersion()

    def reset_trace(self) -> None:
        """Resets the trace."""
        self.trace = []

    def tool_set_enabled(
        self, tool_set_key: type | str, for_sub_agent: bool = False
    ) -> bool:
        """Check if a tool set is enabled. Accepts class or string name."""
        if isinstance(tool_set_key, str):
            return any(type(ts).__name__ == tool_set_key for ts in self.tool_sets)
        return any(isinstance(ts, tool_set_key) for ts in self.tool_sets)

    @property
    def orchestrator_tool_sets(self) -> list[type]:
        return [type(ts) for ts in self.tool_sets]

    @property
    def sub_agents_enabled(self) -> bool:
        return False  # ClaudeCodeAgent doesn't support sub-agents

    @property
    def sub_agent_tool_sets(self) -> list[type]:
        return []

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _create_client(self, max_turns: int | None = None) -> ClaudeSDKClient:
        """Creates the ClaudeSDKClient for this step."""
        from copy import copy

        options = copy(self.options)

        # Set cwd from policy
        if self._session and self._session.project_path:
            options.cwd = self._session.project_path

        # System prompt
        options.system_prompt = self._build_system_prompt()

        # MCP servers
        options.mcp_servers = self._tools
        options.allowed_tools = list(set(self._allowed_tools + options.allowed_tools))

        # Disallowed tools from filesystem accesses
        options.disallowed_tools = self._build_disallowed_tools()

        # Resume from saved state
        if self.state and isinstance(self.state, dict) and "session_id" in self.state:
            options.resume = self.state["session_id"]
            options.continue_conversation = True
            logger.info(f"Resuming Claude Code session: {self.state['session_id']}")

        # Hooks
        if self.enable_hooks:
            hooks_builder = ClaudeCodeHookBuilder(self)
            options.hooks = options.hooks or {}
            options.hooks.update(hooks_builder.get_hooks())

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

        assert self._session, "Session not set!"

        instructions = self._session.instructions

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

        return instructions or ""

    def _build_disallowed_tools(self) -> list[str]:
        """Builds the list of disallowed tools from filesystem accesses."""
        disallowed_tools = []
        if not self._session or not self._session.workspace:
            return disallowed_tools
        for access in self._session.workspace.accesses:
            if access.access_type == AccessType.EXCLUDE:
                disallowed_tools.append(f"Read(./{access.pattern})")
                disallowed_tools.append(f"Write(./{access.pattern})")
                disallowed_tools.append(f"Edit(./{access.pattern})")
            elif access.access_type == AccessType.READ:
                disallowed_tools.append(f"Write(./{access.pattern})")
                disallowed_tools.append(f"Edit(./{access.pattern})")
        return disallowed_tools

    def _create_tools(
        self,
    ) -> tuple[dict[str, McpSdkServerConfig], list[str]]:
        """Creates the tool set instances based on tool_sets."""

        assert self._session, "Session not set!"
        tools: dict[str, McpSdkServerConfig] = {}
        internal_allowed_tools: list[str] = []

        for tool_set in self.tool_sets:
            if hasattr(tool_set, "bind"):
                tool_set.bind(self._session)

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

        from vero.core.utils import maybe_await

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
                        content = await maybe_await(m(**args))
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
