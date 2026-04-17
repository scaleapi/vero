"""Sub-agent tool for spawning child agents with access to vero tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

from vero.exceptions import StreamEventTimeout
from vero.tools.utils import is_tool

if TYPE_CHECKING:
    from agents import Agent, RunResultStreaming, Tool

logger = logging.getLogger(__name__)

MAX_SANITIZATION_RETRIES = 3


def default_sub_agent_tools() -> set[type | Callable]:
    """Default tools available to sub-agents."""
    from vero.tools import (
        BashTool,
        DatasetViewer,
        ExperimentViewer,
        FileRead,
        FileWrite,
        GitViewer,
        Grep,
        TodoList,
        WebFetch,
        WebSearch,
        think,
    )

    return {
        BashTool,
        DatasetViewer,
        ExperimentViewer,
        FileRead,
        FileWrite,
        GitViewer,
        Grep,
        TodoList,
        WebFetch,
        WebSearch,
        think,
    }


class SubAgentSession(NamedTuple):
    """A sub-agent with its agent and session."""

    agent: Agent
    session: RunResultStreaming | None = None

    def to_dict(self) -> dict:
        from agents import RunResultStreaming

        def serialize_agent(agent: Agent) -> dict:
            agent_dict: dict[str, Any] = {"name": agent.name}
            if agent.instructions is not None and isinstance(agent.instructions, str):
                agent_dict["instructions"] = agent.instructions
            agent_dict["tools"] = [tool.name for tool in agent.tools]
            return agent_dict

        session_json = None
        if isinstance(self.session, RunResultStreaming):
            session_json = self.session.to_input_list()

        return {"agent": serialize_agent(self.agent), "session": session_json}


@dataclass
class SubAgentTool:
    """Spawn child agents with access to a subset of vero tools.

    The orchestrator invokes call_sub_agent to delegate tasks.
    Sub-agents can use tools from `allowed_tools` (filtered from `tool_instances`).
    """

    exclude_tools: list[str] = field(default_factory=list)
    allowed_tools: set[type | Callable] = field(default_factory=default_sub_agent_tools)
    sessions: dict[str, SubAgentSession] = field(default_factory=dict)

    # Runtime fields — set by the agent after binding other ToolSets
    tool_instances: dict[type | Callable, list[Tool]] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)

    def bind(self, session) -> None:
        pass  # tool_instances and models are set by the agent after binding other ToolSets

    @property
    def available_tool_sets(self) -> dict[type | Callable, list[Tool]]:
        return {k: v for k, v in self.tool_instances.items() if k in self.allowed_tools}

    @is_tool
    async def call_sub_agent(
        self,
        prompt: str,
        instructions: str | None = None,
        max_turns: int = 50,
        agent_name: str | None = None,
        tool_sets: list[str] | None = None,
        model_alias: str | None = None,
    ) -> str:
        """Invoke a sub-agent to perform a task with the given tools.

        Use sub-agents to delegate tasks like summarizing code, analyzing experiments,
        searching the web, or getting feedback on solutions.

        Args:
            prompt: The prompt for the sub-agent.
            instructions: Sub-agent system instructions.
            max_turns: Maximum turns for the sub-agent.
            agent_name: Name of the sub-agent (reuse to continue a conversation).
            tool_sets: Tool set names to provide (e.g. ["FileRead", "Grep"]). Defaults to all available.
            model_alias: Model alias to use. Defaults to first available model.
        """
        import agents as agents_lib
        from agents import Agent

        from vero.utils.openai_agents import run_agent_with_json_sanitization

        if model_alias is None:
            model_alias = next(iter(self.models.keys()))

        available = self.available_tool_sets

        # Resolve tool_sets by name
        tools: list[Tool] = []
        if tool_sets:
            name_to_key = {
                k.__name__ if hasattr(k, "__name__") else str(k): k for k in available
            }
            for name in tool_sets:
                if name not in name_to_key:
                    raise ValueError(
                        f"Unknown tool set: {name}. Available: {list(name_to_key.keys())}"
                    )
                tools.extend(available[name_to_key[name]])
        else:
            for tool_list in available.values():
                tools.extend(tool_list)

        try:
            model = self.models[model_alias]
        except KeyError:
            raise KeyError(
                f"Invalid model alias: {model_alias}. Available: {list(self.models.keys())}"
            )

        if (
            agent_name is not None
            and agent_name in self.sessions
            and self.sessions[agent_name].session is not None
        ):
            previous = self.sessions[agent_name].session
            if previous:
                input = previous.to_input_list() + [{"content": prompt, "role": "user"}]
            else:
                input = [{"content": prompt, "role": "user"}]

        elif agent_name is not None:
            input = prompt
        else:
            agent_name = f"Sub-Agent {len(self.sessions) + 1}"
            input = prompt

        if agent_name in self.sessions:
            subagent = self.sessions[agent_name].agent
            subagent.instructions = instructions or subagent.instructions
        else:
            subagent = Agent(
                name=agent_name,
                model=model,
                tools=tools,
                instructions=instructions,
            )

        result, error = await run_agent_with_json_sanitization(
            subagent,
            input,
            max_turns=max_turns,
            sanitize_invalid_json=True,
            max_retries=MAX_SANITIZATION_RETRIES,
        )
        self.sessions[agent_name] = SubAgentSession(agent=subagent, session=result)

        if isinstance(error, StreamEventTimeout):
            return f"Sub-agent '{agent_name}' timed out. Use the same agent name to continue."
        elif isinstance(error, agents_lib.MaxTurnsExceeded):
            return f"Sub-agent '{agent_name}' reached max turns ({max_turns}). Use the same agent name to continue."
        elif error is not None:
            raise error

        return f"Sub-agent '{agent_name}' response:\n\n{result.final_output}"
