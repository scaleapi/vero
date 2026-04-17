"""VeroAgent - Agent backend using OpenAI Agents SDK for optimization."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from agents import (
    Agent,
    AgentsException,
    FunctionTool,
    ModelRetrySettings,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    RunContextWrapper,
    RunResultStreaming,
    TResponseInputItem,
    retry_policies,
    set_tracing_disabled,
)
from agents.extensions.models.litellm_model import LitellmModel
from agents.lifecycle import AgentHooks
from pydantic import BaseModel

from vero.agents.base import BaseAgent
from vero.agents.events import AgentEvent
from vero.session import Session
from vero.tools import (
    BashTool,
    DatasetViewer,
    ExperimentRunnerTool,
    ExperimentViewer,
    FileRead,
    FileWrite,
    GitControl,
    GitViewer,
    Grep,
    SubAgentTool,
    TodoList,
    WebFetch,
    WebSearch,
    think,
)
from vero.tools.base import ToolSet
from vero.tools.utils.openai_agents import (
    callable_to_oai_tool,
    tool_set_instance_to_oai_tools,
)
from vero.utils import recursively_serialize
from vero.utils.openai_agents import (
    run_agent_with_json_sanitization,
    strict_mode_from_model,
)
from vero.utils.tokens import get_token_count

logger = logging.getLogger(__name__)

set_tracing_disabled(True)


class MaxTokenCountExceededError(AgentsException):
    """Raised when the total token count exceeds the max token count."""

    pass


def _raise_agents_exception(msg: str):
    raise AgentsException(msg)


def default_tool_sets() -> list[ToolSet | object | Callable]:
    """Default tools for the VeroAgent."""

    return [
        BashTool(),
        DatasetViewer(),
        ExperimentRunnerTool(on_fatal=_raise_agents_exception),
        ExperimentViewer(),
        FileRead(),
        FileWrite(),
        GitControl(),
        GitViewer(),
        Grep(),
        SubAgentTool(),
        TodoList(),
        think,
        WebFetch(),
        WebSearch(),
    ]


class VeroAgentHook(AgentHooks):
    """Hook for the VeroAgent that manages token limits."""

    def __init__(self, agent: VeroAgent):
        self.vero_agent = agent

    async def on_llm_start(
        self,
        context: RunContextWrapper,
        agent: Agent,
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        """Triggered when the LLM starts a new turn."""

        if isinstance(agent.model, (LitellmModel, OpenAIChatCompletionsModel)):
            model = agent.model.model
        elif agent.model is None:
            model = ""
        elif isinstance(agent.model, str):
            model = agent.model
        else:
            raise ValueError(f"Unexpected type for model: {type(agent.model)}")

        try:
            token_count = get_token_count(result=input_items, model=model)
            logger.info(f"Input token count: {token_count}")
        except Exception as e:
            logger.error(f"Error getting token count: {e}")
            if self.vero_agent.max_tokens is not None:
                raise e
            else:
                return

        if (
            self.vero_agent.max_tokens is not None
            and token_count > self.vero_agent.max_tokens
        ):
            raise MaxTokenCountExceededError(
                f"Total token count {token_count} exceeds max token count {self.vero_agent.max_tokens} for model {agent.model}"
            )


def _default_oai_agent() -> Agent:
    import os

    litellm_kwargs = {}
    if os.getenv("LITELLM_API_KEY") and os.getenv("LITELLM_BASE_URL"):
        litellm_kwargs["api_key"] = os.getenv("LITELLM_API_KEY")
        litellm_kwargs["base_url"] = os.getenv("LITELLM_BASE_URL")

        if litellm_kwargs["base_url"]:
            litellm_kwargs["base_url"] = litellm_kwargs["base_url"].rstrip("/")

    model = LitellmModel(model="anthropic/claude-sonnet-4-5-20250929", **litellm_kwargs)

    return Agent(
        name="VeroAgent",
        model=model,
        model_settings=ModelSettings(
            include_usage=True,
            max_tokens=None,
            retry=ModelRetrySettings(
                max_retries=3,
                backoff={
                    "initial_delay": 1.0,
                    "max_delay": 60.0,
                    "multiplier": 2.0,
                    "jitter": True,
                },
                policy=retry_policies.any(
                    retry_policies.provider_suggested(),
                    retry_policies.retry_after(),
                    retry_policies.network_error(),
                    retry_policies.http_status([408, 429, 500, 502, 503, 504]),
                ),
            ),
        ),
    )


@dataclass
class VeroAgent(BaseAgent):
    """Agent backend using the OpenAI Agents SDK (Vero's agentic optimizer)."""

    oai_agent: Agent = field(default_factory=_default_oai_agent)
    tool_sets: list[ToolSet | object | Callable] = field(
        default_factory=default_tool_sets
    )
    max_tokens: int | None = None
    max_retries: int = 3
    event_timeout: int | None = 60 * 12
    state: list[TResponseInputItem] | None = field(default=None, repr=False)

    _session: Session | None = field(default=None, repr=False)
    _tools: dict[type | Callable, list[FunctionTool]] = field(
        default_factory=dict, repr=False
    )
    _agent_hook: VeroAgentHook | None = field(default=None, repr=False)
    _run_result: RunResultStreaming | None = field(default=None, repr=False)

    @property
    def trace(self) -> list[TResponseInputItem] | None:
        return self.state

    @trace.setter
    def trace(self, value: Any) -> None:
        self.state = value

    def init(self, session: Session) -> None:
        """Initialize the agent with a Session context."""
        self._session = session
        for tool_set in self.tool_sets:
            if isinstance(tool_set, ExperimentRunnerTool):
                tool_set.on_fatal = _raise_agents_exception
        self._tools = self._create_tools(session)

    async def step(
        self,
        input: str | list[TResponseInputItem] | None,
        max_turns: int = 200,
        on_event: Callable | None = None,
        **kwargs,
    ) -> RunResultStreaming | None:
        """Execute optimization steps using the OpenAI Agents SDK."""

        assert self._session, "Session is not set! Call init(session) first."

        state = self.state if self.state is not None else []

        if isinstance(input, str):
            inputs = state + [{"role": "user", "content": input}]
        elif isinstance(input, list):
            inputs = state + input
        elif input is None:
            inputs = state
        else:
            raise ValueError(f"Got unexpected type for inputs: {type(input)}")

        if not inputs:
            raise ValueError("No input provided and no state to resume from")

        agent = self._create_agent()
        run_config = RunConfig(
            workflow_name=f"vero::{self._session.session_id}",
            trace_id=self._session.session_id,
        )

        self._run_result, error = await run_agent_with_json_sanitization(
            agent=agent,
            input=inputs,
            max_turns=max_turns,
            sanitize_invalid_json=True,
            max_retries=self.max_retries,
            run_config=run_config,
            event_timeout=self.event_timeout,
            on_event=on_event,
        )
        if self._run_result:
            self.state = self._run_result.to_input_list()

        if isinstance(error, Exception):
            raise error

        return self._run_result

    def serialize_event(self, event: Any) -> AgentEvent | None:
        """Convert an OpenAI Agents SDK StreamEvent to a normalized AgentEvent.

        Returns None for noise events (raw_response_event, etc.) that
        shouldn't be dispatched to callbacks.
        """
        if not hasattr(event, "type") or event.type != "run_item_stream_event":
            return None

        item = event.item.raw_item
        raw = item.model_dump() if isinstance(item, BaseModel) else (item if isinstance(item, dict) else {})
        item_type = raw.get("type", "")

        if item_type == "message":
            parts = []
            for block in raw.get("content", []):
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        parts.append(text)
            return {"kind": "message", "text": "\n".join(parts)} if parts else None

        if item_type == "function_call":
            return {
                "kind": "tool_call",
                "name": raw.get("name", "unknown"),
                "args": raw.get("arguments", ""),
            }

        if item_type == "function_call_output":
            output = raw.get("output", "")
            return {
                "kind": "tool_result",
                "name": raw.get("name", ""),
                "output": output,
                "is_error": isinstance(output, str) and "error" in output.lower()[:80],
            }

        return None

    def serialize_trace(self) -> list[TResponseInputItem] | None:
        """Serialize the execution trace. For VeroAgent, this is the conversation history."""
        return self.trace

    def serialize_state(self) -> list[TResponseInputItem] | None:
        """Serialize state for resumption. Same as trace for VeroAgent."""
        return self.state

    def deserialize_state(self, state: list[TResponseInputItem]) -> None:
        """Restore state from conversation history."""
        self.state = state

    def _get_sub_agent_tool(self) -> SubAgentTool | None:
        """Get the SubAgentTool instance if present."""
        for tool_set in self.tool_sets:
            if isinstance(tool_set, SubAgentTool):
                return tool_set
        return None

    def model_str(self) -> str | None:
        if isinstance(self.oai_agent.model, (LitellmModel, OpenAIChatCompletionsModel)):
            return self.oai_agent.model.model
        elif isinstance(self.oai_agent.model, str):
            return self.oai_agent.model
        return None

    def usage(self) -> dict:
        """Return usage statistics."""

        if self._run_result is None:
            return {}

        orchestrator_usage = self._run_result.context_wrapper.usage
        sub_agent_usages = {}
        sub_agent_tool = self._get_sub_agent_tool()
        if sub_agent_tool:
            sub_agent_usages = {
                name: recursively_serialize(sa.session.context_wrapper.usage)  # type: ignore
                for name, sa in sub_agent_tool.sessions.items()
                if sa.session is not None
            }
        return {
            "orchestrator": recursively_serialize(orchestrator_usage),  # type: ignore
            "sub_agents": sub_agent_usages,
        }

    def dict(self) -> dict:
        """Return VeroAgent-specific fields for Policy.as_dict()."""
        return {
            "model": self.model_str,
            "tool_sets": [
                type(ts).__name__ if hasattr(ts, "__class__") else ts.__name__
                for ts in self.tool_sets
            ],
            "model_settings": recursively_serialize(self.oai_agent.model_settings),  # type: ignore
        }

    def artifacts(self) -> dict:
        try:
            sub_agent_tool = self._get_sub_agent_tool()
            if sub_agent_tool and sub_agent_tool.sessions:
                sub_agent_dicts = {
                    key: sa.to_dict() for key, sa in sub_agent_tool.sessions.items()
                }
                return {"subagents": sub_agent_dicts}
        except Exception as e:
            logger.warning(f"Failed to get sub agent dicts: {e}")

        return {}

    @property
    def strict_mode(self) -> bool:
        if self.oai_agent.model is None:
            return False
        return strict_mode_from_model(self.oai_agent.model)

    @property
    def orchestrator_tool_sets(self) -> list[type]:
        import inspect

        return [type(ts) for ts in self.tool_sets if not inspect.isfunction(ts)]

    @property
    def sub_agent_tool_sets(self) -> list[type]:
        for ts in self.tool_sets:
            if isinstance(ts, SubAgentTool):
                return [type(t) for t in ts.allowed_tools if not callable(t)]
        return []

    @property
    def sub_agents_enabled(self) -> bool:
        return any(isinstance(ts, SubAgentTool) for ts in self.tool_sets)

    def _create_agent(self) -> Agent:
        """Create the OAI Agent by augmenting the template with vero tools and hooks."""
        tools = self._get_tools() + list(self.oai_agent.tools)

        instructions = None
        if self._session:
            instructions = self._session.instructions

        return Agent(
            name=self.oai_agent.name,
            model=self.oai_agent.model,
            model_settings=self.oai_agent.model_settings,
            output_type=self.oai_agent.output_type,
            tools=tools,
            instructions=instructions or self.oai_agent.instructions,
            hooks=self._get_agent_hook(),
        )

    def _get_agent_hook(self) -> VeroAgentHook:
        if self._agent_hook is None:
            self._agent_hook = VeroAgentHook(agent=self)
        return self._agent_hook

    def _get_tools(self) -> list[FunctionTool]:
        tools = []
        for tool_list in self._tools.values():
            tools.extend(tool_list)
        tools.sort(key=lambda tool: tool.name)
        return tools

    def tool_set_enabled(
        self, tool_set_key: type | str, for_sub_agent: bool = False
    ) -> bool:
        """Check if a tool set is enabled. Accepts class or string name."""
        if isinstance(tool_set_key, str):
            return any(
                type(ts).__name__ == tool_set_key
                or (callable(ts) and getattr(ts, "__name__", "") == tool_set_key)
                for ts in self.tool_sets
            )
        return any(
            type(ts) is tool_set_key or ts is tool_set_key for ts in self.tool_sets
        )

    def _create_tools(
        self, session: Session
    ) -> dict[type | Callable, list[FunctionTool]]:
        instances: dict[type | Callable, list[FunctionTool]] = {}
        sub_agent_tool = None

        for ts in self.tool_sets:
            # Skip SubAgentTool — bind it last
            if isinstance(ts, SubAgentTool):
                sub_agent_tool = ts
                continue

            # Bind if it's a ToolSet
            if hasattr(ts, "bind"):
                ts.bind(session)  # type: ignore

            # Convert to list of FunctionTools
            if inspect.isfunction(ts):
                # Plain callable like think
                instances[ts] = [callable_to_oai_tool(ts, strict_mode=self.strict_mode)]
            else:
                instances[type(ts)] = self._to_openai_function_tools(ts)

        # Bind SubAgentTool last (needs other tool instances)
        if sub_agent_tool is not None:
            sub_agent_tool.tool_instances = instances
            model = self.oai_agent.model
            model_name = model if isinstance(model, str) else model.model
            sub_agent_tool.models = {model_name: model}
            instances[SubAgentTool] = self._to_openai_function_tools(sub_agent_tool)

        return instances

    def _to_openai_function_tools(self, instance: object) -> list:
        strict_mode = self.strict_mode
        if inspect.isfunction(instance):
            return [callable_to_oai_tool(instance, strict_mode=strict_mode)]
        else:
            return tool_set_instance_to_oai_tools(instance, strict_mode=strict_mode)
