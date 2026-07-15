from .base import BaseAgent  # noqa: E402
from .producer import AgentCandidateProducer
from .protocol import AgentContext, AgentRunResult, CodingAgent

__all__ = [
    "BaseAgent",
    "AgentCandidateProducer",
    "AgentContext",
    "AgentRunResult",
    "CodingAgent",
]


def __getattr__(name: str):
    if name == "ClaudeCodeAgent":
        from .claude_code import ClaudeCodeAgent

        return ClaudeCodeAgent
    if name == "VeroAgent":
        from .vero import VeroAgent

        return VeroAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
