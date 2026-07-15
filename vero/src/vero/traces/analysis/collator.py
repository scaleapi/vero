from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk.types import (
    AssistantMessage,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from openai.types.responses import ResponseFunctionToolCall, ResponseInputItem
from openai.types.responses.response_input_item import FunctionCallOutput
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, TypeAdapter

from vero.core.db.candidate import Candidate
from vero.core.sessions import get_vero_home_dir
from vero.evaluation import EvaluationDatabase, EvaluationRecord
from vero.workspace.git import GitWorkspace

# =============================================================================
# Types
# =============================================================================

GitCommitHistory = list[Candidate]
OpenAITrace = list[ResponseInputItem]
AnthropicTrace = list[Message]
Trace = OpenAITrace | AnthropicTrace
_OpenAITraceAdapter = TypeAdapter(OpenAITrace)
_AnthropicTraceAdapter = TypeAdapter(AnthropicTrace)


def _is_conversation_item(item: dict) -> bool:
    """Check if a raw item is a conversation message (not a system/result event).

    ClaudeCodeAgent result.json contains SystemMessage items (hook events)
    and ResultMessage items mixed with actual conversation messages. These
    non-conversation items have a 'subtype' field.
    """
    return "subtype" not in item


def _parse_anthropic_item(item: dict) -> AssistantMessage | UserMessage | None:
    """Parse a single raw dict into a claude-agent-sdk message type.

    The SDK types are dataclasses (not pydantic), so we reconstruct them
    manually rather than relying on TypeAdapter validation which fails on
    nested content block shapes.
    """
    content = item.get("content")
    if not isinstance(content, list) or not content:
        return None

    first_block = content[0]
    if not isinstance(first_block, dict):
        return None

    # AssistantMessage: has 'model' field, content blocks are thinking/text/tool_use
    if "model" in item:
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if "thinking" in block:
                blocks.append(ThinkingBlock(**block))
            elif "text" in block:
                blocks.append(TextBlock(**block))
            elif "id" in block and "name" in block:
                blocks.append(ToolUseBlock(**block))
        return AssistantMessage(
            content=blocks,
            model=item.get("model"),
            message_id=item.get("message_id"),
            stop_reason=item.get("stop_reason"),
            session_id=item.get("session_id"),
            uuid=item.get("uuid"),
            parent_tool_use_id=item.get("parent_tool_use_id"),
            usage=item.get("usage"),
            error=item.get("error"),
        )

    # UserMessage: content blocks are tool results (have tool_use_id)
    if "tool_use_id" in first_block:
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                continue
            blocks.append(
                ToolResultBlock(
                    tool_use_id=block.get("tool_use_id", ""),
                    content=block.get("content"),
                    is_error=block.get("is_error"),
                )
            )
        return UserMessage(
            content=blocks,
            uuid=item.get("uuid"),
            parent_tool_use_id=item.get("parent_tool_use_id"),
            tool_use_result=item.get("tool_use_result"),
        )

    return None


def parse_trace(raw: list[dict]) -> Trace:
    """Parse raw result.json into typed trace, handling both agent formats.

    VeroAgent stores OpenAI ResponseInputItem lists.
    ClaudeCodeAgent stores claude_agent_sdk Message lists, mixed with
    SystemMessage/ResultMessage events that must be filtered out.
    """
    filtered = [item for item in raw if _is_conversation_item(item)]
    if not filtered:
        return []

    # Detect format: Anthropic messages have 'model' or 'tool_use_id' in content,
    # OpenAI items have 'type' or 'role' at top level
    first = filtered[0]
    if "model" in first or (
        isinstance(first.get("content"), list)
        and first["content"]
        and isinstance(first["content"][0], dict)
        and "tool_use_id" in first["content"][0]
    ):
        # Anthropic format (ClaudeCodeAgent)
        trace: AnthropicTrace = []
        for item in filtered:
            parsed = _parse_anthropic_item(item)
            if parsed is not None:
                trace.append(parsed)
        return trace

    # OpenAI format (VeroAgent)
    return _OpenAITraceAdapter.validate_python(filtered)


def _dump_trace(trace: Trace) -> list[dict[str, Any]]:
    """Serialize a parsed trace back to JSON-compatible dicts."""
    if not trace:
        return []
    first = trace[0]
    if isinstance(first, (AssistantMessage, UserMessage)):
        from dataclasses import asdict

        return [asdict(item) for item in trace]
    return _OpenAITraceAdapter.dump_python(trace, mode="json")


@dataclass
class TraceSegment:
    """A segment of trace items ending with a commit change."""

    start_idx: int
    end_idx: int
    commit: str | None = None


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str | None = None

    @classmethod
    def from_openai_tool_call(cls, item: ResponseFunctionToolCall) -> ToolCall:
        try:
            args = json.loads(item.arguments)
        except json.JSONDecodeError:
            args = {}
        return cls(
            name=item.name,
            arguments=args,
            id=item.call_id,
        )

    @classmethod
    def from_anthropic_tool_call(cls, item: ToolUseBlock) -> ToolCall:
        return cls(
            name=item.name,
            arguments=item.input if isinstance(item.input, dict) else {},
            id=item.id,
        )

    @classmethod
    def from_openai_span(cls, item: ResponseInputItem) -> ToolCall | None:
        if isinstance(item, ResponseFunctionToolCall):
            return cls.from_openai_tool_call(item)
        return None

    @classmethod
    def from_anthropic_span(
        cls, item: Message, return_all: bool = False
    ) -> ToolCall | list[ToolCall] | None:
        if isinstance(item, AssistantMessage):
            blocks = []
            for block in item.content:
                if isinstance(block, ToolUseBlock):
                    blocks.append(cls.from_anthropic_tool_call(block))

            if return_all:
                return blocks

            assert len(blocks) == 1, (
                "Expected exactly one tool use block in assistant message"
            )
            return blocks[0]
        return None


@dataclass
class ToolResult:
    call_id: str
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None

    @classmethod
    def from_openai_tool_result(cls, item: FunctionCallOutput) -> ToolResult:
        return cls(
            call_id=item.call_id,
            content=item.output,
        )

    @classmethod
    def from_anthropic_tool_result(cls, item: ToolResultBlock) -> ToolResult:
        return cls(
            call_id=item.tool_use_id,
            content=item.content,
            is_error=item.is_error,
        )

    @classmethod
    def from_openai_span(cls, item: ResponseInputItem) -> ToolResult | None:
        if isinstance(item, FunctionCallOutput):
            return cls.from_openai_tool_result(item)
        return None

    @classmethod
    def from_anthropic_span(
        cls, item: Message, return_all: bool = False
    ) -> ToolResult | list[ToolResult] | None:
        if isinstance(item, UserMessage):
            blocks = []
            for block in item.content:
                if isinstance(block, ToolResultBlock):
                    blocks.append(cls.from_anthropic_tool_result(block))
            if return_all:
                return blocks
            assert len(blocks) == 1, (
                "Expected exactly one tool result block in assistant message"
            )
            return blocks[0]
        return None


# =============================================================================
# Git Commit History
# =============================================================================


async def get_commit_history(
    workspace: GitWorkspace, final_commit: str, initial_commit: str
) -> GitCommitHistory:
    """Get the list of commits from initial_commit to final_commit (inclusive).

    Args:
        workspace: The GitWorkspace to query
        final_commit: The ending commit hash
        initial_commit: The starting commit hash

    Returns:
        List of Candidate objects representing the commit history, ordered from oldest to newest
    """
    # Use initial_commit^..final_commit to include initial_commit.
    # If initial_commit is the root (no parent), fall back to just final_commit.
    try:
        log_output = await workspace._git(
            "log",
            "--reverse",
            "--format=%H|%s|%ct",
            f"{initial_commit}^..{final_commit}",
        )
    except Exception:
        # Root commit has no parent — use --ancestry-path instead
        log_output = await workspace._git(
            "log",
            "--reverse",
            "--format=%H|%s|%ct",
            "--ancestry-path",
            f"{initial_commit}..{final_commit}",
        )
        # Prepend the initial commit itself
        initial_line = await workspace._git(
            "log",
            "--format=%H|%s|%ct",
            "-1",
            initial_commit,
        )
        if initial_line.strip():
            log_output = initial_line.strip() + "\n" + log_output

    candidates = []
    lines = log_output.strip().split("\n") if log_output.strip() else []

    prev_commit = None
    for line in lines:
        if not line:
            continue
        parts = line.split("|", 2)
        commit_hash = parts[0]
        message = parts[1] if len(parts) > 1 else None

        candidate = Candidate(
            commit=commit_hash,
            repo_name=workspace.name,
            parent_commit=prev_commit,
            message=message,
        )
        candidates.append(candidate)
        prev_commit = commit_hash

    return candidates


def commits_match(commit_a: str, commit_b: str) -> bool:
    """Check if two commits match, handling 8-char truncation."""
    if commit_a == commit_b:
        return True

    min_len = min(len(commit_a), len(commit_b))
    if min_len >= 7:  # Git short hash is typically 7-8 chars
        return commit_a[:min_len] == commit_b[:min_len]
    return False


async def get_commit_diff(
    workspace: GitWorkspace, candidate: Candidate, other: str | None = None
) -> str:
    """Get the diff between a commit and another commit.

    Args:
        workspace: The GitWorkspace to query
        candidate: The commit to get the diff for

    Returns:
        The git diff output as a string
    """
    commit = candidate.commit
    if other is None:
        other = candidate.parent_commit or f"{commit}^"
    try:
        return await workspace._git("diff", other, commit)
    except Exception:
        # Root commit has no parent — show the full commit diff
        return await workspace._git("show", "--format=", commit)


# =============================================================================
# Trace Parsing Utilities
# =============================================================================


class TraceUtils:
    @classmethod
    def iter_tool_calls(cls, trace: Trace) -> Iterator[tuple[int, ToolCall]]:
        for i, item in enumerate(trace):
            if isinstance(item, ResponseFunctionToolCall):
                tool_call = ToolCall.from_openai_span(item)
                yield i, tool_call

            if isinstance(item, AssistantMessage):
                tool_calls = ToolCall.from_anthropic_span(item, return_all=True)
                for tool_call in tool_calls:
                    yield i, tool_call

    @classmethod
    def iter_tool_results(cls, trace: Trace) -> Iterator[tuple[int, ToolResult]]:
        for i, item in enumerate(trace):
            if isinstance(item, FunctionCallOutput):
                tool_result = ToolResult.from_openai_span(item)
                if tool_result:
                    yield i, tool_result
            if isinstance(item, UserMessage):
                tool_results = ToolResult.from_anthropic_span(item, return_all=True)
                if tool_results:
                    for tool_result in tool_results:
                        yield i, tool_result

    @staticmethod
    def extract_commit_from_response(response: str) -> str | None:
        """Extract commit hash from a tool response string."""

        patterns: list[re.Pattern[str]] = [
            re.compile(r"New commit: ([a-f0-9]{7,40})", re.IGNORECASE),  # GitControl
            re.compile(
                r"Created a new commit ([a-f0-9]{7,40})", re.IGNORECASE
            ),  # FileWrite
            re.compile(
                r"Created commit ([a-f0-9]{7,40})", re.IGNORECASE
            ),  # ResourceControl (8-char)
            re.compile(
                r"^([a-f0-9]{7,40})\s+Committing changes from command:", re.MULTILINE
            ),  # Git Log
            re.compile(
                r"\bCommit:\s*([a-f0-9]{7,40})", re.IGNORECASE
            ),  # Sub-agent response
            re.compile(
                r'"candidate_commit"\s*:\s*"([a-f0-9]{7,40})"'
            ),  # Experiment result JSON
        ]

        for pattern in patterns:
            if match := pattern.search(response):
                return match.group(1)

        return None

    @staticmethod
    def extract_commits_from_git_output(response: str) -> list[str]:
        """Extract all commit hashes from git log/status output (Claude Code style).

        Parses output like:
        7dc50e44 Committing changes from command: Edit to file: ...
        2d355513 Committing changes from command: Edit to ...

        Returns list of commit hashes in order they appear (newest first in git log).
        """
        pattern = re.compile(
            r"^([a-f0-9]{7,40})\s+Committing changes from command:", re.MULTILINE
        )
        return pattern.findall(response)

    @staticmethod
    def iter_commit_changes(trace: Trace) -> Iterator[tuple[int, str]]:
        """Iterate over all tool results that contain commit hashes.

        Extracts commits from any tool response containing commit patterns.
        Yields (trace_idx, commit_hash) pairs.
        """
        seen_commits: set[str] = set()

        for idx, tool_result in TraceUtils.iter_tool_results(trace):
            # Extract string content from tool result
            content = tool_result.content
            if not isinstance(content, str):
                continue

            commit = TraceUtils.extract_commit_from_response(content)
            if commit and commit not in seen_commits:
                seen_commits.add(commit)
                yield idx, commit
                continue

            commits = TraceUtils.extract_commits_from_git_output(content)
            for c in commits:
                if c not in seen_commits:
                    seen_commits.add(c)
                    yield idx, c

    @staticmethod
    def build_trace_segments(trace: Trace) -> list[TraceSegment]:
        """Build segments of trace items, each ending at a commit change."""
        segments: list[TraceSegment] = []

        prev_end = 0
        for idx, commit in TraceUtils.iter_commit_changes(trace):
            end_idx = idx + 1
            segments.append(
                TraceSegment(start_idx=prev_end, end_idx=end_idx, commit=commit)
            )
            prev_end = end_idx

        # Add final span if there's remaining trace
        if prev_end < len(trace):
            segments.append(
                TraceSegment(start_idx=prev_end, end_idx=len(trace), commit=None)
            )

        return segments


# =============================================================================
# Data Models
# =============================================================================


class OptimizationPhase(BaseModel):
    """An optimization phase ending with at least one evaluation."""

    commits: GitCommitHistory = Field(
        description="Commits made during this phase", repr=False
    )
    final_commit: Candidate = Field(description="Final commit that was evaluated")
    evaluations: list[EvaluationRecord] = Field(
        default_factory=list,
        validation_alias=AliasChoices("evaluations", "experiments"),
        description="Evaluations run during this phase",
        repr=False,
    )
    trace_segments: list[TraceSegment] = Field(
        default_factory=list, description="Trace segments for this phase", repr=False
    )
    is_initial: bool = Field(description="Whether this phase is the initial phase")

    def summary(self) -> dict[str, Any]:
        """Get a summary of the phase."""
        return {
            "commits": [
                {"commit": x.commit, "message": x.message} for x in self.commits
            ],
            "final_commit": {
                "commit": self.final_commit.commit,
                "message": self.final_commit.message,
            },
            "is_initial": self.is_initial,
        }

    def contains_commit(self, commit: str) -> bool:
        """Check if this phase contains a commit."""
        return any(
            commits_match(candidate.commit, commit) for candidate in self.commits
        )

    @property
    def earliest_span_idx(self) -> int:
        """Index of the earliest trace segment in this phase."""
        if not self.trace_segments:
            return 0
        return min(segment.start_idx for segment in self.trace_segments)

    @property
    def latest_span_idx(self) -> int:
        """Index of the latest trace segment in this phase."""
        if not self.trace_segments:
            return -1
        return max(segment.end_idx for segment in self.trace_segments)

    @property
    def num_trace_items(self) -> int:
        """Number of trace items in this phase."""
        return self.latest_span_idx - self.earliest_span_idx

    def get_experiment_scores(self) -> list[dict[str, Any]]:
        """Return the legacy trace-analysis score export."""
        scores = []
        for evaluation in self.evaluations:
            evaluation_set = evaluation.request.evaluation_set
            scores.append(
                {
                    "experiment_id": evaluation.id,
                    "commit": evaluation.request.candidate.commit,
                    "dataset": evaluation_set.name,
                    "split": evaluation_set.partition,
                    "mean_score": evaluation.objective.value
                    if evaluation.objective is not None
                    else evaluation.report.metrics.get("score"),
                    "error_rate": evaluation.report.metrics.get("error_rate"),
                    "num_samples": len(evaluation.report.cases),
                }
            )
        return scores

    @property
    def experiments(self) -> list[EvaluationRecord]:
        """Deprecated trace-analysis alias for ``evaluations``."""
        return self.evaluations


class SubAgentInfo(BaseModel):
    """Information about a sub-agent."""

    name: str
    instructions: str | None = None
    tools: list[str] = Field(default_factory=list)


class SubAgentTrace(BaseModel):
    """Trace data for a single sub-agent invocation."""

    agent: SubAgentInfo
    session: OpenAITrace = Field(default_factory=list)

    @property
    def num_turns(self) -> int:
        """Number of turns in the sub-agent session."""
        return len(self.session)

    def get_tool_calls(self) -> list[ToolCall]:
        """Get all tool calls from the sub-agent session."""
        return [tool_call for _, tool_call in TraceUtils.iter_tool_calls(self.session)]


class SessionConfig(BaseModel):
    """Configuration from a session's config.json.

    Accepts extra fields from agent-specific config (e.g. claude_agent_options,
    tool_sets, model_settings) without failing validation.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str
    # Accept both base_commit (old sessions) and base_version (new sessions)
    base_commit: str = Field(
        default="",
        validation_alias=AliasChoices("base_commit", "base_version"),
    )
    base_branch: str | None = None
    current_commit: str | None = None
    instructions: str | None = None
    split_accesses: list[dict[str, str]] = Field(default_factory=list)
    budget: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def final_commit(self) -> str:
        """The ending commit — current_commit at time of save, or base_commit as fallback."""
        return self.current_commit or self.base_commit

    def get_model(self) -> str | None:
        """Extract model name from agent-specific config fields."""
        # VeroAgent stores model at top level (via extra fields)
        extra = self.__pydantic_extra__ or {}
        if "model" in extra:
            return extra["model"]
        # ClaudeCodeAgent stores it in claude_agent_options
        opts = extra.get("claude_agent_options")
        if isinstance(opts, dict):
            return opts.get("model")
        return None


# =============================================================================
# Main Classes
# =============================================================================


class TraceAnalysisPayload(BaseModel):
    """Complete payload for trace analysis."""

    session_id: str
    config: SessionConfig
    phases: list[OptimizationPhase] = Field(default_factory=list)
    evaluations: list[EvaluationRecord] = Field(
        default_factory=list,
        validation_alias=AliasChoices("evaluations", "experiments"),
    )
    agent_trace: Trace = Field(default_factory=list)
    sub_agents_trace: dict[str, SubAgentTrace] = Field(default_factory=dict)

    @classmethod
    async def from_session_id(
        cls,
        session_id: str,
        sessions_dir: Path | str | None = None,
        project_path: Path | str | None = None,
    ) -> TraceAnalysisPayload:
        """Load a TraceAnalysisPayload from a session directory.

        Args:
            session_id: The session UUID
            sessions_dir: Optional custom sessions directory (defaults to ~/.vero/sessions)
            project_path: Path to the project/repo for commit history (required)

        Returns:
            TraceAnalysisPayload populated from session files
        """
        sessions_dir = Path(sessions_dir) if sessions_dir else (get_vero_home_dir() / "sessions")
        session_path = sessions_dir / session_id

        if not session_path.exists():
            raise FileNotFoundError(f"Session directory not found: {session_path}")

        # Load config.json
        config_path = session_path / "config.json"
        with open(config_path) as f:
            config_data = json.load(f)
        config = SessionConfig.model_validate(config_data)

        # Load result.json (agent trace)
        result_path = session_path / "result.json"
        raw_trace: list[dict[str, Any]] = []
        if result_path.exists():
            with open(result_path) as f:
                raw_trace = json.load(f)
        agent_trace = parse_trace(raw_trace)

        # Load database.json
        database: EvaluationDatabase | None = None
        db_path = session_path / "database.json"
        if db_path.exists():
            database = EvaluationDatabase.load_from_file(db_path)

        if not database:
            raise ValueError("database.json is required to build phases")

        # Load sub_agents.json if present (dict mapping agent_name -> trace)
        sub_agents_trace: dict[str, SubAgentTrace] = {}
        sub_agents_path = session_path / "sub_agents.json"
        if sub_agents_path.exists():
            with open(sub_agents_path) as f:
                sub_agents_trace = json.load(f)

        for agent_name, value in sub_agents_trace.items():
            sub_agents_trace[agent_name] = SubAgentTrace.model_validate(value)

        # Create GitWorkspace from project_path (required)
        if not project_path:
            raise ValueError(
                "project_path is required to build phases from commit history"
            )
        workspace = await GitWorkspace.create(str(project_path))

        # Determine final commit: prefer config, fall back to latest evaluated commit,
        # then git branch tip. config.current_commit may equal base_commit if saved early.
        final_commit = config.final_commit
        if final_commit == config.base_commit and database:
            evaluations = database.get_evaluations()
            if evaluations:
                final_commit = evaluations[-1].request.candidate.commit
        if final_commit == config.base_commit:
            # Last resort: use the branch tip
            try:
                final_commit = await workspace.current_version()
            except Exception:
                pass

        # Build phases from commit history
        phases = await cls._build_phases(
            workspace=workspace,
            database=database,
            base_commit=config.base_commit,
            final_commit=final_commit,
            agent_trace=agent_trace,
        )

        return cls(
            session_id=session_id,
            config=config,
            phases=phases,
            evaluations=database.get_evaluations(),
            agent_trace=agent_trace,
            sub_agents_trace=sub_agents_trace,
        )

    @classmethod
    async def _build_phases(
        cls,
        workspace: GitWorkspace,
        database: EvaluationDatabase,
        base_commit: str,
        final_commit: str,
        agent_trace: Trace,
    ) -> list[OptimizationPhase]:
        """Build phases from commit history (source of truth).

        Algorithm:
        1. Get full commit history from base_commit to final_commit
        2. Build evaluations_by_commit dict from database
        3. Cut commit history at evaluated commits to define phases
        4. Correlate trace spans to phases using two-pointer algorithm
        """
        # Step 1: Get commit history
        commit_history = await get_commit_history(workspace, final_commit, base_commit)
        if not commit_history:
            return []

        # Step 2: Build evaluations by commit
        evaluations_by_commit: dict[str, list[EvaluationRecord]] = {}
        for evaluation in database.get_evaluations():
            commit = evaluation.request.candidate.commit
            if commit not in evaluations_by_commit:
                evaluations_by_commit[commit] = []
            evaluations_by_commit[commit].append(evaluation)

        # Step 3: Build trace segments and correlate to phases
        trace_segments = TraceUtils.build_trace_segments(agent_trace)
        trace_segments = trace_segments[:-1]

        # Step 4: Cut commit history at evaluated commits
        # Each phase is a sequence of commits ending with an evaluated commit
        current_phase_commits: GitCommitHistory = []
        phases: list[OptimizationPhase] = []

        for candidate in commit_history:
            current_phase_commits.append(candidate)
            if candidate.commit in evaluations_by_commit:
                # End of phase - this commit was evaluated

                # Check if the base commit is in the current phase commits
                is_initial = base_commit in [
                    commit.commit for commit in current_phase_commits
                ]

                phase = OptimizationPhase(
                    is_initial=is_initial,
                    commits=current_phase_commits,
                    final_commit=candidate,
                )

                phase_trace_segments = []
                while trace_segments and phase.contains_commit(
                    trace_segments[0].commit
                ):
                    phase_trace_segments.append(trace_segments.pop(0))
                phase.trace_segments = phase_trace_segments

                phase_evaluations = []
                for candidate in phase.commits:
                    phase_evaluations.extend(
                        evaluations_by_commit.get(candidate.commit, [])
                    )
                phase.evaluations = phase_evaluations

                phases.append(phase)
                current_phase_commits = []

        # Handle remaining commits (no evaluations at the end)
        if current_phase_commits:
            is_initial = base_commit in [
                commit.commit for commit in current_phase_commits
            ]
            phases.append(
                OptimizationPhase(
                    is_initial=is_initial,
                    commits=current_phase_commits,
                    final_commit=current_phase_commits[-1],
                )
            )

        return phases

    def summary(self) -> dict[str, Any]:
        """Return a summary of the trace analysis payload."""
        return {
            "session_id": self.session_id,
            "model": self.config.get_model(),
            "base_commit": self.config.base_commit,
            "final_commit": self.config.final_commit,
            "num_phases": len(self.phases),
            "num_experiments": len(self.evaluations),
            "total_trace_items": len(self.agent_trace),
            "num_sub_agents": len(self.sub_agents_trace),
            "sub_agent_names": list(self.sub_agents_trace.keys()),
            "phase_summaries": [
                {
                    "phase": i + 1,
                    "commit": phase.final_commit.commit,
                    "num_experiments": len(phase.evaluations),
                    "num_trace_items": phase.num_trace_items,
                }
                for i, phase in enumerate(self.phases)
            ],
        }

    def get_trace_slice(self, phase_index: int | None = None) -> Trace:
        """Get the trace slice for a phase, or full trace if phase_index is None."""
        if phase_index is None:
            return self.agent_trace
        phase = self.get_phase(phase_index)
        if phase is None:
            return None
        return self.agent_trace[phase.earliest_span_idx : phase.latest_span_idx]

    def get_tool_calls(self, phase_index: int | None = None) -> list[ToolCall]:
        """Get tool calls from the agent trace (normalized format).

        Args:
            phase_index: Optional phase index to filter by (0-based)
        """
        trace = self.get_trace_slice(phase_index)
        return [tool_call for _, tool_call in TraceUtils.iter_tool_calls(trace)]

    def get_tool_calls_by_name(
        self, name_pattern: str, phase_index: int | None = None
    ) -> list[ToolCall]:
        """Get tool calls matching a name pattern (regex supported).

        Args:
            name_pattern: Regex pattern to match tool names
            phase_index: Optional phase index to filter by (0-based)
        """
        pattern = re.compile(name_pattern)
        tool_calls = self.get_tool_calls(phase_index)
        return [tc for tc in tool_calls if pattern.search(tc.name)]

    def get_sub_agent_traces(self) -> dict[str, SubAgentTrace]:
        """Get parsed sub-agent traces."""
        return self.sub_agents_trace

    def get_experiment_scores(self) -> list[dict[str, Any]]:
        """Get a summary of all experiment scores."""

        scores = []
        for evaluation in self.evaluations:
            evaluation_set = evaluation.request.evaluation_set
            scores.append(
                {
                    "experiment_id": evaluation.id,
                    "commit": evaluation.request.candidate.commit,
                    "dataset": evaluation_set.name,
                    "split": evaluation_set.partition,
                    "mean_score": evaluation.objective.value
                    if evaluation.objective is not None
                    else evaluation.report.metrics.get("score"),
                    "error_rate": evaluation.report.metrics.get("error_rate"),
                    "num_samples": len(evaluation.report.cases),
                }
            )
        return scores

    @property
    def experiments(self) -> list[EvaluationRecord]:
        """Deprecated trace-analysis alias for ``evaluations``."""
        return self.evaluations

    def get_phase(self, phase_index: int) -> OptimizationPhase | None:
        """Get a specific phase by index (0-based)."""
        if 0 <= phase_index < len(self.phases):
            return self.phases[phase_index]
        return None

    async def get_phase_info(
        self, phase_index: int, workspace: GitWorkspace | None = None
    ) -> dict[str, Any] | None:
        """Get comprehensive info about a phase.

        Args:
            phase_index: 0-based index of the phase
            workspace: Optional GitWorkspace to include diffs for each commit

        Returns:
            Dict with phase info, or None if index out of range
        """
        phase = self.get_phase(phase_index)
        if phase is None:
            return None

        info: dict[str, Any] = {
            "phase_index": phase_index,
            "final_commit": phase.final_commit.commit,
            "phase": phase.summary(),
            "trace_items": _dump_trace(self.get_trace_slice(phase_index)),
            "tool_calls": self.get_tool_calls(phase_index),
            "experiment_scores": phase.get_experiment_scores(),
        }

        if workspace:
            commit_diffs = []
            for candidate in phase.commits:
                diff = await get_commit_diff(workspace, candidate)
                commit_diffs.append({"commit": candidate.commit, "diff": diff})
            info["commit_diffs"] = commit_diffs

        return info

    def view_trace_item(self, index: int) -> Any | None:
        """View a specific trace item by index."""
        if 0 <= index < len(self.agent_trace):
            return self.agent_trace[index]
        return None
