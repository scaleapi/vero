"""Trace analysis module for vero sessions."""

from vero.traces.analysis.analyzer import (
    DEFAULT_PHASE_ANALYSIS_PROMPT,
    ChangeTag,
    PhaseAnalysis,
    TraceAnalyzer,
    plot_session_scores,
    plot_session_scores_with_table,
)
from vero.traces.analysis.collator import (
    GitCommitHistory,
    OptimizationPhase,
    SessionConfig,
    SubAgentInfo,
    SubAgentTrace,
    ToolCall,
    ToolResult,
    Trace,
    TraceAnalysisPayload,
    TraceSegment,
    TraceUtils,
    get_commit_diff,
    get_commit_history,
    parse_trace,
)

__all__ = [
    # Data models (from collator)
    "GitCommitHistory",
    "Trace",
    "ToolCall",
    "ToolResult",
    "TraceSegment",
    "TraceUtils",
    "OptimizationPhase",
    "SubAgentInfo",
    "SubAgentTrace",
    "SessionConfig",
    "TraceAnalysisPayload",
    # Git utilities (from collator)
    "get_commit_history",
    "get_commit_diff",
    "parse_trace",
    # Analyzer
    "TraceAnalyzer",
    "PhaseAnalysis",
    "ChangeTag",
    "DEFAULT_PHASE_ANALYSIS_PROMPT",
    # Visualization
    "plot_session_scores",
    "plot_session_scores_with_table",
]
