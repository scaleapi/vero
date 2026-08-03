"""Display names for machine identifiers.

The taxonomy uses snake_case values because they are dictionary keys, cache-key
components and JSON fields; renaming them would invalidate every cached label and
break the data contract in the figure specs. Presentation is a separate concern, so
the mapping lives here and both the print figures and the HTML page read from it.

Benchmark names follow the published spelling rather than a mechanical
transformation: these appear in a paper, where `Terminal-Bench` and `GAIA` are the
names readers will look up.

Code symbols are deliberately absent. `MAX_TURNS` is a real identifier in the
optimized source, and prettifying it would misrepresent what the optimizer edited.
"""

from __future__ import annotations

ROLE: dict[str, str] = {
    "prompt": "Prompt",
    "control_loop": "Control Loop",
    "tool_surface": "Tool Surface",
    "tool_impl": "Tool Implementation",
    "submission": "Submission",
    "model_client": "Model Client",
    "budget_turns": "Turn Budget",
    "budget_output": "Output Cap",
    "budget_wallclock": "Wall-Clock Budget",
    "context_mgmt": "Context Management",
    "retrieval": "Retrieval",
    "env_setup": "Environment Setup",
    "initialization": "Initialization",
    "tests": "Tests",
    "metadata": "Metadata",
    "other": "Other",
}

ACTION: dict[str, str] = {
    "fix": "Fix",
    "add": "Add",
    "remove": "Remove",
    "tune": "Tune",
    "restructure": "Restructure",
    "reword": "Reword",
    "revert": "Revert",
    "cosmetic": "Cosmetic",
}

PROVENANCE: dict[str, str] = {
    "seed": "Seed defect",
    "own": "Own defect",
    "unknown": "Unknown",
}

BENCHMARK: dict[str, str] = {
    "browsecomp-plus": "BrowseComp-Plus",
    "officeqa": "OfficeQA",
    "swe-atlas-qna": "SWE-Atlas-QnA",
    "terminal-bench": "Terminal-Bench",
    "gaia-shell": "GAIA-Shell",
}

# Shorter forms for axis ticks where the full name will not fit five across a
# text-width figure. Same names, dropped qualifiers -- never a different name.
BENCHMARK_SHORT: dict[str, str] = {
    "browsecomp-plus": "BrowseComp+",
    "officeqa": "OfficeQA",
    "swe-atlas-qna": "SWE-Atlas",
    "terminal-bench": "Terminal-Bench",
    "gaia-shell": "GAIA-Shell",
}


def role(key: str) -> str:
    """Display name, falling back to a readable form for anything unmapped."""
    return ROLE.get(key, key.replace("_", " ").title())


def action(key: str) -> str:
    return ACTION.get(key, key.replace("_", " ").title())


def provenance(key: str) -> str:
    return PROVENANCE.get(key, key.replace("_", " ").title())


def benchmark(key: str, *, short: bool = False) -> str:
    table = BENCHMARK_SHORT if short else BENCHMARK
    return table.get(key, key)
