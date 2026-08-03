"""Facets an edit is labelled on, and the deterministic part of assigning them.

The vocabulary is derived from the symbol distribution the optimizers actually
touched across 100 runs, not chosen in advance: the same handful of targets dominate
in every benchmark, so the roots below are what the corpus contains rather than what
seemed plausible.

Three facets are model-assigned (`action`, `role` where a hint does not fire,
`provenance`); the rest are derived. Keeping the model's job to semantic judgement is
what makes the output checkable — a role hint that disagrees with a model label is a
bug you can find without reading anything.

Bump TAXONOMY_VERSION on any change here. It is part of the label cache key, so a
revision re-labels without re-extracting.
"""

from __future__ import annotations

import re
from enum import StrEnum

TAXONOMY_VERSION = "1"


class Role(StrEnum):
    """What part of the harness an edit targets. Closed and benchmark-agnostic."""

    PROMPT = "prompt"                       # system/instruction text
    CONTROL_LOOP = "control_loop"           # the agent's main turn loop
    TOOL_SURFACE = "tool_surface"           # tool declarations shown to the model
    TOOL_IMPL = "tool_impl"                 # a specific tool's implementation
    SUBMISSION = "submission"               # emitting/parsing the final answer
    MODEL_CLIENT = "model_client"           # completion params, retries, timeouts
    BUDGET_TURNS = "budget_turns"
    BUDGET_OUTPUT = "budget_output"         # tool-output truncation caps
    BUDGET_WALLCLOCK = "budget_wallclock"   # deadlines that stop the loop
    CONTEXT_MGMT = "context_mgmt"           # compaction, history pruning
    RETRIEVAL = "retrieval"                 # search/index parameters and strategy
    ENV_SETUP = "env_setup"                 # setup() work, package installation
    INITIALIZATION = "initialization"       # __init__, wiring
    TESTS = "tests"                         # the candidate's own test suite
    METADATA = "metadata"                   # version/name bookkeeping
    OTHER = "other"


class Action(StrEnum):
    """What was done to it."""

    FIX = "fix"                 # repairs a defect
    ADD = "add"                 # new capability or code path
    REMOVE = "remove"           # deletes a capability or code path
    TUNE = "tune"               # changes a value, no structural change
    RESTRUCTURE = "restructure" # same behaviour, different shape
    REWORD = "reword"           # instruction text changed, intent preserved
    REVERT = "revert"           # undoes the optimizer's own earlier edit
    COSMETIC = "cosmetic"       # cannot change behaviour


class Provenance(StrEnum):
    """For fixes: whose defect was it?

    Distinguishing these matters. An optimizer repairing the seed is doing the task;
    an optimizer repairing damage it caused two candidates ago is paying down its own
    debt, and several cells in this corpus spent most of their budget that way.
    """

    SEED = "seed"
    OWN = "own"
    UNKNOWN = "unknown"


class Direction(StrEnum):
    """For tunes. Derived from before/after values where both are numeric."""

    UP = "up"
    DOWN = "down"
    UNCHANGED = "unchanged"
    NA = "na"


# Symbol-name hints. These fire on the leaf symbol and are exact enough to skip the
# model entirely; anything unmatched goes to the labeller. Ordered, first match wins.
_ROLE_HINTS: list[tuple[re.Pattern[str], Role]] = [
    (re.compile(r"(PROMPT|INSTRUCTIONS?|GUIDANCE|PLAYBOOK)", re.I), Role.PROMPT),
    (re.compile(r"^(MAX_TURNS|MAX_STEPS|TURN_BUDGET|STEP_BUDGET|MAX_ITER\w*)$"), Role.BUDGET_TURNS),
    (re.compile(r"^(MAX_TOOL_OUTPUT\w*|MAX_OUTPUT\w*|MAX_CHARS|MAX_BYTES)$"), Role.BUDGET_OUTPUT),
    (re.compile(r"\w*(DEADLINE|TIME_BUDGET|WALL_CLOCK)\w*"), Role.BUDGET_WALLCLOCK),
    (re.compile(r"^(MAX_HISTORY\w*|MAX_CONTEXT\w*|.*COMPACT.*)$", re.I), Role.CONTEXT_MGMT),
    (re.compile(r"^TOOLS$"), Role.TOOL_SURFACE),
    (re.compile(r"\.run$"), Role.CONTROL_LOOP),
    (re.compile(r"\.(_submit|submit\w*|_answer_payload|_extract\w*answer\w*)$", re.I), Role.SUBMISSION),
    (re.compile(r"^(_answer_payload|_scale_variants|_format_number)$"), Role.SUBMISSION),
    (re.compile(r"\.(_completion_kwargs|_complete|_create|_chat)$"), Role.MODEL_CLIENT),
    (re.compile(r"\.(_run_shell|_exec|_run_python|_index_command|_open\w*|_search\w*)$"), Role.TOOL_IMPL),
    (re.compile(r"\.setup$"), Role.ENV_SETUP),
    (re.compile(r"\.__init__$"), Role.INITIALIZATION),
    (re.compile(r"\.(version|name)$"), Role.METADATA),
]


def role_hint(path: str, symbol: str, symbol_kind: str | None = None) -> Role | None:
    """Deterministic role where path, kind or name settles it, else None.

    Order matters. Path wins first: an edit inside the candidate's own test suite is
    a test edit whatever it touches. Kind wins next, because a long module-level
    string binding is a prompt regardless of what it is called — which is how
    `REVIEW_INSTRUCTIONS` and friends get caught without enumerating names.
    """
    if "test" in path.rsplit("/", 1)[-1]:
        return Role.TESTS
    if symbol_kind == "prompt_text":
        return Role.PROMPT
    for pattern, role in _ROLE_HINTS:
        if pattern.search(symbol):
            return role
    return None


_NUMERIC = re.compile(r"-?\d+(?:\.\d+)?")


def direction_of(before: str | None, after: str | None) -> Direction:
    """Numeric direction of a tune, where both sides parse."""
    if before is None or after is None:
        return Direction.NA
    if before == after:
        return Direction.UNCHANGED
    b, a = _NUMERIC.search(before), _NUMERIC.search(after)
    if not (b and a):
        return Direction.NA
    return Direction.UP if float(a.group()) > float(b.group()) else Direction.DOWN
