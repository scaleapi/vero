"""Assign facets to edits: cached, resumable, and mostly not the model's job.

A deterministic hint overrides the model's role where one fires, but every edit is
sent to the model regardless, because `action` is never derivable from the artifact.
That makes the role comparison free and complete rather than sampled: agreement is
measured on 100% of hinted edits at no extra cost, and a hint that quietly disagrees
with every model reading is a bug in the hint.

The prompt shows the diff and withholds nothing except the commit message's
authority: subjects in this corpus routinely misdescribe their diffs — one reading
"Extend research and normalize wrapped answers" deletes an entire audit pass — so
the message is supplied as a claim to be checked, not as the answer.
"""

from __future__ import annotations

import asyncio
from typing import Iterable

from vero.interpret.cache import Cache, key_of
from vero.interpret.labeling.client import AsyncLLM, LLMError
from vero.interpret.labeling.taxonomy import (
    ACTION_RUBRIC,
    ROLE_RUBRIC,
    TAXONOMY_VERSION,
    Action,
    Direction,
    Provenance,
    Role,
    direction_of,
    role_hint,
)
from vero.interpret.models import Edit, EditLabel

PROMPT_VERSION = "2"

# Diffs were capped at 6000 characters, which truncated 12% of edits mid-hunk to save
# tokens that were never the constraint: the whole corpus costs a couple of dollars.
# These bounds exist only so one pathological edit cannot blow a context window.
MAX_DIFF_CHARS = 40_000
MAX_SOURCE_CHARS = 12_000

SYSTEM = """You classify individual edits made by an AI agent that was told to improve \
another agent's harness.

You are shown ONE edit: a diff restricted to a single symbol (a function, method, or \
module-level binding). Classify only that edit, not the whole commit.

The commit subject is provided for context but is frequently wrong: it may describe \
work that is not in this diff, omit changes that are, or claim a revert while leaving \
behaviour in place. Trust the diff. Where they disagree, say so in `mechanism`.

Judge only what the code does. Do not speculate about whether it improved the score."""

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "role", "provenance", "mechanism", "confidence"],
    "properties": {
        "action": {
            "type": "string",
            "enum": [a.value for a in Action],
            "description": "; ".join(f"{k}: {v}" for k, v in ACTION_RUBRIC.items()),
        },
        "role": {
            "type": "string",
            "enum": [r.value for r in Role],
            "description": "; ".join(f"{k}: {v}" for k, v in ROLE_RUBRIC.items()),
        },
        "provenance": {
            "type": "string",
            "enum": [p.value for p in Provenance],
            "description": "Ignored downstream -- provenance is derived from the seed "
                           "tree. Answer unknown unless the history line settles it.",
        },
        "mechanism": {
            "type": "string",
            "description": "One sentence: what this edit actually does.",
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


def _user_prompt(edit: Edit, subject: str, siblings: int = 0) -> str:
    value = ""
    if edit.before_value is not None or edit.after_value is not None:
        value = f"\nvalue: {edit.before_value} -> {edit.after_value}"

    # History. Without it the model cannot tell a first-time addition from a rewrite of
    # the optimizer's own work, and it guessed "own" on almost every fix when asked.
    if not edit.in_seed:
        history = "this symbol did not exist in the seed — the optimizer created it"
    elif edit.prior_touches:
        history = (
            f"this symbol came from the seed and {edit.prior_touches} earlier "
            f"candidate(s) in this run already modified it"
        )
    else:
        history = "this symbol is still as the seed wrote it; this is the first change to it"

    # The subject often describes OTHER edits in the same commit, so say how many there
    # are. Warning that it "may be inaccurate" was not enough on its own.
    sib = (
        f"\nnote: this commit touched {siblings} symbols in total, so the subject may "
        f"describe a different one"
        if siblings > 1 else ""
    )
    context = (
        f"\n\nthe symbol after the edit, for context:\n{edit.after_source[:MAX_SOURCE_CHARS]}"
        if edit.after_source else ""
    )
    return (
        f"file: {edit.path}\n"
        f"symbol: {edit.symbol}  ({edit.symbol_kind.value})\n"
        f"lines: +{edit.added} -{edit.removed}{value}\n"
        f"history: {history}\n"
        f"commit subject (may be inaccurate): {subject}{sib}\n\n"
        f"diff:\n{edit.diff[:MAX_DIFF_CHARS]}{context}"
    )


def cache_key(edit: Edit, model: str) -> str:
    return key_of(edit.id, model, PROMPT_VERSION, TAXONOMY_VERSION)


class Labeler:
    def __init__(self, llm: AsyncLLM, cache: Cache) -> None:
        self.llm = llm
        self.cache = cache
        self.hint_authoritative = 0
        self.disagreements = 0
        self.failed = 0

    async def label(self, edit: Edit, subject: str = "", siblings: int = 0) -> EditLabel | None:
        key = cache_key(edit, self.llm.settings.model)
        if (cached := self.cache.get_json(key)) is not None:
            return EditLabel.model_validate(cached)

        hint = role_hint(edit.path, edit.symbol, edit.symbol_kind.value)
        # Every edit goes to the model regardless of the hint, because `action` is never
        # derivable and always needs one. That makes the role comparison free rather
        # than sampled: the audit covers 100% of hinted edits at no extra cost. An
        # earlier version sampled it, which bought nothing and under-reported coverage.
        if hint is not None:
            self.hint_authoritative += 1

        try:
            raw = await self.llm.json_call(
                SYSTEM, _user_prompt(edit, subject, siblings), _SCHEMA
            )
        except LLMError:
            self.failed += 1
            return None

        role = hint.value if hint is not None else raw["role"]
        label = EditLabel(
            edit_id=edit.id,
            action=raw["action"],
            role=role,
            provenance=raw.get("provenance", Provenance.UNKNOWN.value),
            direction=direction_of(edit.before_value, edit.after_value).value,
            mechanism=raw.get("mechanism", "")[:300],
            confidence=float(raw.get("confidence", 0.0)),
            hinted=hint is not None,
            model=self.llm.settings.model,
            taxonomy_version=TAXONOMY_VERSION,
        )
        # Disagreement is recorded, not silently resolved: it is the signal that a
        # hint is wrong, and it is only visible if both readings are kept.
        if hint is not None and raw["role"] != hint.value:
            self.disagreements += 1
            label.mechanism = f"[hint={hint.value} model={raw['role']}] {label.mechanism}"
        self.cache.put_json(key, label.model_dump())
        return label

    async def label_all(
        self,
        edits: Iterable[tuple[Edit, str, int]],
        *,
        progress=None,
    ) -> list[EditLabel]:
        tasks = [asyncio.create_task(self.label(e, s, n)) for e, s, n in edits]
        out: list[EditLabel] = []
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            label = await task
            if label is not None:
                out.append(label)
            if progress and i % 50 == 0:
                progress(i, len(tasks))
        return out

    def stats(self) -> str:
        return (
            f"{self.cache.stats()}; calls={self.llm.calls} retries={self.llm.retries} "
            f"rule-decided={self.hint_authoritative} (all audited) "
            f"disagreements={self.disagreements} failed={self.failed}"
        )


def direction_only(edit: Edit) -> Direction:
    """Derived facet, exposed for callers that want it without labelling."""
    return direction_of(edit.before_value, edit.after_value)
