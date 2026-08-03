"""Assign facets to edits: cached, resumable, and mostly not the model's job.

Roles that a deterministic hint settles are not sent to the model at all, but the
model is still asked for a role on a sample of hinted edits so the two can be
compared. Agreement measured beats agreement assumed, and a hint that quietly
disagrees with every model reading is a bug in the hint.

The prompt shows the diff and withholds nothing except the commit message's
authority: subjects in this corpus routinely misdescribe their diffs — one reading
"Extend research and normalize wrapped answers" deletes an entire audit pass — so
the message is supplied as a claim to be checked, not as the answer.
"""

from __future__ import annotations

import asyncio
import random
from typing import Iterable

from vero.interpret.cache import Cache, key_of
from vero.interpret.labeling.client import AsyncLLM, LLMError
from vero.interpret.labeling.taxonomy import (
    TAXONOMY_VERSION,
    Action,
    Direction,
    Provenance,
    Role,
    direction_of,
    role_hint,
)
from vero.interpret.models import Edit, EditLabel

PROMPT_VERSION = "1"

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
        "action": {"type": "string", "enum": [a.value for a in Action]},
        "role": {"type": "string", "enum": [r.value for r in Role]},
        "provenance": {"type": "string", "enum": [p.value for p in Provenance]},
        "mechanism": {
            "type": "string",
            "description": "One sentence: what this edit actually does.",
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


def _user_prompt(edit: Edit, subject: str) -> str:
    value = ""
    if edit.before_value is not None or edit.after_value is not None:
        value = f"\nvalue: {edit.before_value} -> {edit.after_value}"
    return (
        f"file: {edit.path}\n"
        f"symbol: {edit.symbol}  ({edit.symbol_kind.value})\n"
        f"lines: +{edit.added} -{edit.removed}{value}\n"
        f"commit subject (may be inaccurate): {subject}\n\n"
        f"diff:\n{edit.diff[:6000]}"
    )


def cache_key(edit: Edit, model: str) -> str:
    return key_of(edit.id, model, PROMPT_VERSION, TAXONOMY_VERSION)


class Labeler:
    def __init__(
        self,
        llm: AsyncLLM,
        cache: Cache,
        *,
        audit_rate: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.audit_rate = audit_rate
        self._rng = random.Random(seed)
        self.skipped_by_hint = 0
        self.audited = 0
        self.failed = 0

    async def label(self, edit: Edit, subject: str = "") -> EditLabel | None:
        key = cache_key(edit, self.llm.settings.model)
        if (cached := self.cache.get_json(key)) is not None:
            return EditLabel.model_validate(cached)

        hint = role_hint(edit.path, edit.symbol, edit.symbol_kind.value)
        # A hinted role still needs an action, so the call happens either way; the
        # hint decides whether the model's role is authoritative or merely audited.
        audit = hint is not None and self._rng.random() < self.audit_rate
        if hint is not None and not audit:
            self.skipped_by_hint += 1
        if audit:
            self.audited += 1

        try:
            raw = await self.llm.json_call(SYSTEM, _user_prompt(edit, subject), _SCHEMA)
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
            label.mechanism = f"[hint={hint.value} model={raw['role']}] {label.mechanism}"
        self.cache.put_json(key, label.model_dump())
        return label

    async def label_all(
        self,
        edits: Iterable[tuple[Edit, str]],
        *,
        progress=None,
    ) -> list[EditLabel]:
        tasks = [asyncio.create_task(self.label(e, s)) for e, s in edits]
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
            f"hint-authoritative={self.skipped_by_hint} audited={self.audited} "
            f"failed={self.failed}"
        )


def direction_only(edit: Edit) -> Direction:
    """Derived facet, exposed for callers that want it without labelling."""
    return direction_of(edit.before_value, edit.after_value)
