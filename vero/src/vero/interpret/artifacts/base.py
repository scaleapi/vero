"""The adapter boundary.

An adapter takes raw run artifacts from one producer and canonicalises them into
`models`. Everything downstream — edit decomposition, labelling, analysis — consumes
only the canonical types, so supporting a second producer of optimization runs is a
new adapter and nothing else.

Adapters do two things and no more: find runs under the roots they are given, and
load one run. They must not interpret, classify, or score anything; that judgement
belongs in `edits/` (deterministic) and `labeling/` (model-assisted), which is what
keeps the boundary useful.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

from vero.interpret.models import CellRef, Trajectory


@runtime_checkable
class SourceAdapter(Protocol):
    """Canonicalises one producer's run artifacts."""

    name: str

    def discover(self, roots: Iterable[Path]) -> list[CellRef]:
        """Find every run under these roots. Cheap: no archives are opened."""
        ...

    def load(self, ref: CellRef) -> Trajectory:
        """Materialise one run. May be expensive; results are cached by the caller."""
        ...


_REGISTRY: dict[str, SourceAdapter] = {}


def register(adapter: SourceAdapter) -> SourceAdapter:
    _REGISTRY[adapter.name] = adapter
    return adapter


def get(name: str) -> SourceAdapter:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "none registered"
        raise KeyError(f"unknown source adapter {name!r} (known: {known})")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
