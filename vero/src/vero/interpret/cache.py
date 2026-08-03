"""Content-addressed file cache, shared by artifact extraction and labelling.

Two callers with very different economics use this: unpacking a session archive costs
minutes and hundreds of megabytes, while an LLM label costs a fraction of a cent. They
get separate namespaces so revising a taxonomy re-labels everything without
re-extracting anything — the failure mode that actually wastes an afternoon.

Writes go to a temp file and are renamed into place, so an interrupted run leaves
either a complete entry or none, never a truncated one that poisons the next attempt.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def key_of(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8", "replace")).hexdigest()


class Cache:
    """A namespaced content-addressed store under one root."""

    def __init__(self, root: Path, namespace: str, *, refresh: bool = False) -> None:
        self.root = Path(root) / namespace
        self.refresh = refresh
        self.hits = 0
        self.misses = 0

    def _path(self, key: str, suffix: str) -> Path:
        return self.root / key[:2] / f"{key}{suffix}"

    # -- JSON entries (labels, parsed records) --------------------------------

    def get_json(self, key: str) -> Any | None:
        if self.refresh:
            return None
        path = self._path(key, ".json")
        if not path.is_file():
            self.misses += 1
            return None
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put_json(self, key: str, value: Any) -> None:
        path = self._path(key, ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(value, indent=None, default=str))
        os.replace(tmp, path)

    # -- Directory entries (unpacked archives) --------------------------------

    def get_dir(self, key: str) -> Path | None:
        """A completed directory entry, or None.

        Completion is marked by a sentinel written after the unpack finishes, so a
        directory left behind by a crash is treated as absent and redone rather than
        silently used half-populated.
        """
        path = self._path(key, ".d")
        if self.refresh or not (path / ".complete").is_file():
            self.misses += 1
            return None
        self.hits += 1
        return path

    def reserve_dir(self, key: str) -> Path:
        """Empty directory to unpack into. Call `commit_dir` when finished."""
        path = self._path(key, ".d")
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def commit_dir(self, key: str) -> Path:
        path = self._path(key, ".d")
        (path / ".complete").write_text("")
        return path

    def stats(self) -> str:
        total = self.hits + self.misses
        pct = (100.0 * self.hits / total) if total else 0.0
        return f"{self.root.name}: {self.hits} hit / {self.misses} miss ({pct:.0f}%)"
