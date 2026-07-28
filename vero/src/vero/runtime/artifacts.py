"""Safe storage for session-level runtime artifacts."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from vero.evaluation.store.persistence import _atomic_write_json


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root

    def path(self, relative_path: str) -> Path:
        value = PurePosixPath(relative_path)
        if (
            not relative_path
            or "\\" in relative_path
            or value.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
        ):
            raise ValueError("artifact path must be a safe relative POSIX path")
        resolved = (self.root / Path(*value.parts)).resolve()
        if not resolved.is_relative_to(self.root.resolve()):
            raise ValueError("artifact path escapes the session artifact directory")
        return resolved

    def write_text(self, relative_path: str, value: str) -> Path:
        path = self.path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return path

    def write_json(self, relative_path: str, value: Any) -> Path:
        path = self.path(relative_path)
        _atomic_write_json(path, value)
        return path

    def read_json(self, relative_path: str) -> Any:
        return json.loads(self.path(relative_path).read_text(encoding="utf-8"))
