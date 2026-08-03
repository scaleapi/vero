"""Read-only git access to a candidate repository.

Bare-repo reads only, via subprocess. No checkout ever happens: the analysis needs
trees and diffs, and materialising working copies for 100 cells would cost disk for
nothing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class CandidateRepo:
    def __init__(self, git_dir: Path) -> None:
        self.git_dir = Path(git_dir)

    def _run(self, *args: str) -> str:
        out = subprocess.run(
            ["git", "--git-dir", str(self.git_dir), *args],
            capture_output=True,
            text=True,
        )
        return out.stdout

    def log(self) -> list[tuple[str, str, str]]:
        """(sha, subject, body) oldest first, so index 0 is the seed.

        `--all` because a candidate chain can leave commits unreachable from any
        branch head once the optimizer reaches back past them.
        """
        raw = self._run("log", "--all", "--format=%H%x1f%s%x1f%b%x1e")
        rows = []
        for record in raw.split("\x1e"):
            record = record.strip("\n")
            if not record:
                continue
            sha, subject, body = (record.split("\x1f") + ["", "", ""])[:3]
            rows.append((sha, subject, body))
        rows.reverse()
        return rows

    def tree_sha(self, sha: str) -> str:
        return self._run("rev-parse", f"{sha}^{{tree}}").strip()

    def parent(self, sha: str) -> str | None:
        out = self._run("rev-parse", f"{sha}^").strip()
        return out or None

    def files(self, sha: str) -> list[str]:
        return [
            f
            for f in self._run("show", "--name-only", "--format=", sha).splitlines()
            if f.strip() and "__pycache__" not in f
        ]

    def show_file(self, sha: str, path: str) -> str:
        return self._run("show", f"{sha}:{path}")

    def diff(self, a: str, b: str, path: str | None = None, context: int = 0) -> str:
        args = ["diff", f"-U{context}", a, b, "--", path or ".", ":(exclude)*__pycache__*"]
        return self._run(*args)

    def numstat(self, a: str, b: str) -> dict[str, tuple[int, int]]:
        """path -> (added, removed). Binary files report as (0, 0)."""
        out: dict[str, tuple[int, int]] = {}
        raw = self._run("diff", "--numstat", a, b, "--", ".", ":(exclude)*__pycache__*")
        for line in raw.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            add, rem, path = parts
            out[path] = (
                int(add) if add.isdigit() else 0,
                int(rem) if rem.isdigit() else 0,
            )
        return out
