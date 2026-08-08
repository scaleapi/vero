"""Unpack the two things worth keeping out of a harbor `session.tar.gz`.

These archives run to hundreds of megabytes, almost all of it agent transcripts and
container logs. Only the candidate git repository and the sidecar evaluation records
are needed here, so members are filtered on the way out and the result is cached by
the archive's own digest — re-running the pipeline never re-extracts.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path

from vero.interpret.cache import Cache

_WANTED_DIR = "/candidates/repository.git/"
_WANTED_FILE = "evaluation.json"


def _contained(member: tarfile.TarInfo, dest: Path) -> bool:
    """Would extracting `member` stay inside `dest`?

    Name-matching alone is not containment: a member named
    `../../candidates/repository.git/config` matches the wanted prefix and still
    escapes. Links are dropped outright rather than resolved, which is what 3.12's
    `filter="data"` does and all these archives ever contain anyway.
    """
    if not (member.isfile() or member.isdir()):
        return False
    try:
        # An absolute member name makes Path() discard `dest` entirely, which
        # resolve() then exposes as an escape.
        return Path(dest, member.name).resolve().is_relative_to(dest.resolve())
    except (OSError, ValueError):
        return False


def digest(path: Path, *, chunk: int = 1 << 20) -> str:
    """Hash the archive itself, so the cache key is independent of where it sits."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def unpack(archive: Path, cache: Cache) -> Path | None:
    """Extract the wanted members, returning the session root."""
    key = digest(archive)
    if (hit := cache.get_dir(key)) is not None:
        return hit

    dest = cache.reserve_dir(key)
    try:
        with tarfile.open(archive) as tar:
            members = [
                m
                for m in tar.getmembers()
                if _WANTED_DIR in m.name or m.name.endswith(_WANTED_FILE)
            ]
            # Containment is checked here rather than left to `filter=`, which is
            # 3.12+; on 3.11 a member named `../…` matching the wanted prefix would
            # otherwise be written outside the cache.
            members = [m for m in members if _contained(m, dest)]
            if not members:
                return None
            if sys.version_info >= (3, 12):
                tar.extractall(dest, members=members, filter="data")
            else:
                tar.extractall(dest, members=members)
    except (tarfile.TarError, OSError):
        return None

    cache.commit_dir(key)
    return dest


def find_repo(session_root: Path) -> Path | None:
    hits = list(session_root.glob("**/candidates/repository.git"))
    return hits[0] if hits else None


def read_evaluations(session_root: Path) -> list[dict]:
    """Every sidecar evaluation record, unaggregated.

    Repeats of the same candidate on the same partition are kept separate: collapsing
    them into a per-partition map is exactly how a re-score silently overwrites the
    earlier one, which hides the corpus's only direct measurement of scoring noise.
    """
    out = []
    for path in session_root.glob("**/evaluations/*/evaluation.json"):
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return out
