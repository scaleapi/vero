"""Filesystem provisioning for the isolated harness user.

Single source of truth for the commands that hand the dropped-privilege harness
user access to its workspace — the eval launch (``backend.py``) and the isolation
integration test both use these so they can't drift. Kept stdlib-only and free of
``vero`` package imports so the test can load it standalone inside a minimal
container (see ``tests/test_v05_harbor_isolation_container.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath


def harness_grant_commands(
    user: str, *, chown_paths: Sequence[str], checkout_root: str
) -> list[list[str]]:
    """Commands to hand ``user`` its work dirs and make the checkout reachable.

    Chowns each work dir to the user, then grants traversal (``o+x``) on the
    checkout's parent: ``mktemp -d`` leaves that parent ``0700`` root, and the
    dropped user runs as its own uid+gid (so it is "other" for the root-owned
    parent) and otherwise can't traverse in to resolve the editable candidate
    package's absolute path — the import fails with "No module named <agent>"
    while harbor itself still loads from the user-owned cache. The parent holds
    only candidate code, so widening traversal exposes no trusted data.
    """
    owner = f"{user}:{user}"
    commands = [["chown", "-R", owner, str(path)] for path in chown_paths]
    checkout_parent = str(PurePosixPath(checkout_root).parent)
    commands.append(["chmod", "o+x", checkout_parent])
    return commands


def harness_reachability_probe(project_path: str) -> list[str]:
    """Command (run as the dropped user) that fails if the workspace is unreachable.

    Readability of ``project_path`` requires traversing every ancestor, so a
    non-zero exit means a provisioning gap left the workspace out of reach — caught
    at the launch site rather than as a cryptic "No module named <agent>" downstream.
    """
    return ["test", "-r", str(project_path)]
