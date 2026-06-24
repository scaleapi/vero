"""Admin-token auth for the eval sidecar.

The token gates the admin `finalize` endpoint. It is generated per trial by the
sidecar and written `root:600` on a volume mounted into `main`, so the verifier
(root, shared mode) can read it but the optimizer (`agent.user`) cannot. The
optimizer therefore can only reach the agent endpoints, never `finalize`.
"""

from __future__ import annotations

import secrets
from pathlib import Path

_BEARER = "Bearer "


def generate_token() -> str:
    return secrets.token_urlsafe(32)


def write_admin_token(path: Path | str, token: str, *, mode: int = 0o600) -> Path:
    """Write the token to ``path`` with restrictive perms (caller runs as root so the
    file is root-owned and unreadable by ``agent.user``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(token)
    p.chmod(mode)
    return p


def read_admin_token(path: Path | str) -> str:
    return Path(path).read_text().strip()


def check_admin(authorization: str | None, expected_token: str) -> bool:
    """Constant-time check of an ``Authorization: Bearer <token>`` header."""
    if not authorization or not authorization.startswith(_BEARER):
        return False
    return secrets.compare_digest(authorization[len(_BEARER):], expected_token)
