"""Admin bearer-token helpers for the Harbor evaluation sidecar."""

from __future__ import annotations

import os
import secrets
import tempfile
from pathlib import Path


def generate_admin_token() -> str:
    return secrets.token_urlsafe(32)


def write_admin_token(
    path: Path | str,
    token: str,
    *,
    mode: int = 0o600,
) -> Path:
    """Atomically write a restrictive token file for the trusted verifier."""
    if not token.strip() or "\x00" in token:
        raise ValueError("admin token must not be empty")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(token)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        destination.chmod(mode)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return destination


def read_admin_token(path: Path | str) -> str:
    token = Path(path).read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("admin token file is empty")
    return token


def check_admin_token(authorization: str | None, expected_token: str) -> bool:
    prefix = "Bearer "
    if authorization is None or not authorization.startswith(prefix):
        return False
    return secrets.compare_digest(authorization[len(prefix) :], expected_token)
