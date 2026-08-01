"""Build and serve Harbor sidecar components from a trusted factory."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from vero.sidecar.auth import (
    generate_admin_token,
    read_admin_token,
    write_admin_token,
)
from vero.sidecar.sidecar import EvaluationSidecar
from vero.sidecar.verifier import CanonicalVerifier

if TYPE_CHECKING:
    from vero.runtime.wandb import InferenceTelemetryPoller


@dataclass(frozen=True)
class SidecarComponents:
    sidecar: EvaluationSidecar
    verifier: CanonicalVerifier
    telemetry: "InferenceTelemetryPoller | None" = None


SidecarFactory = Callable[
    [dict[str, Any]],
    SidecarComponents | Awaitable[SidecarComponents],
]


def load_factory(import_path: str) -> SidecarFactory:
    module_name, separator, attribute_name = import_path.partition(":")
    if not separator or not module_name.strip() or not attribute_name.strip():
        raise ValueError("sidecar factory must use module:attribute syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError("sidecar factory is not callable")
    return factory


async def build_components(
    *,
    factory_path: str,
    config_path: Path | str,
) -> SidecarComponents:
    path = Path(config_path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"invalid sidecar config {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("sidecar config must be a JSON object")
    built = load_factory(factory_path)(config)
    if inspect.isawaitable(built):
        built = await built
    if not isinstance(built, SidecarComponents):
        raise TypeError("sidecar factory must return SidecarComponents")
    return built


async def build_app(
    *,
    factory_path: str,
    config_path: Path | str,
    admin_token_path: Path | str,
):
    from vero.sidecar.app import create_app

    components = await build_components(
        factory_path=factory_path,
        config_path=config_path,
    )
    # Reuse the admin token already on the volume instead of minting a fresh one
    # on every start. Minting unconditionally meant that a sidecar restart inside
    # a run silently invalidated the token the outer agent was already holding:
    # its next admin call 401'd and the run was dead, even though the session
    # directory, the evaluation database and the budget ledger had all survived
    # the restart intact. Reuse does not widen exposure, the token file is 0400
    # inside a 0700 directory (see write_admin_token) and the volume is per-run,
    # so the only readers are the ones that could already read it before.
    token_path = Path(admin_token_path)
    try:
        token = read_admin_token(token_path)
    except (OSError, ValueError):
        # No token yet (first start), or one that cannot be read back as a token:
        # either way the holder has nothing usable, so mint and persist one.
        token = generate_admin_token()
        write_admin_token(token_path, token)
    return create_app(
        sidecar=components.sidecar,
        verifier=components.verifier,
        admin_token=token,
        telemetry=components.telemetry,
    )


def serve(
    *,
    factory_path: str,
    config_path: Path | str,
    admin_token_path: Path | str,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    import uvicorn

    app = asyncio.run(
        build_app(
            factory_path=factory_path,
            config_path=config_path,
            admin_token_path=admin_token_path,
        )
    )
    uvicorn.run(app, host=host, port=port)
