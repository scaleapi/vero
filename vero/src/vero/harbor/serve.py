"""Build and serve Harbor sidecar components from a trusted factory."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from vero.harbor.auth import generate_admin_token, write_admin_token
from vero.harbor.sidecar import EvaluationSidecar
from vero.harbor.verifier import CanonicalVerifier

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
    from vero.harbor.app import create_app

    components = await build_components(
        factory_path=factory_path,
        config_path=config_path,
    )
    token = generate_admin_token()
    write_admin_token(admin_token_path, token)
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
