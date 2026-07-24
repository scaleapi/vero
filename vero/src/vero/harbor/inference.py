"""Credential-isolating, budgeted inference gateway for Harbor tasks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import Field, field_validator, model_validator

from vero.evaluation import EvaluationModel
from vero.evaluation.persistence import _atomic_write_json

logger = logging.getLogger(__name__)


def token_digest(token: str) -> str:
    """Return the stable digest stored in trusted gateway configuration."""
    if not token.strip():
        raise ValueError("inference token must not be empty")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_inference_token() -> str:
    return secrets.token_urlsafe(32)


class InferenceScopeConfig(EvaluationModel):
    """One independently authenticated and metered inference consumer."""

    token_sha256: str
    allowed_models: list[str]
    max_requests: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_concurrency: int = Field(default=8, ge=1)

    @field_validator("token_sha256")
    @classmethod
    def validate_token_digest(cls, value: str) -> str:
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("token_sha256 must be a lowercase SHA-256 digest")
        return value

    @field_validator("allowed_models")
    @classmethod
    def validate_models(cls, value: list[str]) -> list[str]:
        if not value or any(not model.strip() for model in value):
            raise ValueError("allowed_models must contain non-empty model names")
        if len(value) != len(set(value)):
            raise ValueError("allowed_models must be unique")
        return value


class InferenceRequestLogConfig(EvaluationModel):
    """Durable JSONL capture of every request the gateway proxies or denies.

    Operator telemetry: the log lives on the gateway state volume, which is
    never agent-visible. Bodies are stored head+tail truncated so the terminal
    usage frame of a stream survives; ``body_bytes: 0`` keeps metadata only.
    """

    directory: str = "/state/inference/requests"
    body_bytes: int = Field(default=16384, ge=0)
    rotate_bytes: int = Field(default=64 * 1024 * 1024, ge=1_048_576)

    @field_validator("directory")
    @classmethod
    def validate_directory(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("request log directory must be absolute")
        return value


class InferenceGatewayConfig(EvaluationModel):
    """Trusted configuration for an OpenAI-compatible inference gateway."""

    upstream_api_key_env: str = "OPENAI_API_KEY"
    upstream_base_url_env: str | None = "OPENAI_BASE_URL"
    default_upstream_base_url: str = "https://api.openai.com/v1"
    state_path: str = "/state/inference/usage.json"
    request_log: InferenceRequestLogConfig | None = None
    scopes: dict[str, InferenceScopeConfig]

    @field_validator("upstream_api_key_env", "upstream_base_url_env")
    @classmethod
    def validate_environment_name(cls, value: str | None) -> str | None:
        if value is not None and (
            not value
            or not (value[0].isalpha() or value[0] == "_")
            or any(not (character.isalnum() or character == "_") for character in value)
        ):
            raise ValueError("gateway environment names must be valid identifiers")
        return value

    @field_validator("default_upstream_base_url")
    @classmethod
    def validate_upstream_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("default_upstream_base_url must be HTTP(S)")
        return value.rstrip("/")

    @field_validator("state_path")
    @classmethod
    def validate_state_path(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("gateway state_path must be absolute")
        return value

    @model_validator(mode="after")
    def validate_scopes(self) -> InferenceGatewayConfig:
        if not self.scopes:
            raise ValueError("gateway requires at least one scope")
        for name in self.scopes:
            if not name or any(
                not (character.isalnum() or character in "_-") for character in name
            ):
                raise ValueError(f"invalid inference scope {name!r}")
        return self


class InferenceAttributionUsage(EvaluationModel):
    requests: int = Field(default=0, ge=0)
    upstream_errors: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    # Provider-reported cache reads; a subset of input_tokens, not additive.
    cached_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class InferenceScopeUsage(InferenceAttributionUsage):
    active_requests: int = Field(default=0, ge=0)
    attributions: dict[str, InferenceAttributionUsage] = Field(default_factory=dict)


class InferenceUsageLedger(EvaluationModel):
    schema_version: Literal[1] = 1
    scopes: dict[str, InferenceScopeUsage]


class InferenceBudgetExceeded(RuntimeError):
    pass


class InferenceUsageStore:
    """Cancellation-safe request reservations and durable provider usage."""

    def __init__(self, config: InferenceGatewayConfig):
        self.config = config
        self.path = Path(config.state_path)
        self._lock = asyncio.Lock()
        self._limits = {
            name: asyncio.Semaphore(scope.max_concurrency)
            for name, scope in config.scopes.items()
        }
        self.ledger = self._load()

    def _load(self) -> InferenceUsageLedger:
        if self.path.exists():
            loaded = InferenceUsageLedger.model_validate_json(
                self.path.read_text(encoding="utf-8")
            )
            unknown = set(loaded.scopes) - set(self.config.scopes)
            if unknown:
                raise ValueError(
                    "inference usage contains unknown scopes: "
                    + ", ".join(sorted(unknown))
                )
            scopes = dict(loaded.scopes)
            for name in self.config.scopes:
                scopes.setdefault(name, InferenceScopeUsage())
            # Active requests cannot survive a gateway restart.
            scopes = {
                name: value.model_copy(update={"active_requests": 0})
                for name, value in scopes.items()
            }
            return loaded.model_copy(update={"scopes": scopes})
        return InferenceUsageLedger(
            scopes={name: InferenceScopeUsage() for name in self.config.scopes}
        )

    def _save(self) -> None:
        _atomic_write_json(self.path, self.ledger.model_dump(mode="json"))

    async def reserve(self, scope_name: str, attribution: str) -> None:
        limiter = self._limits[scope_name]
        await limiter.acquire()
        try:
            async with self._lock:
                limits = self.config.scopes[scope_name]
                usage = self.ledger.scopes[scope_name]
                if (
                    limits.max_requests is not None
                    and usage.requests >= limits.max_requests
                ):
                    raise InferenceBudgetExceeded("inference request budget exhausted")
                if (
                    limits.max_tokens is not None
                    and usage.total_tokens >= limits.max_tokens
                ):
                    raise InferenceBudgetExceeded("inference token budget exhausted")
                attribution_usage = usage.attributions.get(
                    attribution, InferenceAttributionUsage()
                )
                usage = usage.model_copy(
                    update={
                        "requests": usage.requests + 1,
                        "active_requests": usage.active_requests + 1,
                        "attributions": {
                            **usage.attributions,
                            attribution: attribution_usage.model_copy(
                                update={"requests": attribution_usage.requests + 1}
                            ),
                        },
                    }
                )
                self.ledger = self.ledger.model_copy(
                    update={"scopes": {**self.ledger.scopes, scope_name: usage}}
                )
                self._save()
        except BaseException:
            limiter.release()
            raise

    async def complete(
        self,
        scope_name: str,
        attribution: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cached_input_tokens: int = 0,
        upstream_error: bool = False,
    ) -> None:
        try:
            async with self._lock:
                usage = self.ledger.scopes[scope_name]
                attribution_usage = usage.attributions[attribution]
                update = {
                    "upstream_errors": usage.upstream_errors + int(upstream_error),
                    "input_tokens": usage.input_tokens + input_tokens,
                    "cached_input_tokens": usage.cached_input_tokens
                    + cached_input_tokens,
                    "output_tokens": usage.output_tokens + output_tokens,
                    "total_tokens": usage.total_tokens + total_tokens,
                    "active_requests": max(0, usage.active_requests - 1),
                    "attributions": {
                        **usage.attributions,
                        attribution: attribution_usage.model_copy(
                            update={
                                "upstream_errors": attribution_usage.upstream_errors
                                + int(upstream_error),
                                "input_tokens": attribution_usage.input_tokens
                                + input_tokens,
                                "cached_input_tokens": (
                                    attribution_usage.cached_input_tokens
                                    + cached_input_tokens
                                ),
                                "output_tokens": attribution_usage.output_tokens
                                + output_tokens,
                                "total_tokens": attribution_usage.total_tokens
                                + total_tokens,
                            }
                        ),
                    },
                }
                self.ledger = self.ledger.model_copy(
                    update={
                        "scopes": {
                            **self.ledger.scopes,
                            scope_name: usage.model_copy(update=update),
                        }
                    }
                )
                self._save()
        finally:
            self._limits[scope_name].release()

    def public_status(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, scope in self.config.scopes.items():
            usage = self.ledger.scopes[name]
            result[name] = {
                **usage.model_dump(mode="json"),
                "max_requests": scope.max_requests,
                "max_tokens": scope.max_tokens,
                "max_concurrency": scope.max_concurrency,
                "allowed_models": list(scope.allowed_models),
                "remaining_requests": (
                    None
                    if scope.max_requests is None
                    else max(0, scope.max_requests - usage.requests)
                ),
                "remaining_tokens": (
                    None
                    if scope.max_tokens is None
                    else max(0, scope.max_tokens - usage.total_tokens)
                ),
            }
        return result


def _bearer_token(authorization: str | None) -> str | None:
    prefix = "Bearer "
    if authorization is None or not authorization.startswith(prefix):
        return None
    return authorization[len(prefix) :]


def _usage(value: Any) -> tuple[int, int, int, int]:
    if not isinstance(value, dict):
        return (0, 0, 0, 0)
    usage = value.get("usage")
    if not isinstance(usage, dict):
        response = value.get("response")
        usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return (0, 0, 0, 0)
    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(
        usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    )
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    # Responses API nests cache reads under input_tokens_details; chat
    # completions under prompt_tokens_details.
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details")
    cached_tokens = (
        int(details.get("cached_tokens", 0) or 0) if isinstance(details, dict) else 0
    )
    return (input_tokens, output_tokens, total_tokens, cached_tokens)


class _StreamingUsage:
    def __init__(self):
        self.buffer = b""
        self.tokens = (0, 0, 0, 0)

    def feed(self, chunk: bytes) -> None:
        self.buffer += chunk
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            if not line.startswith(b"data:"):
                continue
            payload = line.removeprefix(b"data:").strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                value = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            tokens = _usage(value)
            if tokens != (0, 0, 0, 0):
                self.tokens = tokens


class _BoundedCapture:
    """Keep the head and tail of a byte stream within a fixed byte budget."""

    def __init__(self, cap: int):
        self.cap = cap
        self.head_limit = cap // 2
        self.tail_limit = cap - self.head_limit
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        if self.cap <= 0:
            return
        if len(self.head) < self.head_limit:
            take = min(self.head_limit - len(self.head), len(chunk))
            self.head += chunk[:take]
            chunk = chunk[take:]
        if chunk:
            self.tail += chunk
            if len(self.tail) > self.tail_limit:
                del self.tail[: len(self.tail) - self.tail_limit]

    def snapshot(self) -> dict[str, Any]:
        if self.cap <= 0:
            return {"bytes": self.total, "truncated": self.total > 0}
        kept = len(self.head) + len(self.tail)
        result: dict[str, Any] = {"bytes": self.total, "truncated": self.total > kept}
        text = bytes(self.head).decode("utf-8", errors="replace")
        if self.tail and not result["truncated"]:
            text += bytes(self.tail).decode("utf-8", errors="replace")
        elif self.tail:
            result["tail"] = bytes(self.tail).decode("utf-8", errors="replace")
        result["text"] = text
        return result


class InferenceRequestLog:
    """Size-rotated JSONL log of every request the gateway proxies or denies.

    Best-effort by construction: a logging failure must never affect the
    proxied request.
    """

    def __init__(self, config: InferenceRequestLogConfig):
        self.config = config
        self.directory = Path(config.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        existing = sorted(self.directory.glob("requests-*.jsonl"))
        if existing:
            self._current = existing[-1]
            self._index = len(existing)
            self._size = self._current.stat().st_size
        else:
            self._index = 1
            self._current = self.directory / "requests-00001.jsonl"
            self._size = 0

    def body(self, data: bytes) -> dict[str, Any]:
        capture = self.capture()
        capture.feed(data)
        return capture.snapshot()

    def capture(self) -> _BoundedCapture:
        return _BoundedCapture(self.config.body_bytes)

    async def record(self, **fields: Any) -> None:
        try:
            line = json.dumps(
                {
                    "schema_version": 1,
                    "ts": datetime.now(UTC).isoformat(),
                    **fields,
                },
                ensure_ascii=False,
                default=str,
            )
            encoded = (line + "\n").encode("utf-8")
            async with self._lock:
                if self._size and self._size + len(encoded) > self.config.rotate_bytes:
                    self._index += 1
                    self._current = self.directory / f"requests-{self._index:05d}.jsonl"
                    self._size = 0
                await asyncio.to_thread(self._append, self._current, encoded)
                self._size += len(encoded)
        except Exception:
            logger.warning("inference request log write failed", exc_info=True)

    @staticmethod
    def _append(path: Path, data: bytes) -> None:
        with open(path, "ab") as handle:
            handle.write(data)


_REQUEST_HEADERS = {
    "accept",
    "content-type",
    "openai-beta",
    "user-agent",
    "x-stainless-arch",
    "x-stainless-async",
    "x-stainless-lang",
    "x-stainless-os",
    "x-stainless-package-version",
    "x-stainless-retry-count",
    "x-stainless-runtime",
    "x-stainless-runtime-version",
    "x-stainless-timeout",
}
_RESPONSE_HEADERS = {
    "content-type",
    "openai-processing-ms",
    "request-id",
    "x-request-id",
}
def _provider_error(status_code: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "vero_inference_gateway_error",
                "code": code,
            }
        },
    )


def create_inference_gateway_app(
    *,
    config: InferenceGatewayConfig,
    upstream_api_key: str,
    upstream_base_url: str | None = None,
    transport: Any = None,
) -> FastAPI:
    """Create a scoped reverse proxy without exposing the upstream credential."""
    if not upstream_api_key.strip():
        raise ValueError("upstream API key must not be empty")
    try:
        import httpx
    except ImportError as error:
        raise RuntimeError("install scale-vero[harbor] to serve inference") from error

    base_url = (upstream_base_url or config.default_upstream_base_url).rstrip("/")
    store = InferenceUsageStore(config)
    request_log = (
        InferenceRequestLog(config.request_log)
        if config.request_log is not None
        else None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(
            timeout=None,
            transport=transport,
            follow_redirects=False,
        ) as client:
            app.state.upstream = client
            yield

    app = FastAPI(title="VeRO inference gateway", version="1", lifespan=lifespan)
    app.state.usage_store = store
    app.state.request_log = request_log

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/usage/{scope_name}")
    async def usage_status(
        scope_name: str,
        authorization: Annotated[str | None, Header()] = None,
    ):
        scope = config.scopes.get(scope_name)
        token = _bearer_token(authorization)
        if (
            scope is None
            or token is None
            or not secrets.compare_digest(token_digest(token), scope.token_sha256)
        ):
            raise HTTPException(status_code=403, detail="invalid inference scope token")
        return store.public_status()[scope_name]

    @app.api_route(
        "/scopes/{scope_name}/{attribution}/v1/{endpoint:path}",
        methods=["POST"],
    )
    async def proxy(
        scope_name: str,
        attribution: str,
        endpoint: str,
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
        x_api_key: Annotated[str | None, Header()] = None,
    ):
        scope = config.scopes.get(scope_name)
        # Provider-agnostic auth: OpenAI/codex clients send the scope token as
        # `Authorization: Bearer`, Anthropic/Claude clients send it as `x-api-key`.
        token = _bearer_token(authorization) or (
            x_api_key.strip() if x_api_key and x_api_key.strip() else None
        )
        if (
            scope is None
            or token is None
            or not secrets.compare_digest(token_digest(token), scope.token_sha256)
        ):
            return _provider_error(
                403, "invalid inference scope token", "invalid_token"
            )
        # Provider-agnostic passthrough: forward any inference endpoint
        # (responses, chat/completions, messages, embeddings, ...) to the upstream
        # litellm proxy, which decides what it supports. Guard only against path
        # traversal escaping the scoped upstream base — not a per-provider list.
        endpoint = endpoint.strip("/")
        if not endpoint or ".." in endpoint.split("/"):
            return _provider_error(
                400, "invalid inference endpoint", "invalid_endpoint"
            )
        try:
            body = await request.body()
            value = json.loads(body) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _provider_error(400, "request body must be JSON", "invalid_request")

        started = time.monotonic()

        async def log_request(
            *,
            status: int,
            error: str | None = None,
            stream: bool = False,
            tokens: tuple[int, int, int, int] = (0, 0, 0, 0),
            response: dict[str, Any] | None = None,
            upstream_error: bool = False,
        ) -> None:
            if request_log is None:
                return
            await request_log.record(
                scope=scope_name,
                attribution=attribution,
                endpoint=endpoint,
                model=value.get("model") if isinstance(value, dict) else None,
                stream=stream,
                status=status,
                error=error,
                latency_ms=round((time.monotonic() - started) * 1000, 1),
                input_tokens=tokens[0],
                output_tokens=tokens[1],
                total_tokens=tokens[2],
                cached_input_tokens=tokens[3],
                upstream_error=upstream_error,
                request=request_log.body(body),
                response=response,
            )
        # Per-scope allow-list: producer (optimizer) and evaluation (target) have
        # separate tokens + allowed_models, so the target is normally confined to
        # its fixed eval model. NOTE (known, deferred): this is not a *hard*
        # fixed-target guarantee. Both scopes share one gateway host, split only by
        # URL path, and the optimizer holds the producer token and authors the
        # candidate. An adversarial optimizer can smuggle its producer token into
        # the candidate and, from the eval sandbox (which already reaches this host
        # for its legit calls), hit /scopes/producer to run the optimizer model.
        # Closing this needs structural egress isolation (per-role/per-eval
        # endpoints), not path-scoping — deferred as it requires an adversarial
        # producer.
        model = value.get("model") if isinstance(value, dict) else None
        if not isinstance(model, str) or model not in scope.allowed_models:
            await log_request(status=403, error="model_denied")
            return _provider_error(
                403, "model is not allowed for this scope", "model_denied"
            )
        if not attribution or any(
            not (character.isalnum() or character in "_.-") for character in attribution
        ):
            return _provider_error(
                400, "invalid inference attribution", "invalid_attribution"
            )
        try:
            await store.reserve(scope_name, attribution)
        except InferenceBudgetExceeded as error:
            await log_request(status=429, error="budget_exhausted")
            return _provider_error(429, str(error), "budget_exhausted")

        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() in _REQUEST_HEADERS
        }
        headers["authorization"] = f"Bearer {upstream_api_key}"
        url = f"{base_url}/{endpoint}"
        try:
            upstream = await app.state.upstream.send(
                app.state.upstream.build_request(
                    "POST",
                    url,
                    params=request.query_params,
                    content=body,
                    headers=headers,
                ),
                stream=True,
            )
        except httpx.HTTPError:
            await asyncio.shield(
                store.complete(scope_name, attribution, upstream_error=True)
            )
            await log_request(status=502, error="upstream_error", upstream_error=True)
            return _provider_error(
                502, "upstream inference request failed", "upstream_error"
            )
        except BaseException:
            await asyncio.shield(
                store.complete(scope_name, attribution, upstream_error=True)
            )
            raise
        response_headers = {
            name: value
            for name, value in upstream.headers.items()
            if name.lower() in _RESPONSE_HEADERS
        }
        is_stream = "text/event-stream" in upstream.headers.get("content-type", "")
        if is_stream:
            observed = _StreamingUsage()
            captured = request_log.capture() if request_log is not None else None

            async def chunks() -> AsyncIterator[bytes]:
                failed = upstream.status_code >= 400

                async def finish() -> None:
                    try:
                        await upstream.aclose()
                    finally:
                        await store.complete(
                            scope_name,
                            attribution,
                            input_tokens=observed.tokens[0],
                            output_tokens=observed.tokens[1],
                            total_tokens=observed.tokens[2],
                            cached_input_tokens=observed.tokens[3],
                            upstream_error=failed,
                        )
                        await log_request(
                            status=upstream.status_code,
                            stream=True,
                            tokens=observed.tokens,
                            response=(
                                captured.snapshot() if captured is not None else None
                            ),
                            upstream_error=failed,
                        )

                try:
                    async for chunk in upstream.aiter_bytes():
                        observed.feed(chunk)
                        if captured is not None:
                            captured.feed(chunk)
                        yield chunk
                except asyncio.CancelledError:
                    # OpenAI clients commonly stop consuming an SSE stream as soon
                    # as they observe its terminal response event. That downstream
                    # cancellation is not an upstream provider failure.
                    raise
                except httpx.HTTPError:
                    failed = True
                    raise
                except BaseException:
                    failed = True
                    raise
                finally:
                    await asyncio.shield(finish())

            return StreamingResponse(
                chunks(),
                status_code=upstream.status_code,
                headers=response_headers,
                media_type="text/event-stream",
            )

        try:
            content = await upstream.aread()
            try:
                tokens = _usage(json.loads(content))
            except (json.JSONDecodeError, UnicodeDecodeError):
                tokens = (0, 0, 0, 0)
        except BaseException:
            await asyncio.shield(
                store.complete(scope_name, attribution, upstream_error=True)
            )
            raise
        finally:
            await upstream.aclose()
        await asyncio.shield(
            store.complete(
                scope_name,
                attribution,
                input_tokens=tokens[0],
                output_tokens=tokens[1],
                total_tokens=tokens[2],
                cached_input_tokens=tokens[3],
                upstream_error=upstream.status_code >= 400,
            )
        )
        await log_request(
            status=upstream.status_code,
            tokens=tokens,
            response=request_log.body(content) if request_log is not None else None,
            upstream_error=upstream.status_code >= 400,
        )
        return Response(
            content=content,
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
        )

    return app


def load_inference_gateway_config(path: Path | str) -> InferenceGatewayConfig:
    return InferenceGatewayConfig.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def serve_inference_gateway(
    *,
    config_path: Path | str,
    host: str = "0.0.0.0",
    port: int = 8001,
) -> None:
    import uvicorn

    config = load_inference_gateway_config(config_path)
    upstream_api_key = os.environ.get(config.upstream_api_key_env)
    if not upstream_api_key:
        raise RuntimeError(
            f"gateway upstream credential {config.upstream_api_key_env} is missing"
        )
    upstream_base_url = (
        os.environ.get(config.upstream_base_url_env)
        if config.upstream_base_url_env is not None
        else None
    )
    uvicorn.run(
        create_inference_gateway_app(
            config=config,
            upstream_api_key=upstream_api_key,
            upstream_base_url=upstream_base_url,
        ),
        host=host,
        port=port,
    )
