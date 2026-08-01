from __future__ import annotations

import asyncio
import gc
import json

import httpx
from fastapi.testclient import TestClient

from vero.gateway.inference import (
    InferenceGatewayConfig,
    InferenceScopeConfig,
    create_inference_gateway_app,
    token_digest,
)

_STREAM = (
    'event: response.created\ndata: {"type":"response.created"}\n\n'
    'event: response.completed\ndata: {"type":"response.completed",'
    '"response":{"usage":{"input_tokens":5,"output_tokens":3,'
    '"total_tokens":8}}}\n\n'
)


def _streaming_app(tmp_path):
    """A gateway whose upstream always answers with a two-event SSE stream."""

    def upstream(_request: httpx.Request):
        return httpx.Response(
            200,
            content=_STREAM,
            headers={"content-type": "text/event-stream"},
        )

    return create_inference_gateway_app(
        config=InferenceGatewayConfig(
            state_path=str(tmp_path / "usage.json"),
            scopes={
                "producer": InferenceScopeConfig(
                    token_sha256=token_digest("scoped-token"),
                    allowed_models=["gpt-test"],
                    # One permit, so a single leaked reservation wedges the whole
                    # scope and the wedge is observable in one request.
                    max_concurrency=1,
                )
            },
        ),
        upstream_api_key="upstream-secret",
        transport=httpx.MockTransport(upstream),
    )


def _scope(body: bytes) -> dict:
    # No "spec_version" in the asgi extension, so starlette takes its pre-2.4
    # branch and races stream_response against listen_for_disconnect in an anyio
    # task group. That is the branch uvicorn drives today and the one where the
    # body iterator can be cancelled before its first step.
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/scopes/producer/optimizer/v1/responses",
        "raw_path": b"/scopes/producer/optimizer/v1/responses",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"authorization", b"Bearer scoped-token"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }


def test_streaming_client_that_disconnects_first_returns_its_permit(tmp_path):
    """A client gone before the first chunk must not keep a concurrency permit.

    This is the wedge from the audit: starlette cancels its streaming task group
    the moment listen_for_disconnect sees http.disconnect, so an unprimed body
    iterator is never stepped, the generator's finally never runs, and the
    reservation it holds is never returned. After max_concurrency of these,
    reserve() waits on the semaphore forever with no timeout and no traceback.
    """
    app = _streaming_app(tmp_path)
    body = json.dumps({"model": "gpt-test", "input": "hello", "stream": True}).encode()

    async def drive() -> None:
        pending = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive() -> dict:
            # The request body first, then a client that has already hung up.
            return pending.pop(0) if pending else {"type": "http.disconnect"}

        sent: list[dict] = []

        async def send(message: dict) -> None:
            # A real ASGI server suspends here while it writes to the socket, and
            # that suspension is where the cancelled scope lands: anyio refuses to
            # cancel a task that has not started yet, so the streaming task always
            # gets as far as its first await and no further.
            await asyncio.sleep(0)
            sent.append(message)

        # What the app's own lifespan does. Spelled out because this test drives
        # the raw ASGI callable rather than TestClient, which cannot disconnect
        # before reading the response.
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    content=_STREAM,
                    headers={"content-type": "text/event-stream"},
                )
            )
        ) as client:
            app.state.upstream = client
            await app(_scope(body), receive, send)

            # The premise of the test: nothing of the stream reached the client,
            # so the disconnect really did land before the first chunk.
            assert not any(message.get("body") for message in sent)

            store = app.state.usage_store
            limiter = store._limits["producer"]
            # The permit comes back through asyncio's async-generator
            # finalization, which needs the suspended generator collected first:
            # the cancelled task's traceback keeps it in a cycle, so plain
            # reference counting will not do it. A generator that was never
            # started is not registered for finalization at all, so under the old
            # behaviour no number of collections brings its permit back.
            for _ in range(50):
                gc.collect()
                await asyncio.sleep(0)
                if not limiter.locked():
                    break

            assert store.ledger.scopes["producer"].active_requests == 0
            assert not limiter.locked()
            # The permit itself, not just the counter. With one permit in the
            # scope, a second reservation can only be granted if the first one
            # really came back, and hanging here forever is precisely how the two
            # lost runs presented, so the timeout is the assertion.
            await asyncio.wait_for(store.reserve("producer", "optimizer"), timeout=1)

    asyncio.run(drive())


def test_primed_stream_still_delivers_every_upstream_byte(tmp_path):
    """Not a regression test for the leak: a guard on the priming step itself.

    Priming consumes the first SSE event inside the handler, so this pins that the
    client still receives that event, once and in order. Nothing else in the suite
    compares a streamed body byte for byte, and a dropped first event would change
    what every target agent reads.
    """
    app = _streaming_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "input": "hello", "stream": True},
        )

    assert response.status_code == 200
    assert response.text == _STREAM
    usage = app.state.usage_store.ledger.scopes["producer"]
    assert usage.active_requests == 0
    assert usage.total_tokens == 8
