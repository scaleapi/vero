from __future__ import annotations

import asyncio
import json

import httpx
from fastapi.testclient import TestClient

from vero.harbor.inference import (
    InferenceGatewayConfig,
    InferenceScopeConfig,
    create_inference_gateway_app,
    token_digest,
)


def _config(tmp_path, *, max_requests=2, max_tokens=100):
    return InferenceGatewayConfig(
        state_path=str(tmp_path / "usage.json"),
        scopes={
            "producer": InferenceScopeConfig(
                token_sha256=token_digest("scoped-token"),
                allowed_models=["gpt-test"],
                max_requests=max_requests,
                max_tokens=max_tokens,
                max_concurrency=1,
            )
        },
    )


def test_gateway_replaces_credentials_enforces_scope_and_persists_usage(tmp_path):
    observed = []

    def upstream(request: httpx.Request):
        observed.append(request)
        return httpx.Response(
            200,
            json={
                "id": "response",
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                },
            },
        )

    app = create_inference_gateway_app(
        config=_config(tmp_path, max_requests=1),
        upstream_api_key="upstream-secret",
        upstream_base_url="https://provider.example/v1",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        denied = client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer wrong"},
            json={"model": "gpt-test", "input": "hello"},
        )
        wrong_model = client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "other", "input": "hello"},
        )
        accepted = client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "input": "hello"},
        )
        exhausted = client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "input": "again"},
        )
        usage = client.get(
            "/usage/producer",
            headers={"Authorization": "Bearer scoped-token"},
        )

    assert denied.status_code == 403
    assert wrong_model.status_code == 403
    assert accepted.status_code == 200
    assert exhausted.status_code == 429
    assert len(observed) == 1
    assert observed[0].headers["authorization"] == "Bearer upstream-secret"
    assert b"scoped-token" not in observed[0].content
    assert usage.json()["requests"] == 1
    assert usage.json()["total_tokens"] == 18
    assert usage.json()["remaining_requests"] == 0
    persisted = json.loads((tmp_path / "usage.json").read_text())
    assert persisted["scopes"]["producer"]["active_requests"] == 0
    assert persisted["scopes"]["producer"]["attributions"]["optimizer"] == {
        "requests": 1,
        "upstream_errors": 0,
        "input_tokens": 11,
        "cached_input_tokens": 0,
        "output_tokens": 7,
        "total_tokens": 18,
    }


def test_gateway_records_cached_input_tokens(tmp_path):
    def upstream(request: httpx.Request):
        if b'"stream": true' in request.content or b'"stream":true' in request.content:
            payload = (
                'event: response.completed\ndata: {"type":"response.completed",'
                '"response":{"usage":{"input_tokens":40,"output_tokens":4,'
                '"total_tokens":44,"input_tokens_details":{"cached_tokens":32}}}}\n\n'
            )
            return httpx.Response(
                200, content=payload, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(
            200,
            json={
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                    "prompt_tokens_details": {"cached_tokens": 16},
                }
            },
        )

    app = create_inference_gateway_app(
        config=_config(tmp_path, max_requests=None, max_tokens=None),
        upstream_api_key="upstream-secret",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        client.post(
            "/scopes/producer/optimizer/v1/chat/completions",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "messages": []},
        )
        client.post(
            "/scopes/producer/eval-1/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "input": "hi", "stream": True},
        )

    persisted = json.loads((tmp_path / "usage.json").read_text())
    scope = persisted["scopes"]["producer"]
    # both the chat (prompt_tokens_details) and responses (input_tokens_details)
    # shapes are recognized, per-attribution and in the scope total
    assert scope["cached_input_tokens"] == 48
    assert scope["attributions"]["optimizer"]["cached_input_tokens"] == 16
    assert scope["attributions"]["eval-1"]["cached_input_tokens"] == 32


def test_gateway_records_usage_without_enforcing_omitted_limits(tmp_path):
    def upstream(_request: httpx.Request):
        return httpx.Response(
            200,
            json={
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                }
            },
        )

    app = create_inference_gateway_app(
        config=_config(tmp_path, max_requests=None, max_tokens=None),
        upstream_api_key="upstream-secret",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        responses = [
            client.post(
                "/scopes/producer/optimizer/v1/responses",
                headers={"Authorization": "Bearer scoped-token"},
                json={"model": "gpt-test", "input": f"request {index}"},
            )
            for index in range(3)
        ]
        usage = client.get(
            "/usage/producer",
            headers={"Authorization": "Bearer scoped-token"},
        ).json()

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert usage["requests"] == 3
    assert usage["total_tokens"] == 54
    assert usage["max_requests"] is None
    assert usage["max_tokens"] is None
    assert usage["remaining_requests"] is None
    assert usage["remaining_tokens"] is None


def test_gateway_meters_streaming_responses(tmp_path):
    payload = (
        'event: response.created\ndata: {"type":"response.created"}\n\n'
        'event: response.completed\ndata: {"type":"response.completed",'
        '"response":{"usage":{"input_tokens":5,"output_tokens":3,'
        '"total_tokens":8}}}\n\n'
    )

    def upstream(_request: httpx.Request):
        return httpx.Response(
            200,
            content=payload,
            headers={"content-type": "text/event-stream"},
        )

    app = create_inference_gateway_app(
        config=_config(tmp_path),
        upstream_api_key="upstream-secret",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "input": "hello", "stream": True},
        )

    assert response.status_code == 200
    assert "response.completed" in response.text
    persisted = json.loads((tmp_path / "usage.json").read_text())
    usage = persisted["scopes"]["producer"]
    assert usage["active_requests"] == 0
    assert usage["total_tokens"] == 8


def test_gateway_reloads_usage_and_enforces_budget(tmp_path):
    config = _config(tmp_path, max_requests=1)
    (tmp_path / "usage.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scopes": {
                    "producer": {
                        "requests": 1,
                        "upstream_errors": 0,
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                        "active_requests": 4,
                        "attributions": {},
                    }
                },
            }
        )
    )
    app = create_inference_gateway_app(
        config=config,
        upstream_api_key="upstream-secret",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={})),
    )
    with TestClient(app) as client:
        exhausted = client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "input": "hello"},
        )

    assert exhausted.status_code == 429
    assert app.state.usage_store.ledger.scopes["producer"].active_requests == 0


def test_gateway_forwards_arbitrary_endpoints_to_upstream(tmp_path):
    # Provider-agnostic passthrough: an endpoint the gateway has never heard of
    # (here the Anthropic Messages surface) is forwarded to the upstream proxy,
    # with the same scope-token + model + budget enforcement as any other call.
    observed = []

    def upstream(request: httpx.Request):
        observed.append(request)
        return httpx.Response(
            200, json={"usage": {"input_tokens": 3, "output_tokens": 2}}
        )

    app = create_inference_gateway_app(
        config=_config(tmp_path),
        upstream_api_key="upstream-secret",
        upstream_base_url="https://provider.example/v1",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        forwarded = client.post(
            "/scopes/producer/optimizer/v1/messages",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "messages": []},
        )
        wrong_model = client.post(
            "/scopes/producer/optimizer/v1/messages",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "other", "messages": []},
        )

    assert forwarded.status_code == 200
    assert wrong_model.status_code == 403  # model allow-list still applies
    assert str(observed[0].url) == "https://provider.example/v1/messages"
    assert observed[0].headers["authorization"] == "Bearer upstream-secret"


def test_gateway_accepts_scope_token_via_x_api_key_header(tmp_path):
    # Anthropic/Claude clients send the scope token as `x-api-key`, not
    # `Authorization: Bearer`; the gateway must accept either.
    observed = []

    def upstream(request: httpx.Request):
        observed.append(request)
        return httpx.Response(200, json={"usage": {"input_tokens": 1}})

    app = create_inference_gateway_app(
        config=_config(tmp_path),
        upstream_api_key="upstream-secret",
        upstream_base_url="https://provider.example/v1",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        via_x_api_key = client.post(
            "/scopes/producer/optimizer/v1/messages",
            headers={"x-api-key": "scoped-token"},
            json={"model": "gpt-test", "messages": []},
        )
        no_token = client.post(
            "/scopes/producer/optimizer/v1/messages",
            headers={"x-api-key": "wrong"},
            json={"model": "gpt-test", "messages": []},
        )

    assert via_x_api_key.status_code == 200
    assert no_token.status_code == 403
    # the upstream never sees the scope token; it gets the upstream credential
    assert observed[0].headers["authorization"] == "Bearer upstream-secret"


def test_gateway_rejects_path_traversal_endpoints(tmp_path):
    app = create_inference_gateway_app(
        config=_config(tmp_path),
        upstream_api_key="upstream-secret",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={})),
    )
    with TestClient(app) as client:
        # Percent-encoded so the client doesn't normalize `..` away before it
        # reaches the route; the server-side guard must still reject it.
        traversal = client.post(
            "/scopes/producer/optimizer/v1/%2e%2e/admin",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test"},
        )

    assert traversal.status_code == 400
    assert traversal.json()["error"]["code"] == "invalid_endpoint"


def test_gateway_releases_concurrency_reservation_on_upstream_failure(tmp_path):
    def unavailable(request: httpx.Request):
        raise httpx.ConnectError("offline", request=request)

    app = create_inference_gateway_app(
        config=_config(tmp_path),
        upstream_api_key="upstream-secret",
        transport=httpx.MockTransport(unavailable),
    )
    with TestClient(app) as client:
        response = client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "input": "hello"},
        )

    assert response.status_code == 502
    usage = app.state.usage_store.ledger.scopes["producer"]
    assert usage.active_requests == 0
    assert usage.upstream_errors == 1


def _logged_config(tmp_path, **kwargs):
    from vero.harbor.inference import InferenceRequestLogConfig

    config = _config(tmp_path, **kwargs)
    return config.model_copy(
        update={
            "request_log": InferenceRequestLogConfig(
                directory=str(tmp_path / "requests"),
                body_bytes=64,
            )
        }
    )


def _log_records(tmp_path):
    records = []
    for path in sorted((tmp_path / "requests").glob("requests-*.jsonl")):
        for line in path.read_text().splitlines():
            records.append(json.loads(line))
    return records


def test_gateway_request_log_captures_responses_streams_and_denials(tmp_path):
    stream_payload = (
        'event: response.completed\ndata: {"type":"response.completed",'
        '"response":{"usage":{"input_tokens":5,"output_tokens":3,'
        '"total_tokens":8}}}\n\n'
    )

    def upstream(request: httpx.Request):
        if b'"stream": true' in request.content or b'"stream":true' in request.content:
            return httpx.Response(
                200,
                content=stream_payload,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={"id": "r", "usage": {"input_tokens": 11, "output_tokens": 7}},
        )

    app = create_inference_gateway_app(
        config=_logged_config(tmp_path, max_requests=2),
        upstream_api_key="upstream-secret",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "input": "hi"},
        )
        client.post(
            "/scopes/producer/eval-1/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "input": "hi", "stream": True},
        )
        client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "denied-model", "input": "hi"},
        )
        client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "input": "over budget"},
        )

    records = _log_records(tmp_path)
    assert [record["status"] for record in records] == [200, 200, 403, 429]
    plain, stream, denied, exhausted = records
    assert plain["scope"] == "producer"
    assert plain["attribution"] == "optimizer"
    assert plain["model"] == "gpt-test"
    assert plain["input_tokens"] == 11 and plain["output_tokens"] == 7
    assert "gpt-test" in plain["request"]["text"]
    assert '"id":"r"' in plain["response"]["text"]
    assert plain["latency_ms"] >= 0
    assert stream["stream"] is True
    assert stream["attribution"] == "eval-1"
    assert stream["total_tokens"] == 8
    # the head+tail capture keeps the stream's terminal usage frame
    assert "total_tokens" in (
        stream["response"]["text"] + stream["response"].get("tail", "")
    )
    assert denied["error"] == "model_denied" and denied["response"] is None
    assert exhausted["error"] == "budget_exhausted"


def test_gateway_request_log_truncates_and_rotates(tmp_path):
    from vero.harbor.inference import InferenceRequestLogConfig

    config = _config(tmp_path, max_requests=None, max_tokens=None).model_copy(
        update={
            "request_log": InferenceRequestLogConfig(
                directory=str(tmp_path / "requests"),
                body_bytes=32,
                rotate_bytes=1_048_576,
            )
        }
    )
    app = create_inference_gateway_app(
        config=config,
        upstream_api_key="upstream-secret",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"data": "y" * 200})
        ),
    )
    with TestClient(app) as client:
        client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "input": "x" * 500},
        )

    (record,) = _log_records(tmp_path)
    assert record["request"]["truncated"] is True
    assert record["request"]["bytes"] > 500
    assert len(record["request"]["text"]) <= 16
    assert "tail" in record["request"]
    assert record["response"]["truncated"] is True


def test_gateway_request_log_rotation_boundary(tmp_path):
    from vero.harbor.inference import InferenceRequestLog, InferenceRequestLogConfig

    log = InferenceRequestLog(
        InferenceRequestLogConfig(
            directory=str(tmp_path / "requests"),
            body_bytes=16384,
            rotate_bytes=1_048_576,
        )
    )
    # Force a tiny rotation threshold without violating the config floor.
    log.config = log.config.model_copy(update={"rotate_bytes": 400})

    async def fill():
        for index in range(6):
            await log.record(scope="producer", index=index, payload="z" * 100)

    asyncio.run(fill())
    files = sorted((tmp_path / "requests").glob("requests-*.jsonl"))
    assert len(files) > 1
    total = sum(
        len(path.read_text().splitlines()) for path in files
    )
    assert total == 6
