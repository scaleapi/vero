from __future__ import annotations

import asyncio
import json

import httpx
from fastapi.testclient import TestClient

from vero.gateway.inference import (
    InferenceGatewayConfig,
    InferenceRequestLogConfig,
    InferenceScopeConfig,
    _unsupported_params,
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
    # 402, not 429: budget exhaustion is terminal, and 429 would be retried
    # by EvaluationLimits.retry_status_codes and by target-agent SDKs.
    assert exhausted.status_code == 402
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


def test_gateway_folds_anthropic_cache_tokens_into_input(tmp_path):
    """Anthropic counts only the uncached slice of the prompt as input_tokens.

    A cached optimizer turn answers ``"input_tokens": 2`` and carries the real
    prompt in the cache siblings. Metering the bare 2 read producer input about
    four orders of magnitude low on every live run -- output was right, so it
    looked plausible -- and left the producer scope budget effectively unbound.
    """

    def upstream(request: httpx.Request):
        if b'"stream": true' in request.content or b'"stream":true' in request.content:
            # Streaming splits usage: input arrives nested under `message` in
            # message_start, output accumulates in message_delta.
            payload = (
                'event: message_start\ndata: {"type":"message_start","message":'
                '{"usage":{"input_tokens":2,"cache_read_input_tokens":900,'
                '"cache_creation_input_tokens":100,"output_tokens":1}}}\n\n'
                'event: message_delta\ndata: {"type":"message_delta",'
                '"usage":{"output_tokens":40}}\n\n'
            )
            return httpx.Response(
                200, content=payload, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(
            200,
            json={
                "usage": {
                    "input_tokens": 2,
                    "cache_read_input_tokens": 4000,
                    "cache_creation_input_tokens": 1000,
                    "output_tokens": 50,
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
            "/scopes/producer/optimizer/v1/messages",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "messages": []},
        )
        client.post(
            "/scopes/producer/stream-1/v1/messages",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "messages": [], "stream": True},
        )

    scope = json.loads((tmp_path / "usage.json").read_text())["scopes"]["producer"]
    blocking = scope["attributions"]["optimizer"]
    assert blocking["input_tokens"] == 5002  # 2 uncached + 4000 read + 1000 written
    assert blocking["cached_input_tokens"] == 4000  # the documented subset
    assert blocking["total_tokens"] == 5052  # derived; Anthropic sends no total
    streamed = scope["attributions"]["stream-1"]
    assert streamed["input_tokens"] == 1002
    assert streamed["cached_input_tokens"] == 900
    # message_delta must not clobber the input that only message_start carried,
    # and the total must follow the raised output rather than stay at 1003.
    assert streamed["output_tokens"] == 40
    assert streamed["total_tokens"] == 1042
    assert scope["input_tokens"] == 6004


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

    assert exhausted.status_code == 402
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
    from vero.gateway.inference import InferenceRequestLogConfig

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
    assert [record["status"] for record in records] == [200, 200, 403, 402]
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
    from vero.gateway.inference import InferenceRequestLogConfig

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
    from vero.gateway.inference import InferenceRequestLog, InferenceRequestLogConfig

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
    total = sum(len(path.read_text().splitlines()) for path in files)
    assert total == 6


def _attributed_config(tmp_path, **kwargs):
    from vero.gateway.inference import InferenceRequestLogConfig

    return _config(tmp_path, **kwargs).model_copy(
        update={
            "request_log": InferenceRequestLogConfig(
                directory=str(tmp_path / "requests"),
                body_bytes=1024,
                attribution=True,
            )
        }
    )


def test_gateway_attribution_threads_stateful_and_stateless_requests(tmp_path):
    calls = {"n": 0}

    def upstream(request: httpx.Request):
        calls["n"] += 1
        if b'"stream": true' in request.content or b'"stream":true' in request.content:
            payload = (
                f'event: response.created\ndata: {{"type":"response.created",'
                f'"response":{{"id":"resp_stream{calls["n"]}"}}}}\n\n'
                'event: response.completed\ndata: {"type":"response.completed",'
                '"response":{"usage":{"input_tokens":1,"output_tokens":1,'
                '"total_tokens":2}}}\n\n'
            )
            return httpx.Response(
                200, content=payload, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(
            200,
            json={"id": f"resp_{calls['n']}", "usage": {"input_tokens": 1}},
        )

    app = create_inference_gateway_app(
        config=_attributed_config(tmp_path, max_requests=None, max_tokens=None),
        upstream_api_key="upstream-secret",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer scoped-token"}
        # responses API: root turn, then a stateful follow-up with no prompt
        client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers=headers,
            json={"model": "gpt-test", "input": "Solve task 42: find the answer"},
        )
        client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers=headers,
            json={"model": "gpt-test", "previous_response_id": "resp_1", "input": []},
        )
        # a streamed root and a follow-up chained onto its SSE-delivered id
        client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers=headers,
            json={
                "model": "gpt-test",
                "input": "Another task entirely",
                "stream": True,
            },
        )
        client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers=headers,
            json={
                "model": "gpt-test",
                "previous_response_id": "resp_stream3",
                "input": [],
            },
        )
        # anthropic-messages shape: giant system prompt, threading keys off the
        # first user message; the second call resends history (stateless)
        system = "x" * 30000
        client.post(
            "/scopes/producer/optimizer/v1/messages",
            headers=headers,
            json={
                "model": "gpt-test",
                "system": system,
                "messages": [
                    {"role": "user", "content": "Solve task 42: find the answer"},
                ],
            },
        )
        client.post(
            "/scopes/producer/optimizer/v1/messages",
            headers=headers,
            json={
                "model": "gpt-test",
                "system": system,
                "messages": [
                    {"role": "user", "content": "Solve task 42: find the answer"},
                    {"role": "assistant", "content": [{"type": "text", "text": "hm"}]},
                    {"role": "user", "content": "continue"},
                ],
            },
        )

    records = _log_records(tmp_path)
    assert len(records) == 6
    threads = [record.get("thread_id") for record in records]
    assert all(threads)
    # stateful follow-ups inherit their root's thread
    assert threads[1] == threads[0]
    assert threads[3] == threads[2]
    assert threads[2] != threads[0]
    # stateless resends group by first-user-message digest — and the messages
    # conversation shares its root text with the first responses thread
    assert threads[4] == threads[5] == threads[0]
    assert records[0]["root_snippet"].startswith("Solve task 42")
    assert records[0]["thread_root_digest"] == records[4]["thread_root_digest"]
    # chained records carry the thread but no root fields
    assert "root_snippet" not in records[1]


def test_gateway_attribution_never_breaks_proxying(tmp_path):
    app = create_inference_gateway_app(
        config=_attributed_config(tmp_path, max_requests=None, max_tokens=None),
        upstream_api_key="upstream-secret",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"id": 7, "usage": {}})
        ),
    )
    hostile_bodies = [
        {"model": "gpt-test", "input": {"nested": {"weird": True}}},
        {"model": "gpt-test", "messages": [{"role": "user", "content": [1, None]}]},
        {"model": "gpt-test", "messages": "not-a-list", "previous_response_id": 9},
        {"model": "gpt-test", "input": [{"role": "user", "content": {"a": "b"}}]},
    ]
    with TestClient(app) as client:
        for body in hostile_bodies:
            response = client.post(
                "/scopes/producer/optimizer/v1/responses",
                headers={"Authorization": "Bearer scoped-token"},
                json=body,
            )
            assert response.status_code == 200
    assert len(_log_records(tmp_path)) == len(hostile_bodies)


def test_gateway_attribution_disabled_by_default_and_memory_bounded(tmp_path):
    from vero.gateway.inference import _RequestAttributor

    # default config: no attributor, no thread fields in records
    app = create_inference_gateway_app(
        config=_logged_config(tmp_path, max_requests=None, max_tokens=None),
        upstream_api_key="upstream-secret",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"usage": {}})
        ),
    )
    assert app.state.request_attributor is None
    with TestClient(app) as client:
        client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "input": "hello"},
        )
    (record,) = _log_records(tmp_path)
    assert "thread_id" not in record

    # FIFO caps hold under churn
    attributor = _RequestAttributor()
    attributor._CAP = 10
    for index in range(50):
        fields = attributor.stamp_request({"input": f"root {index}"})
        attributor.register_response(fields, f'{{"id":"resp_{index}"}}'.encode())
    assert len(attributor._root_threads) <= 10
    assert len(attributor._response_threads) <= 10
    assert attributor.errors == 0


def test_budget_exhaustion_status_is_not_retryable():
    """Budget exhaustion must not wear a status anything will retry.

    It is terminal -- waiting never restores the quota -- so returning 429 made
    every layer above retry it: EvaluationLimits.retry_status_codes defaults to
    [429, 503, 529], and target-agent SDKs retry 429 on their own. officeqa run
    #2 reissued 3672 doomed requests after exhausting its finalization scope.
    """
    from vero.evaluation.models import EvaluationLimits

    budget_exhausted_status = 402
    assert budget_exhausted_status not in EvaluationLimits().retry.retry_status_codes
    # The transient ones stay retryable.
    assert 429 in EvaluationLimits().retry.retry_status_codes


def test_gateway_request_log_records_the_upstream_cost_headers(tmp_path):
    """Per-request cost, because no other method attributes spend to a run.

    Per-key billed deltas stop decomposing as soon as a credential carries
    concurrent work, and reconstructing from a public price table ran 3.1x high
    across a real pass. The proxy states the answer on every response; this keeps
    it. Covers the streamed path too, where the cost header arrives with the
    response head long before the body finishes.
    """
    stream_payload = (
        'event: response.completed\ndata: {"type":"response.completed",'
        '"response":{"usage":{"input_tokens":5,"output_tokens":3,'
        '"total_tokens":8}}}\n\n'
    )
    cost_headers = {
        "x-litellm-response-cost": "0.00031",
        "x-litellm-response-cost-original": "0.00040",
        "x-litellm-response-cost-discount-amount": "0.00009",
        # Deliberately unparseable: a malformed header must be skipped, never
        # raise into the proxy path.
        "x-litellm-response-cost-margin-amount": "not-a-number",
    }

    def upstream(request: httpx.Request):
        if b'"stream": true' in request.content or b'"stream":true' in request.content:
            return httpx.Response(
                200,
                content=stream_payload,
                headers={"content-type": "text/event-stream", **cost_headers},
            )
        return httpx.Response(
            200,
            json={"id": "r", "usage": {"input_tokens": 11, "output_tokens": 7}},
            headers=cost_headers,
        )

    app = create_inference_gateway_app(
        config=_logged_config(tmp_path, max_requests=4),
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

    records = _log_records(tmp_path)
    assert len(records) == 2
    for record in records:
        assert record["upstream_cost_usd"] == 0.00031
        assert record["upstream_cost_usd_original"] == 0.00040
        assert record["upstream_cost_discount_usd"] == 0.00009
        # The unparseable margin header is absent rather than zero, so a missing
        # value is never mistaken for a genuine zero cost.
        assert "upstream_cost_margin_usd" not in record

    # Summing the field is the whole point: this is what makes per-cell cost exact.
    assert round(sum(r["upstream_cost_usd"] for r in records), 5) == 0.00062


def test_gateway_request_log_omits_cost_when_the_upstream_sends_none(tmp_path):
    """A provider that reports no cost must not produce a spurious 0.0."""

    def upstream(request: httpx.Request):
        return httpx.Response(200, json={"id": "r", "usage": {"input_tokens": 3}})

    app = create_inference_gateway_app(
        config=_logged_config(tmp_path),
        upstream_api_key="upstream-secret",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "input": "hi"},
        )

    record = _log_records(tmp_path)[0]
    assert not any(key.startswith("upstream_cost") for key in record)


_UNSUPPORTED = (
    "litellm.UnsupportedParamsError: fireworks_ai does not support parameters: "
    "['tool_choice'], for model=accounts/fireworks/models/kimi-k3."
)


def test_gateway_retries_without_parameters_the_provider_refuses(tmp_path):
    """A provider that refuses tool_choice must not kill the whole request.

    Fireworks answers 400 for `tool_choice`, which every tool-driving harness
    sends, so the model is unusable through the gateway -- the run dies on its
    first real request. Neither documented workaround is general: two configure
    the upstream proxy we do not own, and `allowed_openai_params` is honoured on
    /chat/completions but ignored on /responses, which is the endpoint an
    openai-prefixed harness actually uses.

    Retrying without the blamed parameter is endpoint-agnostic and engages only
    on an already-failed request.
    """
    seen = []

    def upstream(request: httpx.Request):
        payload = json.loads(request.content)
        seen.append(payload)
        if "tool_choice" in payload:
            return httpx.Response(400, json={"error": {"message": _UNSUPPORTED}})
        return httpx.Response(200, json={"id": "response", "usage": {}})

    app = create_inference_gateway_app(
        config=_config(tmp_path, max_requests=4),
        upstream_api_key="upstream-secret",
        upstream_base_url="https://provider.example/v1",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        # /responses is the endpoint that matters: this is where the documented
        # allowed_openai_params escape hatch does not work.
        ok = client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "tool_choice": "auto", "input": []},
        )

    assert ok.status_code == 200
    assert len(seen) == 2, "expected one rejected attempt then one retry"
    assert seen[0]["tool_choice"] == "auto"
    assert "tool_choice" not in seen[1]
    assert seen[1]["model"] == "gpt-test", "the rest of the request must survive"


def test_gateway_does_not_retry_when_the_provider_accepts_the_parameter(tmp_path):
    """Providers that accept tool_choice keep it, and their exact semantics.

    This is the reason the fix is a retry rather than an unconditional strip: a
    strip would silently change behaviour for every harness that relies on
    tool_choice against a provider that supports it.
    """
    seen = []

    def upstream(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "response", "usage": {}})

    app = create_inference_gateway_app(
        config=_config(tmp_path, max_requests=4),
        upstream_api_key="upstream-secret",
        upstream_base_url="https://provider.example/v1",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        client.post(
            "/scopes/producer/optimizer/v1/chat/completions",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "tool_choice": "required", "messages": []},
        )

    assert len(seen) == 1, "a successful request must not be retried"
    assert seen[0]["tool_choice"] == "required"


def test_gateway_does_not_retry_a_400_that_blames_nothing_droppable(tmp_path):
    """An unrelated 400 is relayed once, not retried into a second charge."""
    seen = []

    def upstream(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    app = create_inference_gateway_app(
        config=_config(tmp_path, max_requests=4),
        upstream_api_key="upstream-secret",
        upstream_base_url="https://provider.example/v1",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        failed = client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "tool_choice": "auto", "input": []},
        )

    assert failed.status_code == 400
    assert len(seen) == 1


def test_unsupported_params_survives_json_escaped_quoting():
    """The blamed names must parse whatever quoting the upstream uses.

    Today's upstream builds the list with Python's repr, so it arrives
    single-quoted and JSON leaves it alone. Were it ever double-quoted, JSON
    would escape it to ``[\\"tool_choice\\"]``; matching the raw body then yields
    ``tool_choice\\`` with a trailing backslash, which fails the caller's
    "is this parameter in the request" check. The retry would silently never fire
    -- a worse failure than the 400, because nothing says why.
    """
    message = (
        "litellm.UnsupportedParamsError: fireworks_ai does not support parameters: "
    )
    single = json.dumps({"error": {"message": message + "['tool_choice']"}}).encode()
    double = json.dumps({"error": {"message": message + '["tool_choice"]'}}).encode()

    assert _unsupported_params(single) == ["tool_choice"]
    assert _unsupported_params(double) == ["tool_choice"]
    # Several parameters, and an unrelated error, both behave.
    many = json.dumps(
        {"error": {"message": "does not support parameters: ['tool_choice', 'top_k']"}}
    ).encode()
    assert _unsupported_params(many) == ["tool_choice", "top_k"]
    assert _unsupported_params(b'{"error": {"message": "rate limited"}}') == []
    # A non-JSON body still gets a best-effort read rather than giving up.
    assert _unsupported_params(b"does not support parameters: ['tool_choice']") == [
        "tool_choice"
    ]


def test_gateway_records_dropped_params_in_the_request_log(tmp_path):
    """A degraded request must be auditable, not merely successful.

    The retry silently changes what was asked of the provider. If the audit field
    regresses, every future degraded request becomes invisible and the only signal
    left is a behaviour difference nobody is looking for.
    """

    def upstream(request: httpx.Request):
        payload = json.loads(request.content)
        if "tool_choice" in payload:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": (
                            "litellm.UnsupportedParamsError: fireworks_ai does not "
                            "support parameters: ['tool_choice'],"
                        )
                    }
                },
            )
        return httpx.Response(200, json={"id": "response", "usage": {}})

    config = _config(tmp_path, max_requests=4).model_copy(
        update={
            "request_log": InferenceRequestLogConfig(
                directory=str(tmp_path / "requests")
            )
        }
    )
    app = create_inference_gateway_app(
        config=config,
        upstream_api_key="upstream-secret",
        upstream_base_url="https://provider.example/v1",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        retried = client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "tool_choice": "auto", "input": []},
        )
        untouched = client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "input": []},
        )
    assert retried.status_code == 200
    assert untouched.status_code == 200

    records = [
        json.loads(line)
        for path in sorted((tmp_path / "requests").glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 2, records
    assert records[0]["dropped_params"] == ["tool_choice"]
    # A request nothing was dropped from must not carry a misleading empty list.
    assert records[1].get("dropped_params") is None


def test_gateway_learns_a_refusal_once_instead_of_retrying_every_request(tmp_path):
    """The discovery must cost one extra round-trip, not one per request.

    A tool-driving harness sends `tool_choice` on nearly every call (52 of 66 in a
    conformance run), so retrying each one would double that traffic against the
    RPM limit to re-learn something the gateway already knows.
    """
    attempts = []

    def upstream(request: httpx.Request):
        payload = json.loads(request.content)
        attempts.append("tool_choice" in payload)
        if "tool_choice" in payload:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": (
                            "fireworks_ai does not support parameters: ['tool_choice'],"
                        )
                    }
                },
            )
        return httpx.Response(200, json={"id": "response", "usage": {}})

    app = create_inference_gateway_app(
        config=_config(tmp_path, max_requests=20, max_tokens=10_000),
        upstream_api_key="upstream-secret",
        upstream_base_url="https://provider.example/v1",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        for _ in range(5):
            response = client.post(
                "/scopes/producer/optimizer/v1/responses",
                headers={"Authorization": "Bearer scoped-token"},
                json={"model": "gpt-test", "tool_choice": "auto", "input": []},
            )
            assert response.status_code == 200

    # Five client requests cost six upstream calls: the first pays a rejected
    # attempt plus its retry, the remaining four are stripped up front and go
    # straight through. Retrying per request would have cost ten.
    assert attempts == [True, False, False, False, False, False], attempts
    assert attempts.count(True) == 1, "the refusal must be discovered only once"


def test_model_alias_rewrites_upstream_after_the_allow_list(tmp_path):
    """A declared alias changes which deployment serves the request, not what the
    caller may ask for.

    The distinction is the point. `allowed_models` is what makes "this cell ran
    gpt-test" a true statement, so the alias is applied only once that check has
    passed: the caller still cannot reach an unlisted model by naming its alias
    key, and the log records both names.
    """
    observed: list[dict] = []

    def upstream(request: httpx.Request):
        observed.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "response", "usage": {}})

    config = InferenceGatewayConfig(
        state_path=str(tmp_path / "usage.json"),
        request_log=InferenceRequestLogConfig(directory=str(tmp_path / "log")),
        scopes={
            "producer": InferenceScopeConfig(
                token_sha256=token_digest("scoped-token"),
                allowed_models=["gpt-test", "plain"],
                model_aliases={"gpt-test": "vendor_x/gpt-test"},
                max_concurrency=1,
            )
        },
    )
    app = create_inference_gateway_app(
        config=config,
        upstream_api_key="upstream-secret",
        upstream_base_url="https://provider.example/v1",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        aliased = client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "input": "hello"},
        )
        untouched = client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "plain", "input": "hello"},
        )
        # The alias TARGET is not itself allow-listed, so naming it directly is
        # still refused. An alias must not become a second way in.
        target_direct = client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "vendor_x/gpt-test", "input": "hello"},
        )

    assert aliased.status_code == 200
    assert untouched.status_code == 200
    assert target_direct.status_code == 403
    assert target_direct.json()["error"]["code"] == "model_denied"

    # Rewritten on the way upstream; the unaliased model passes through as-is.
    assert [payload["model"] for payload in observed] == [
        "vendor_x/gpt-test",
        "plain",
    ]

    records = [
        json.loads(line)
        for path in sorted((tmp_path / "log").glob("requests-*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    proxied = [record for record in records if record["status"] == 200]
    assert [(r["model"], r.get("aliased_from")) for r in proxied] == [
        ("vendor_x/gpt-test", "gpt-test"),
        ("plain", None),
    ]
