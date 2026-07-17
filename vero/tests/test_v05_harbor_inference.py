from __future__ import annotations

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
        "output_tokens": 7,
        "total_tokens": 18,
    }


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


def test_gateway_reloads_usage_and_denies_disallowed_endpoints(tmp_path):
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
        endpoint = client.post(
            "/scopes/producer/optimizer/v1/files",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test"},
        )
        exhausted = client.post(
            "/scopes/producer/optimizer/v1/responses",
            headers={"Authorization": "Bearer scoped-token"},
            json={"model": "gpt-test", "input": "hello"},
        )

    assert endpoint.status_code == 403
    assert exhausted.status_code == 429
    assert app.state.usage_store.ledger.scopes["producer"].active_requests == 0


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
