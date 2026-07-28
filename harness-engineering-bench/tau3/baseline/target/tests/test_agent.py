import base64
import json
import re
from types import SimpleNamespace

import pytest

from tau3_agent import Tau3Agent
import tau3_agent.agent as agent_module


def _agent(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gateway.invalid/v1")
    return Tau3Agent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-5.4-mini-2026-03-17",
        **kwargs,
    )


def _server(url="http://tau3-runtime:8000/mcp"):
    return SimpleNamespace(name="tau3-runtime", transport="streamable-http", url=url)


def _sse(message: dict, *, headers: str = "") -> str:
    head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n" + headers + "\r\n"
    return f"{head}\nevent: message\ndata: {json.dumps(message)}\n\n"


def test_agent_resolves_one_streamable_http_server(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, mcp_servers=[_server()])
    assert agent._server_url() == "http://tau3-runtime:8000/mcp"


def test_agent_requires_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="requires a Harbor model"):
        Tau3Agent(logs_dir=tmp_path)


def test_session_id_parsed_from_headers():
    stdout = "HTTP/1.1 200 OK\r\nMcp-Session-Id: abc123\r\n\r\ndata: {}"
    assert Tau3Agent._session_id_from(stdout) == "abc123"
    assert Tau3Agent._session_id_from("HTTP/1.1 200 OK\r\n\r\n") is None


def test_jsonrpc_prefers_result_over_other_events():
    stdout = (
        "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\"}\n"
        "data: {\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"ok\":true}}\n"
    )
    assert Tau3Agent._jsonrpc_from(stdout) == {
        "jsonrpc": "2.0",
        "id": 3,
        "result": {"ok": True},
    }


def test_openai_tools_excludes_start_conversation():
    tools = [
        {"name": "start_conversation", "description": "x", "inputSchema": {}},
        {"name": "send_message_to_user", "description": "y", "inputSchema": {}},
    ]
    names = {t["function"]["name"] for t in Tau3Agent._openai_tools(tools)}
    assert names == {"send_message_to_user"}


class _RecordingEnv:
    """Fake environment that decodes each base64 MCP payload and returns canned SSE."""

    def __init__(self):
        self.commands = []

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        if "curl" not in command:  # setup() probe
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        encoded = re.search(r"printf '%s' '([^']+)'", command).group(1)
        payload = json.loads(base64.b64decode(encoded))
        method = payload.get("method")
        if method == "initialize":
            return SimpleNamespace(
                return_code=0,
                stdout=_sse(
                    {"jsonrpc": "2.0", "id": payload["id"], "result": {}},
                    headers="mcp-session-id: sess-1\r\n",
                ),
                stderr="",
            )
        if method == "notifications/initialized":
            return SimpleNamespace(return_code=0, stdout="HTTP/1.1 202\r\n\r\n", stderr="")
        if method == "tools/list":
            tools = [
                {"name": n, "description": n, "inputSchema": {"type": "object"}}
                for n in (
                    "start_conversation",
                    "send_message_to_user",
                    "end_conversation",
                )
            ]
            return SimpleNamespace(
                return_code=0,
                stdout=_sse(
                    {"jsonrpc": "2.0", "id": payload["id"], "result": {"tools": tools}}
                ),
                stderr="",
            )
        # tools/call
        name = payload["params"]["name"]
        result = {"content": [{"type": "text", "text": f"ran:{name}"}]}
        return SimpleNamespace(
            return_code=0,
            stdout=_sse({"jsonrpc": "2.0", "id": payload["id"], "result": result}),
            stderr="",
        )


def _response(*, calls=None, text="", rid="r1"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text or None, tool_calls=calls or None)
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            prompt_tokens_details=SimpleNamespace(cached_tokens=3),
        ),
    )


def _call(name, arguments):
    return SimpleNamespace(
        id=f"c-{name}",
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


@pytest.mark.asyncio
async def test_setup_requires_curl_and_base64(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, mcp_servers=[_server()])

    class _OkEnv:
        async def exec(self, command, **kwargs):
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    await agent.setup(_OkEnv())  # no exception

    class _FailEnv:
        async def exec(self, command, **kwargs):
            return SimpleNamespace(return_code=1, stdout="", stderr="no curl")

    with pytest.raises(RuntimeError, match="curl and base64"):
        await agent.setup(_FailEnv())


@pytest.mark.asyncio
async def test_run_drives_loop_and_records_usage(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, mcp_servers=[_server()])
    env = _RecordingEnv()

    # turn 1: plain text -> send_message_to_user; turn 2: end_conversation tool call.
    script = [
        _response(text="Hello, how can I help?", rid="r1"),
        _response(calls=[_call("end_conversation", {"message": "bye"})], rid="r2"),
    ]

    class _FakeResponses:
        def __init__(self, script):
            self._script, self._i = script, 0

        async def create(self, **kwargs):
            item = self._script[self._i]
            self._i += 1
            return item

    class _FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=_FakeResponses(script))

    monkeypatch.setattr(agent_module, "AsyncOpenAI", lambda *a, **k: _FakeClient())

    context = SimpleNamespace(
        n_input_tokens=0, n_output_tokens=0, n_cache_tokens=0, metadata=None
    )
    await agent.run("POLICY", env, context)

    # base64 payloads + threaded session id on tool calls
    assert any("mcp-session-id: sess-1" in c for c in env.commands if "curl" in c)
    assert any("base64 -d" in c for c in env.commands if "curl" in c)
    # two model turns => usage accumulated twice
    assert context.n_input_tokens == 20
    assert context.n_output_tokens == 10
    assert context.n_cache_tokens == 6
    assert context.metadata["turns"] == 2


@pytest.mark.asyncio
async def test_agent_prefers_dedicated_gateway_creds(tmp_path, monkeypatch):
    # When the eval reroutes OPENAI_* to the task's own LLM services, the agent's
    # metered gateway arrives on VERO_AGENT_INFERENCE_* — the agent must use those.
    agent = _agent(tmp_path, monkeypatch, mcp_servers=[_server()])
    monkeypatch.setenv("VERO_AGENT_INFERENCE_API_KEY", "gw-key")
    monkeypatch.setenv("VERO_AGENT_INFERENCE_BASE_URL", "http://gw/scopes/evaluation/e/v1")

    captured = {}
    script = [_response(calls=[_call("end_conversation", {"message": "bye"})], rid="r1")]

    class _FakeResponses:
        def __init__(self):
            self._i = 0

        async def create(self, **kwargs):
            item = script[self._i]
            self._i += 1
            return item

    class _FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=_FakeResponses())

    def _fake(*a, **k):
        captured.update(k)
        return _FakeClient()

    monkeypatch.setattr(agent_module, "AsyncOpenAI", _fake)
    context = SimpleNamespace(
        n_input_tokens=0, n_output_tokens=0, n_cache_tokens=0, metadata=None
    )
    await agent.run("POLICY", _RecordingEnv(), context)

    assert captured.get("api_key") == "gw-key"
    assert captured.get("base_url") == "http://gw/scopes/evaluation/e/v1"
