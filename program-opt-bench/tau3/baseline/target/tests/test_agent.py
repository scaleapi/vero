from types import SimpleNamespace

import pytest

from tau3_agent import Tau3Agent


def _agent(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return Tau3Agent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-5.4-mini-2026-03-17",
        **kwargs,
    )


def test_agent_resolves_one_streamable_http_server(tmp_path, monkeypatch):
    server = SimpleNamespace(
        name="tau3-runtime",
        transport="streamable-http",
        url="http://tau3-runtime:8000/mcp",
    )
    agent = _agent(tmp_path, monkeypatch, mcp_servers=[server])

    assert agent._server_url() == "http://tau3-runtime:8000/mcp"


class FakeEnvironment:
    def __init__(self):
        self.uploads = []

    async def exec(self, command, **kwargs):
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source, destination):
        self.uploads.append((source, destination))


@pytest.mark.asyncio
async def test_setup_installs_and_uploads_in_environment(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    environment = FakeEnvironment()

    await agent.setup(environment)

    assert environment.uploads[0][0].name == "runner.py"
    assert environment.uploads[0][1] == "/tmp/vero-tau3-agent/runner.py"


def test_agent_requires_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="requires a Harbor model"):
        Tau3Agent(logs_dir=tmp_path)
