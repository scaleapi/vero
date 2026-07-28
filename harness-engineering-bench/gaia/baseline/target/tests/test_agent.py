from pathlib import Path
from types import SimpleNamespace

import pytest

from gaia_agent import GaiaAgent


class FakeEnvironment:
    def __init__(self):
        self.uploads: list[tuple[Path, str]] = []

    async def exec(self, command, **kwargs):
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source_path, target_path):
        self.uploads.append((Path(source_path), target_path))


class FakeResponses:
    async def create(self, **kwargs):
        return SimpleNamespace(
            id="response-1",
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="submit_answer",
                    arguments='{"answer":"42"}',
                    call_id="call-1",
                )
            ],
            output_text="",
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=8,
                input_tokens_details=SimpleNamespace(cached_tokens=20),
            ),
        )


@pytest.mark.asyncio
async def test_agent_submits_answer_and_populates_context(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = GaiaAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-5.4-mini-2026-03-17",
    )
    agent._client = SimpleNamespace(responses=FakeResponses())
    environment = FakeEnvironment()
    context = SimpleNamespace(
        metadata=None,
        n_input_tokens=None,
        n_output_tokens=None,
        n_cache_tokens=None,
    )

    await agent.run("What is six times seven?", environment, context)

    assert len(environment.uploads) == 1
    answer_path, remote_path = environment.uploads[0]
    assert answer_path.read_text(encoding="utf-8") == "42\n"
    assert remote_path == "/app/answer.txt"
    assert context.n_input_tokens == 120
    assert context.n_output_tokens == 8
    assert context.n_cache_tokens == 20
    assert context.metadata == {"turns": 1, "trace": "gaia-trace.jsonl"}


def test_agent_requires_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="requires a Harbor model"):
        GaiaAgent(logs_dir=tmp_path)
