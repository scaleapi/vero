from types import SimpleNamespace

import pytest

from atlas_agent import AtlasAgent


class FakeEnvironment:
    async def exec(self, command, **kwargs):
        return SimpleNamespace(return_code=0, stdout="evidence", stderr="")


class FakeResponses:
    async def create(self, **kwargs):
        return SimpleNamespace(
            id="response-1",
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="submit_answer",
                    arguments='{"answer":"See src/example.py:10."}',
                    call_id="call-1",
                )
            ],
            output_text="",
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=12,
                input_tokens_details=SimpleNamespace(cached_tokens=25),
            ),
        )


@pytest.mark.asyncio
async def test_agent_writes_wrapped_answer_and_context(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = AtlasAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-5.4-mini-2026-03-17",
    )
    agent._client = SimpleNamespace(responses=FakeResponses())
    context = SimpleNamespace(
        metadata=None,
        n_input_tokens=None,
        n_output_tokens=None,
        n_cache_tokens=None,
    )

    await agent.run("Investigate the code", FakeEnvironment(), context)

    assert (tmp_path / "logs" / "answer.txt").read_text() == (
        "<<FINAL_ANSWER>>\nSee src/example.py:10.\n<<FINAL_ANSWER>>\n"
    )
    assert context.n_input_tokens == 100
    assert context.n_output_tokens == 12
    assert context.n_cache_tokens == 25
    assert context.metadata == {"turns": 1, "trace": "atlas-trace.jsonl"}


def test_agent_requires_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="requires a Harbor model"):
        AtlasAgent(logs_dir=tmp_path)
