from types import SimpleNamespace

import pytest

from swebench_pro_agent import SweBenchProAgent


class FakeEnvironment:
    def __init__(self):
        self.commands: list[str] = []

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source_path, target_path):  # pragma: no cover
        pass


class FakeResponses:
    """Returns a single submit call, then would loop forever if asked again."""

    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            id=f"response-{self.calls}",
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="submit",
                    arguments="{}",
                    call_id="call-1",
                )
            ],
            output_text="",
            usage=SimpleNamespace(
                input_tokens=200,
                output_tokens=12,
                input_tokens_details=SimpleNamespace(cached_tokens=30),
            ),
        )


class FlakyResponses:
    """Fails once, then returns a submit call: exercises the retry helper."""

    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient upstream error")
        return SimpleNamespace(
            id="response-ok",
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="submit",
                    arguments="{}",
                    call_id="call-1",
                )
            ],
            output_text="",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=1,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )


@pytest.mark.asyncio
async def test_agent_submits_and_populates_context(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = SweBenchProAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4o",
    )
    agent._client = SimpleNamespace(responses=FakeResponses())
    environment = FakeEnvironment()
    context = SimpleNamespace(
        metadata=None,
        n_input_tokens=None,
        n_output_tokens=None,
        n_cache_tokens=None,
    )

    await agent.run("Fix the failing test in the repository.", environment, context)

    assert context.n_input_tokens == 200
    assert context.n_output_tokens == 12
    assert context.n_cache_tokens == 30
    assert context.metadata == {
        "turns": 1,
        "trace": "swe-bench-pro-trace.jsonl",
    }


@pytest.mark.asyncio
async def test_responses_create_retries_transient_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # Avoid real backoff sleeps in the test.
    monkeypatch.setattr("swebench_pro_agent.agent.API_RETRY_BASE_DELAY", 0.0)
    monkeypatch.setattr("swebench_pro_agent.agent.API_RETRY_MAX_DELAY", 0.0)
    agent = SweBenchProAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4o",
    )
    flaky = FlakyResponses()
    agent._client = SimpleNamespace(responses=flaky)
    environment = FakeEnvironment()
    context = SimpleNamespace(
        metadata=None,
        n_input_tokens=None,
        n_output_tokens=None,
        n_cache_tokens=None,
    )

    await agent.run("Fix the failing test in the repository.", environment, context)

    assert flaky.calls == 2  # one failure, then one success
    assert context.metadata == {
        "turns": 1,
        "trace": "swe-bench-pro-trace.jsonl",
    }


def test_agent_requires_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="requires a Harbor model"):
        SweBenchProAgent(logs_dir=tmp_path)
