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


class TwoTurnResponses:
    """Runs one shell command, then submits. Records every request it received."""

    def __init__(self):
        self.calls = 0
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        name = "run_shell" if self.calls == 1 else "submit"
        arguments = '{"command": "git status --short"}' if self.calls == 1 else "{}"
        return SimpleNamespace(
            id=f"response-{self.calls}",
            output=[
                SimpleNamespace(
                    type="function_call",
                    name=name,
                    arguments=arguments,
                    call_id=f"call-{self.calls}",
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
async def test_conversation_is_resent_and_never_relies_on_the_provider(
    tmp_path, monkeypatch
):
    """The task and prior turns must be in the request, not on the server.

    Regression test for the bug that scored 0.0000 on all 66 sampled cases:
    the agent sent only the newest tool result plus ``previous_response_id``,
    so a gateway without a response store dropped the task entirely.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = SweBenchProAgent(logs_dir=tmp_path / "logs", model_name="openai/gpt-4o")
    responses = TwoTurnResponses()
    agent._client = SimpleNamespace(responses=responses)
    context = SimpleNamespace(
        metadata=None,
        n_input_tokens=None,
        n_output_tokens=None,
        n_cache_tokens=None,
    )

    task = "Preserve this exact objective after every tool call."
    await agent.run(task, FakeEnvironment(), context)

    assert len(responses.requests) == 2
    # Never delegate memory to the provider.
    assert all("previous_response_id" not in r for r in responses.requests)
    # The task survives into the second turn, carried by us.
    second = responses.requests[1]["input"]
    assert second[0] == {"role": "user", "content": task}
    # ... and so does the first turn's call together with its output.
    kinds = [item.get("type") for item in second[1:]]
    assert "function_call" in kinds and "function_call_output" in kinds
    call = next(i for i in second[1:] if i.get("type") == "function_call")
    output = next(i for i in second[1:] if i.get("type") == "function_call_output")
    assert call["call_id"] == output["call_id"], "orphaned output is a 400"
    assert "git status --short" in call["arguments"]


def test_trim_history_drops_whole_turns_and_keeps_pairs_matched(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = SweBenchProAgent(logs_dir=tmp_path / "logs", model_name="openai/gpt-4o")
    blocks = [
        [
            {"type": "function_call", "call_id": f"c{i}", "name": "run_shell",
             "arguments": "{}"},
            {"type": "function_call_output", "call_id": f"c{i}", "output": "x" * 80_000},
        ]
        for i in range(10)
    ]
    dropped = agent._trim_history(blocks)

    assert dropped > 0, "an oversized transcript must be trimmed"
    assert agent._history_size(blocks) <= 300_000
    for block in blocks:
        ids = [i["call_id"] for i in block]
        assert len(set(ids)) == 1, "a turn must keep its call and output together"


def test_agent_requires_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="requires a Harbor model"):
        SweBenchProAgent(logs_dir=tmp_path)
