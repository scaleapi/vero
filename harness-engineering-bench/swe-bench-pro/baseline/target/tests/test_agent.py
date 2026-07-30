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


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _completion(response_id, calls, *, content="", usage=(200, 12, 30)):
    prompt, completion, cached = usage
    return SimpleNamespace(
        id=response_id,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=calls or None)
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
    )


def _client(completions):
    """The agent calls ``client.chat.completions.create``."""
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class FakeCompletions:
    """Returns a single submit call, then would loop forever if asked again."""

    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return _completion(
            f"response-{self.calls}", [_tool_call("call-1", "submit", "{}")]
        )


class FlakyCompletions:
    """Fails once, then returns a submit call: exercises the retry helper."""

    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient upstream error")
        return _completion(
            "response-ok",
            [_tool_call("call-1", "submit", "{}")],
            usage=(10, 1, 0),
        )


def _context():
    return SimpleNamespace(
        metadata=None,
        n_input_tokens=None,
        n_output_tokens=None,
        n_cache_tokens=None,
    )


@pytest.mark.asyncio
async def test_agent_submits_and_populates_context(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = SweBenchProAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4o",
    )
    agent._client = _client(FakeCompletions())
    environment = FakeEnvironment()
    context = _context()

    await agent.run("Fix the failing test in the repository.", environment, context)

    # Chat Completions reports usage as prompt/completion, not input/output.
    assert context.n_input_tokens == 200
    assert context.n_output_tokens == 12
    assert context.n_cache_tokens == 30
    assert context.metadata == {
        "turns": 1,
        "trace": "swe-bench-pro-trace.jsonl",
    }


@pytest.mark.asyncio
async def test_completion_create_retries_transient_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # Avoid real backoff sleeps in the test.
    monkeypatch.setattr("swebench_pro_agent.agent.API_RETRY_BASE_DELAY", 0.0)
    monkeypatch.setattr("swebench_pro_agent.agent.API_RETRY_MAX_DELAY", 0.0)
    agent = SweBenchProAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4o",
    )
    flaky = FlakyCompletions()
    agent._client = _client(flaky)
    context = _context()

    await agent.run("Fix the failing test in the repository.", FakeEnvironment(), context)

    assert flaky.calls == 2  # one failure, then one success
    assert context.metadata == {
        "turns": 1,
        "trace": "swe-bench-pro-trace.jsonl",
    }


class TwoTurnCompletions:
    """Runs one shell command, then submits. Records every request it received."""

    def __init__(self):
        self.calls = 0
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        first = self.calls == 1
        return _completion(
            f"response-{self.calls}",
            [
                _tool_call(
                    f"call-{self.calls}",
                    "run_shell" if first else "submit",
                    '{"command": "git status --short"}' if first else "{}",
                )
            ],
            usage=(10, 1, 0),
        )


@pytest.mark.asyncio
async def test_conversation_is_resent_and_never_relies_on_the_provider(
    tmp_path, monkeypatch
):
    """The task and prior turns must be in the request, not on the server.

    Regression test for the bug that scored 0.0000 on all 66 sampled cases: the
    agent sent only the newest tool result plus ``previous_response_id``, so a
    gateway without a response store dropped the task entirely. Chat Completions
    has no such field, and this asserts we never reintroduce one.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = SweBenchProAgent(logs_dir=tmp_path / "logs", model_name="openai/gpt-4o")
    completions = TwoTurnCompletions()
    agent._client = _client(completions)

    task = "Preserve this exact objective after every tool call."
    await agent.run(task, FakeEnvironment(), _context())

    assert len(completions.requests) == 2
    # Never delegate memory to the provider, by any spelling.
    assert all("previous_response_id" not in r for r in completions.requests)
    assert all("input" not in r for r in completions.requests)

    second = completions.requests[1]["messages"]
    # The instructions and the task both survive into the second turn, carried by us.
    assert second[0]["role"] == "system"
    assert second[1] == {"role": "user", "content": task}
    # ... and so does the first turn's call together with its result.
    assistant = next(m for m in second[2:] if m["role"] == "assistant")
    tool = next(m for m in second[2:] if m["role"] == "tool")
    assert second.index(assistant) < second.index(tool), "result before its call is a 400"
    declared = {c["id"] for c in assistant["tool_calls"]}
    assert tool["tool_call_id"] in declared, "an undeclared tool_call_id is a 400"
    assert "git status --short" in assistant["tool_calls"][0]["function"]["arguments"]


@pytest.mark.asyncio
async def test_tools_are_sent_in_chat_completions_shape(tmp_path, monkeypatch):
    """Chat Completions nests the schema under "function"; a flat one is a 400."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = SweBenchProAgent(logs_dir=tmp_path / "logs", model_name="openai/gpt-4o")
    completions = TwoTurnCompletions()
    agent._client = _client(completions)

    await agent.run("Fix it.", FakeEnvironment(), _context())

    tools = completions.requests[0]["tools"]
    assert {tool["function"]["name"] for tool in tools} == {
        "run_shell",
        "read_file",
        "write_file",
        "apply_patch",
        "run_tests",
        "submit",
    }
    for tool in tools:
        assert tool["type"] == "function"
        assert set(tool) == {"type", "function"}, "the schema must be nested"
        assert "parameters" in tool["function"]


@pytest.mark.asyncio
async def test_reasoning_effort_is_only_sent_to_reasoning_models(tmp_path, monkeypatch):
    """`reasoning_effort` on gpt-4o is a hard 400 on the very first turn."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    plain = SweBenchProAgent(logs_dir=tmp_path / "plain", model_name="openai/gpt-4o")
    plain_completions = TwoTurnCompletions()
    plain._client = _client(plain_completions)
    await plain.run("Fix it.", FakeEnvironment(), _context())
    assert "reasoning_effort" not in plain_completions.requests[0]

    thinker = SweBenchProAgent(logs_dir=tmp_path / "o", model_name="openai/gpt-5.6-sol")
    thinker_completions = TwoTurnCompletions()
    thinker._client = _client(thinker_completions)
    await thinker.run("Fix it.", FakeEnvironment(), _context())
    assert thinker_completions.requests[0]["reasoning_effort"] == "high"


def test_trim_history_drops_whole_turns_and_keeps_pairs_matched(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = SweBenchProAgent(logs_dir=tmp_path / "logs", model_name="openai/gpt-4o")
    from swebench_pro_agent.agent import MAX_HISTORY_CHARS

    blocks = [
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {"name": "run_shell", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 80_000},
        ]
        for i in range(10)
    ]
    dropped = agent._trim_history(blocks)

    assert dropped > 0, "an oversized transcript must be trimmed"
    assert agent._history_size(blocks) <= MAX_HISTORY_CHARS
    for block in blocks:
        declared = {c["id"] for c in block[0]["tool_calls"]}
        results = {m["tool_call_id"] for m in block[1:]}
        assert results == declared, "a turn must keep its calls and results together"


def test_agent_requires_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="requires a Harbor model"):
        SweBenchProAgent(logs_dir=tmp_path)
