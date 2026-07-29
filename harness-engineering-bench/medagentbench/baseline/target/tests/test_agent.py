import json
from types import SimpleNamespace

import pytest

from medagentbench_agent import MedAgentBenchAgent


class FakeEnvironment:
    """Records the commands the agent shells into the task container."""

    def __init__(self, return_code=0, stdout="ok"):
        self.commands: list[str] = []
        self._return_code = return_code
        self._stdout = stdout

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        return SimpleNamespace(
            return_code=self._return_code, stdout=self._stdout, stderr=""
        )


def _usage(prompt=120, completion=8, cached=20):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def _message(content=None, tool=None, arguments=None):
    calls = None
    if tool is not None:
        calls = [
            SimpleNamespace(
                id="call-1",
                type="function",
                function=SimpleNamespace(name=tool, arguments=arguments),
            )
        ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))
        ],
        usage=_usage(),
    )


class ScriptedCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return self._responses.pop(0)


def _agent(tmp_path, responses):
    agent = MedAgentBenchAgent(
        logs_dir=tmp_path / "logs", model_name="fireworks_ai/deepseek-v4-flash"
    )
    agent._client = SimpleNamespace(
        chat=SimpleNamespace(completions=ScriptedCompletions(responses))
    )
    return agent


def _context():
    return SimpleNamespace(
        metadata=None, n_input_tokens=None, n_output_tokens=None, n_cache_tokens=None
    )


@pytest.mark.asyncio
async def test_finish_routes_through_the_cli(tmp_path, monkeypatch):
    """The grader reads the file only the CLI writes, so finish must shell it."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = _agent(tmp_path, [_message(tool="finish", arguments='{"result":[-1]}')])
    environment = FakeEnvironment()
    context = _context()

    await agent.run("What is the last HbA1C?", environment, context)

    assert len(environment.commands) == 1
    command = environment.commands[0]
    assert command.startswith("/usr/local/bin/medagentbench_cli.py finish ")
    assert "[-1]" in command
    assert context.metadata["turns"] == 1
    assert context.n_input_tokens == 120


@pytest.mark.asyncio
async def test_post_is_recorded_through_the_cli(tmp_path, monkeypatch):
    """A POST must go through the CLI or the verifier never sees the action."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    payload = {"resourceType": "ServiceRequest", "code": {"text": "A1C"}}
    agent = _agent(
        tmp_path,
        [
            _message(
                tool="fhir_post",
                arguments=json.dumps(
                    {"url": "http://localhost:8080/fhir/ServiceRequest", "payload": payload}
                ),
            ),
            _message(tool="finish", arguments='{"result":["ordered"]}'),
        ],
    )
    environment = FakeEnvironment()

    await agent.run("Order an HbA1C.", environment, _context())

    post = environment.commands[0]
    assert post.startswith("/usr/local/bin/medagentbench_cli.py post ")
    assert "ServiceRequest" in post
    # The payload is shell-quoted as one argument, so braces survive intact.
    assert '{"resourceType"' in post


@pytest.mark.asyncio
async def test_non_list_finish_is_fed_back_not_raised(tmp_path, monkeypatch):
    """The CLI rejects a non-list result; the agent must recover, not die."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = _agent(
        tmp_path,
        [
            _message(tool="finish", arguments='{"result":"42"}'),
            _message(tool="finish", arguments='{"result":[42]}'),
        ],
    )
    environment = FakeEnvironment()
    context = _context()

    await agent.run("How many?", environment, context)

    assert context.metadata["turns"] == 2
    assert environment.commands[-1].endswith("'[42]'")


@pytest.mark.asyncio
async def test_empty_turn_continues_instead_of_raising(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = _agent(
        tmp_path,
        [
            _message(content=None, tool=None),
            _message(tool="finish", arguments='{"result":[1]}'),
        ],
    )
    context = _context()

    await agent.run("How many?", FakeEnvironment(), context)

    assert context.metadata["turns"] == 2


@pytest.mark.asyncio
async def test_token_accounting_survives_a_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class Exploding:
        calls = 0

        async def create(self, **kwargs):
            Exploding.calls += 1
            if Exploding.calls == 1:
                return _message(
                    tool="fhir_get",
                    arguments='{"url":"http://localhost:8080/fhir/Patient"}',
                )
            raise RuntimeError("upstream is down")

    agent = MedAgentBenchAgent(
        logs_dir=tmp_path / "logs", model_name="fireworks_ai/deepseek-v4-flash"
    )
    agent._client = SimpleNamespace(chat=SimpleNamespace(completions=Exploding()))
    context = _context()

    with pytest.raises(RuntimeError, match="upstream is down"):
        await agent.run("Who?", FakeEnvironment(), context)

    assert context.n_input_tokens == 120
    assert context.n_cache_tokens == 20


def test_requires_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="requires a Harbor model"):
        MedAgentBenchAgent(logs_dir=tmp_path)
