from pathlib import Path
from types import SimpleNamespace

import pytest

from dabstep_agent import DabstepAgent


class FakeEnvironment:
    def __init__(self):
        self.uploads: list[tuple[Path, str]] = []

    async def exec(self, command, **kwargs):
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source_path, target_path):
        self.uploads.append((Path(source_path), target_path))


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
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))],
        usage=_usage(),
    )


class ScriptedCompletions:
    """Returns one queued response per call, so a turn sequence can be asserted."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return self._responses.pop(0)


def _agent(tmp_path, responses):
    agent = DabstepAgent(logs_dir=tmp_path / "logs", model_name="fireworks_ai/deepseek-v4-flash")
    completions = ScriptedCompletions(responses)
    agent._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return agent, completions


def _context():
    return SimpleNamespace(
        metadata=None, n_input_tokens=None, n_output_tokens=None, n_cache_tokens=None
    )


@pytest.mark.asyncio
async def test_submits_answer_and_populates_context(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent, _ = _agent(tmp_path, [_message(tool="submit_answer", arguments='{"answer":"138236"}')])
    environment = FakeEnvironment()
    context = _context()

    await agent.run("How many transactions are there?", environment, context)

    assert len(environment.uploads) == 1
    answer_path, remote_path = environment.uploads[0]
    assert answer_path.read_text(encoding="utf-8") == "138236\n"
    assert remote_path == "/app/answer.txt"
    assert context.n_input_tokens == 120
    assert context.metadata == {"turns": 1, "trace": "dabstep-trace.jsonl"}


@pytest.mark.asyncio
async def test_answer_is_reduced_to_one_line(tmp_path, monkeypatch):
    """The scorer reads the first line, so a chatty answer must not smuggle prose."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent, _ = _agent(
        tmp_path,
        [_message(tool="submit_answer", arguments='{"answer":"0.42\\nbecause of fees"}')],
    )
    environment = FakeEnvironment()

    await agent.run("What is the rate?", environment, _context())

    assert environment.uploads[0][0].read_text(encoding="utf-8") == "0.42\n"


@pytest.mark.asyncio
async def test_empty_turn_continues_instead_of_raising(tmp_path, monkeypatch):
    """A reasoning-only turn must not end the trial at the failure value."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent, completions = _agent(
        tmp_path,
        [
            _message(content=None, tool=None),
            _message(tool="submit_answer", arguments='{"answer":"7"}'),
        ],
    )
    environment = FakeEnvironment()
    context = _context()

    await agent.run("How many?", environment, context)

    assert completions.calls == 2
    assert environment.uploads[0][0].read_text(encoding="utf-8") == "7\n"
    assert context.metadata == {"turns": 2, "trace": "dabstep-trace.jsonl"}


@pytest.mark.asyncio
async def test_token_accounting_survives_a_raise(tmp_path, monkeypatch):
    """Usage is reported even when the trial dies, so cost is not biased low."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class Exploding:
        calls = 0

        async def create(self, **kwargs):
            Exploding.calls += 1
            if Exploding.calls == 1:
                return _message(tool="run_shell", arguments='{"command":"ls"}')
            raise RuntimeError("upstream is down")

    agent = DabstepAgent(logs_dir=tmp_path / "logs", model_name="fireworks_ai/deepseek-v4-flash")
    agent._client = SimpleNamespace(chat=SimpleNamespace(completions=Exploding()))
    context = _context()

    with pytest.raises(RuntimeError, match="upstream is down"):
        await agent.run("How many?", FakeEnvironment(), context)

    assert context.n_input_tokens == 120
    assert context.n_output_tokens == 8
    assert context.n_cache_tokens == 20


def test_requires_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="requires a Harbor model"):
        DabstepAgent(logs_dir=tmp_path)


def test_no_image_tool_is_declared(monkeypatch):
    """The pinned target is text-only; an image tool would 400 and score zero."""
    from dabstep_agent import agent as module

    assert {tool["function"]["name"] for tool in module.TOOLS} == {
        "run_shell",
        "submit_answer",
    }
