from pathlib import Path
from types import SimpleNamespace

import pytest
from browsecomp_plus_agent import BrowseCompPlusAgent


class FakeEnvironment:
    def __init__(self):
        self.uploads: list[tuple[Path, str]] = []

    async def exec(self, command, **kwargs):
        return SimpleNamespace(return_code=0, stdout="[]", stderr="")

    async def upload_file(self, source_path, target_path):
        self.uploads.append((Path(source_path), target_path))


class FakeCompletions:
    async def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                type="function",
                                function=SimpleNamespace(
                                    name="submit_response",
                                    arguments=json_response(
                                        "Explanation: Evidence [12].\n"
                                        "Exact Answer: 42\n"
                                        "Confidence: 100%"
                                    ),
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=120,
                completion_tokens=8,
                prompt_tokens_details=SimpleNamespace(cached_tokens=20),
            ),
        )


def json_response(value: str) -> str:
    import json

    return json.dumps({"response": value})


@pytest.fixture
def gateway_env(monkeypatch):
    """The metered, allow-listed inference the agent is required to use.

    These are the only credentials the agent reads: OPENAI_* is deliberately not
    a fallback, because this benchmark sets task_services_use_upstream and so
    points OPENAI_* at the raw upstream. Constructing the agent without these
    fails closed, so every test that builds one has to supply them, exactly as
    the compiled task does.
    """
    monkeypatch.setenv("VERO_AGENT_INFERENCE_API_KEY", "test-gateway-key")
    monkeypatch.setenv("VERO_AGENT_INFERENCE_BASE_URL", "http://gateway.invalid/v1")


@pytest.mark.asyncio
async def test_agent_submits_response_and_populates_context(tmp_path, gateway_env):
    agent = BrowseCompPlusAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-5.4-mini-2026-03-17",
    )
    agent._client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    environment = FakeEnvironment()
    context = SimpleNamespace(
        metadata=None,
        n_input_tokens=None,
        n_output_tokens=None,
        n_cache_tokens=None,
    )

    await agent.run("Find the answer.", environment, context)

    assert len(environment.uploads) == 1
    answer_path, remote_path = environment.uploads[0]
    assert "Exact Answer: 42" in answer_path.read_text(encoding="utf-8")
    assert remote_path == "/app/answer.txt"
    assert context.n_input_tokens == 120
    assert context.n_output_tokens == 8
    assert context.n_cache_tokens == 20
    assert context.metadata == {
        "turns": 1,
        "trace": "browsecomp-plus-trace.jsonl",
    }


@pytest.mark.asyncio
async def test_search_uses_fixed_index_cli(tmp_path, gateway_env):
    agent = BrowseCompPlusAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-5.4-mini-2026-03-17",
    )
    environment = FakeEnvironment()

    assert await agent._index_command(environment, "search", "quoted query") == []


def test_agent_requires_model(tmp_path, gateway_env):
    with pytest.raises(ValueError, match="requires a Harbor model"):
        BrowseCompPlusAgent(logs_dir=tmp_path)


@pytest.mark.parametrize(
    "present",
    [
        {},
        {"VERO_AGENT_INFERENCE_API_KEY": "k"},
        {"VERO_AGENT_INFERENCE_BASE_URL": "http://gateway.invalid/v1"},
        {"VERO_AGENT_INFERENCE_API_KEY": "", "VERO_AGENT_INFERENCE_BASE_URL": ""},
    ],
)
def test_agent_refuses_to_run_without_the_gateway(tmp_path, monkeypatch, present):
    # The regression this guards: target inference silently fell back to OPENAI_*,
    # which under task_services_use_upstream is the raw upstream credential. That
    # bought an unmetered 7.5-hour run whose only symptom was an empty evaluation
    # scope in the request log, which is why nobody noticed. A half-configured
    # gateway must fail the same way as none at all.
    for name in ("VERO_AGENT_INFERENCE_API_KEY", "VERO_AGENT_INFERENCE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in present.items():
        monkeypatch.setenv(name, value)
    # Set so a fallback would succeed if one existed, making the test meaningful.
    monkeypatch.setenv("OPENAI_API_KEY", "upstream-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://upstream.invalid/v1")

    with pytest.raises(RuntimeError, match="requires"):
        BrowseCompPlusAgent(
            logs_dir=tmp_path / "logs",
            model_name="openai/gpt-5.4-mini-2026-03-17",
        )
