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


class FakeClient:
    """Stands in for `AsyncOpenAI`, including the per-call options wrapper.

    `_complete` sets its timeout (and sometimes `max_retries`) through
    `with_options`, so a fake without it fails on attribute access before any
    request is made.
    """

    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)

    def with_options(self, **_kwargs):
        return self


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
    agent._client = FakeClient(FakeCompletions())
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


TRUNCATED_REASONING = "So the candidate could be either one, unless " * 900


class TruncatedThenAnswer:
    """First turn is cut off mid-thought; the second gives the real answer."""

    def __init__(self):
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        # Snapshot: the agent keeps mutating the same `messages` list, so a
        # stored reference would only ever show the final state.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        first = len(self.requests) == 1
        message = SimpleNamespace(
            content=TRUNCATED_REASONING if first else None,
            tool_calls=None
            if first
            else [
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
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message,
                    finish_reason="length" if first else "tool_calls",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=1,
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )


@pytest.mark.asyncio
async def test_truncated_reasoning_is_never_submitted_as_the_answer(
    tmp_path, monkeypatch
):
    """A `length` finish means truncated thinking, not an answer.

    Regression test for the scoring bug: the agent used to submit `content`
    verbatim whenever a turn produced no tool calls, so 42k-52k characters of
    mid-sentence reasoning went to the judge and scored 0, indistinguishable
    from a genuinely wrong answer.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = BrowseCompPlusAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-5.4-mini-2026-03-17",
    )
    completions = TruncatedThenAnswer()
    agent._client = FakeClient(completions)
    environment = FakeEnvironment()
    context = SimpleNamespace(
        metadata=None,
        n_input_tokens=None,
        n_output_tokens=None,
        n_cache_tokens=None,
    )

    await agent.run("Find the answer.", environment, context)

    assert len(environment.uploads) == 1
    submitted = environment.uploads[0][0].read_text(encoding="utf-8")
    assert "Exact Answer: 42" in submitted
    assert "unless" not in submitted, "truncated reasoning must not be submitted"

    # The truncated turn is recovered, not fatal: the model is asked for just
    # the labeled lines, and the unfinished thinking is what it replaces.
    assert len(completions.requests) == 2
    nudge = completions.requests[1]["messages"][-1]
    assert nudge["role"] == "user"
    assert "cut off mid-thought" in nudge["content"]


class AlwaysTruncated:
    """Truncated every turn, then a clean answer on the forced-final call."""

    def __init__(self):
        self.calls = 0
        self.tool_free_calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if "tools" not in kwargs:
            self.tool_free_calls += 1
            content = (
                "Explanation: best effort.\nExact Answer: 42\nConfidence: 10%"
            )
            finish = "stop"
        else:
            content = TRUNCATED_REASONING
            finish = "length"
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, tool_calls=None),
                    finish_reason=finish,
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=1,
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )


@pytest.mark.asyncio
async def test_repeated_truncation_is_reported_as_empty_retries(tmp_path, monkeypatch):
    """`forced_final_reason` must separate this from MAX_TURNS exhaustion.

    A post-hoc conservative score uses that field to pick out the trials the
    pre-fix agent would have lost, so a trial that gave up on turn 4 after three
    truncated turns cannot be labelled the same as one that used all 32 turns.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = BrowseCompPlusAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-5.4-mini-2026-03-17",
    )
    completions = AlwaysTruncated()
    agent._client = FakeClient(completions)
    environment = FakeEnvironment()
    context = SimpleNamespace(
        metadata=None,
        n_input_tokens=None,
        n_output_tokens=None,
        n_cache_tokens=None,
    )

    await agent.run("Find the answer.", environment, context)

    assert context.metadata["forced_final"] is True
    assert context.metadata["forced_final_reason"] == "empty_retries"
    # Bounded: three nudged turns, then exactly one tool-free forced final.
    assert completions.tool_free_calls == 1
    assert completions.calls == 4
    assert "Exact Answer: 42" in environment.uploads[0][0].read_text(encoding="utf-8")


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
