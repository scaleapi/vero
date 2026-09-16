"""Behaviour the wrapper must keep: model routing and the build contract.

Nothing here builds opencode or talks to a model; those paths are exercised by
the benchmark's smoke run.
"""

from __future__ import annotations

import pytest
from harbor.agents.installed.opencode import OpenCode

from terminal_bench_agent import _build
from terminal_bench_agent.agent import WIRE_PROVIDER_ID, TerminalBenchAgent, _bare_model


def test_agent_is_the_stock_opencode_runner_with_a_vendored_binary(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.test/v1")
    agent = TerminalBenchAgent(logs_dir=tmp_path / "logs", model_name="xai/grok-build-0.1")
    assert isinstance(agent, OpenCode)
    # The gateway allow-lists the bare name; opencode sees it under a provider id it
    # has no builtin (Responses-API-forcing) loader for.
    assert agent.model_name == f"{WIRE_PROVIDER_ID}/grok-build-0.1"
    provider = agent._opencode_config["provider"][WIRE_PROVIDER_ID]
    assert "grok-build-0.1" in provider["models"]
    # The proxy's model_group for this model is registered under the
    # vendor-prefixed form, not the bare name.
    assert provider["models"]["grok-build-0.1"]["id"] == "xai/grok-build-0.1"
    assert provider["options"]["baseURL"] == "https://gateway.test/v1"
    assert provider["options"]["apiKey"] == "test-key"


def test_agent_requires_model(tmp_path):
    with pytest.raises(ValueError, match="requires a Harbor model"):
        TerminalBenchAgent(logs_dir=tmp_path)


def test_bare_model_strips_any_provider_prefix():
    assert _bare_model("xai/grok-build-0.1") == "grok-build-0.1"
    assert _bare_model("openai/grok-build-0.1") == "grok-build-0.1"
    assert _bare_model("grok-build-0.1") == "grok-build-0.1"


def test_tree_hash_ignores_build_outputs(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    before = _build.tree_hash(tmp_path)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("1")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "opencode").write_text("bin")
    assert _build.tree_hash(tmp_path) == before
    (tmp_path / "package.json").write_text('{"changed": true}')
    assert _build.tree_hash(tmp_path) != before


def test_prebuilt_binary_short_circuits_the_build(tmp_path, monkeypatch):
    binary = tmp_path / "opencode"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv("VERO_OPENCODE_BINARY", str(binary))
    assert _build.build_binary(tmp_path) == binary


def test_missing_source_is_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("VERO_OPENCODE_BINARY", raising=False)
    monkeypatch.setenv("VERO_OPENCODE_BUILD_CACHE", str(tmp_path / "cache"))
    with pytest.raises(RuntimeError, match="no opencode source"):
        _build.build_binary(tmp_path / "empty")
