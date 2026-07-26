from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

import vero
from vero.candidate import Candidate
from vero.cli import main


def test_root_package_exposes_only_canonical_program_api():
    assert vero.Candidate is Candidate
    assert not hasattr(vero, "Experiment")


def test_canonical_root_and_tools_do_not_import_legacy_core():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, vero, vero.tools; "
                "assert not any(name == 'vero.core' or name.startswith('vero.core.') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def initialize_repository(path: Path) -> None:
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "--all"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=vero",
            "-c",
            "user.email=vero@localhost",
            "commit",
            "-m",
            "baseline",
        ],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_cli_optimizes_non_python_program_and_inspects_session(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "program.txt").write_text("slow\n", encoding="utf-8")
    initialize_repository(target)

    harness = tmp_path / "harness"
    harness.mkdir()
    harness_script = harness / "evaluate.py"
    harness_script.write_text(
        """
import json
import sys
from pathlib import Path
workspace, report = map(Path, sys.argv[1:])
latency = 1.0 if (workspace / "program.txt").read_text().strip() == "fast" else 9.0
report.write_text(json.dumps({
    "schema_version": 1,
    "status": "success",
    "metrics": {"latency_ms": latency, "correct": 1.0},
}))
""",
        encoding="utf-8",
    )
    producer = tmp_path / "producer"
    producer.mkdir()
    producer_script = producer / "improve.py"
    producer_script.write_text(
        """
import sys
from pathlib import Path
Path(sys.argv[1], "program.txt").write_text("fast\\n")
""",
        encoding="utf-8",
    )
    session_dir = tmp_path / "sessions" / "nested" / "cli-run"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "optimize",
            str(target),
            "--harness-root",
            str(harness),
            "--evaluate",
            shlex.join(
                [
                    sys.executable,
                    str(harness_script),
                    "{workspace}",
                    "{report}",
                ]
            ),
            "--producer-root",
            str(producer),
            "--produce",
            shlex.join([sys.executable, str(producer_script), "{workspace}"]),
            "--metric",
            "latency_ms",
            "--direction",
            "minimize",
            "--constraint",
            "correct",
            "==",
            "1",
            "--evaluation-set",
            "performance",
            "--parameter",
            "threshold=0.5",
            "--session-dir",
            str(session_dir),
            "--session-id",
            "cli-session",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Baseline:" in result.output
    assert "(9.0)" in result.output
    assert "Best:" in result.output
    assert "(1.0)" in result.output
    assert (target / "program.txt").read_text(encoding="utf-8") == "slow\n"

    inspect_result = runner.invoke(main, ["session", "inspect", str(session_dir)])
    assert inspect_result.exit_code == 0, inspect_result.output
    inspection = json.loads(inspect_result.output)
    assert inspection["manifest"]["id"] == "cli-session"
    assert inspection["manifest"]["status"] == "completed"
    assert len(inspection["candidates"]) == 2
    assert len(inspection["evaluations"]) == 2

    list_result = runner.invoke(
        main,
        ["session", "list", "--root", str(tmp_path / "sessions")],
    )
    assert list_result.exit_code == 0, list_result.output
    assert "cli-session\tcompleted" in list_result.output
    assert "nested/cli-run" in list_result.output

    archive = tmp_path / "cli-export.tar.gz"
    export_result = runner.invoke(
        main,
        ["session", "export", str(session_dir), "--output", str(archive)],
    )
    assert export_result.exit_code == 0, export_result.output
    assert archive.is_file()

    fork_dir = tmp_path / "sessions" / "forked"
    fork_result = runner.invoke(
        main,
        [
            "session",
            "fork",
            str(session_dir),
            str(fork_dir),
            "--max-proposals",
            "2",
            "--reset-budgets",
        ],
    )
    assert fork_result.exit_code == 0, fork_result.output
    forked = json.loads((fork_dir / "manifest.json").read_text())
    assert forked["id"] == "forked"
    assert forked["run"]["max_proposals"] == 2
    assert json.loads((fork_dir / "database.json").read_text())["id"] == "forked"

    clear_result = runner.invoke(
        main,
        ["session", "clear", str(fork_dir), "--yes"],
    )
    assert clear_result.exit_code == 0, clear_result.output
    assert not fork_dir.exists()

    evaluate_only_dir = tmp_path / "sessions" / "evaluate-only"
    evaluate_only = runner.invoke(
        main,
        [
            "optimize",
            str(target),
            "--harness-root",
            str(harness),
            "--evaluate",
            shlex.join(
                [
                    sys.executable,
                    str(harness_script),
                    "{workspace}",
                    "{report}",
                ]
            ),
            "--metric",
            "latency_ms",
            "--direction",
            "minimize",
            "--max-proposals",
            "0",
            "--session-dir",
            str(evaluate_only_dir),
        ],
    )
    assert evaluate_only.exit_code == 0, evaluate_only.output
    assert "(9.0)" in evaluate_only.output


def test_cli_init_and_check_config(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(main, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output

    target = tmp_path / "target"
    target.mkdir()
    (target / "program.txt").write_text("baseline\n", encoding="utf-8")
    initialize_repository(target)
    (tmp_path / "harness").mkdir()

    checked = runner.invoke(
        main,
        ["check", "--config", str(tmp_path / "vero.toml")],
    )
    assert checked.exit_code == 0, checked.output
    assert "selection='validation'" in checked.output


def test_cli_requires_exactly_one_producer(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    harness = tmp_path / "harness"
    harness.mkdir()
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "optimize",
            str(target),
            "--harness-root",
            str(harness),
            "--evaluate",
            "evaluate {workspace} {report}",
            "--metric",
            "score",
            "--direction",
            "maximize",
        ],
    )

    assert result.exit_code == 2
    assert "exactly one of --produce or --agent" in result.output


def test_cli_rejects_options_that_do_not_apply(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    harness = tmp_path / "harness"
    harness.mkdir()
    base = [
        "optimize",
        str(target),
        "--harness-root",
        str(harness),
        "--evaluate",
        "evaluate {workspace} {report}",
        "--metric",
        "score",
        "--direction",
        "maximize",
        "--max-proposals",
        "0",
    ]
    runner = CliRunner()

    max_turns = runner.invoke(main, [*base, "--max-turns", "10"])
    producer_timeout = runner.invoke(main, [*base, "--producer-timeout", "10"])
    wandb = runner.invoke(main, [*base, "--wandb-mode", "offline"])

    assert max_turns.exit_code == 2
    assert "--max-turns is only valid with --agent" in max_turns.output
    assert producer_timeout.exit_code == 2
    assert "are only valid with --produce" in producer_timeout.output
    assert wandb.exit_code == 2
    assert "require --wandb-project" in wandb.output


def test_opencode_non_openai_provider_gets_a_gateway_base_url(tmp_path):
    """opencode reaches non-openai providers only if we supply the baseURL.

    Harbor's adapter injects one for `openai/...` alone, so `anthropic/...`
    otherwise calls api.anthropic.com and dies on 401 (the optimizer holds only a
    scoped token). Forcing `openai/` instead puts a Claude model on the Responses
    API, whose litellm translation opencode cannot parse. This keeps Anthropic on
    Messages, through the gateway.
    """
    from vero.harbor.cli import _opencode_gateway_args

    task = tmp_path / "task"
    (task / "environment" / "gateway").mkdir(parents=True)
    (task / "environment" / "gateway" / "launch.json").write_text(
        json.dumps({"producer_base_url": "http://inference-gateway:8001/scopes/p/o/v1"}),
        encoding="utf-8",
    )

    args = _opencode_gateway_args("opencode", "anthropic/claude-sonnet-5", task)
    assert args[0] == "--ak"
    key, _, value = args[1].partition("=")
    assert key == "opencode_config"
    assert json.loads(value) == {
        "provider": {
            "anthropic": {
                "options": {"baseURL": "http://inference-gateway:8001/scopes/p/o/v1"}
            }
        }
    }

    # openai is the one provider the adapter already handles; don't fight it.
    assert _opencode_gateway_args("opencode", "openai/gpt-5.4", task) == []
    # Other agents and bare model names are untouched.
    assert _opencode_gateway_args("claude-code", "anthropic/claude-sonnet-5", task) == []
    assert _opencode_gateway_args("opencode", "claude-sonnet-5", task) == []
    # A task compiled without a gateway simply gets no override.
    assert _opencode_gateway_args("opencode", "anthropic/x", tmp_path / "none") == []


class _Gateway:
    """Minimal stand-in for InferenceGatewaySpec's preflight-relevant surface."""

    upstream_api_key_env = "OPENAI_API_KEY"
    upstream_base_url_env = "OPENAI_BASE_URL"
    default_upstream_base_url = "https://api.openai.com/v1"
    finalization = None

    def __init__(self, producer, evaluation):
        self.producer = type("S", (), {"allowed_models": producer})()
        self.evaluation = type("S", (), {"allowed_models": evaluation})()


class _Build:
    def __init__(self, gateway):
        self.inference_gateway = gateway


def test_preflight_blocks_only_on_a_definitively_missing_deployment(monkeypatch):
    """A missing deployment is otherwise invisible until a whole trial has burned.

    The upstream 404s, the agent makes no progress, and every case is scored 0.0
    as an honest-looking task failure.
    """
    import click

    from vero.harbor import cli as harbor_cli

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/openai/v1")
    probed: list[str] = []

    def _probe(base_url, api_key, model):
        probed.append(model)
        if model == "dead-model":
            return 404, '{"error": {"code": "DeploymentNotFound"}}'
        return 200, ""

    monkeypatch.setattr(harbor_cli, "_probe_model", _probe)

    with pytest.raises(click.ClickException) as raised:
        harbor_cli._preflight_models(
            _Build(_Gateway(["gpt-5.3-codex"], ["dead-model"]))
        )
    assert "dead-model (evaluation scope)" in str(raised.value)
    assert "DeploymentNotFound" in str(raised.value)
    # A provider prefix routes on a proxy and is meaningless to a single
    # provider endpoint, so the configured spelling is tried first.
    probed.clear()
    harbor_cli._preflight_models(_Build(_Gateway(["openai/gpt-4o"], ["gpt-4o"])))
    assert probed == ["openai/gpt-4o", "gpt-4o"]


def test_preflight_falls_back_to_the_bare_name_before_calling_a_model_missing(
    monkeypatch,
):
    """One spelling 404ing is not evidence the model is absent.

    Azure serves `gpt-5.3-codex` and 404s `openai/gpt-5.3-codex`; a routing
    proxy does the reverse. Blocking on the first spelling would refuse a run
    that would have worked.
    """
    from vero.harbor import cli as harbor_cli

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/openai/v1")
    probed: list[str] = []

    def _probe(base_url, api_key, model):
        probed.append(model)
        if "/" in model:
            return 404, '{"error": {"code": "DeploymentNotFound"}}'
        return 200, ""

    monkeypatch.setattr(harbor_cli, "_probe_model", _probe)
    harbor_cli._preflight_models(_Build(_Gateway(["openai/gpt-5.3-codex"], [])))
    assert probed == ["openai/gpt-5.3-codex", "gpt-5.3-codex"]


def test_probe_model_separates_a_missing_route_from_a_missing_model(monkeypatch):
    """An upstream that serves only Chat Completions must not read as missing.

    Both failures are HTTP 404 and they mean opposite things, so the body is
    the only discriminator. gaia's agent uses the Responses API and the rest
    use Chat Completions, so either route may legitimately be absent.
    """
    from vero.harbor import cli as harbor_cli

    seen: list[str] = []

    def _route(base_url, api_key, model, route, input_key):
        seen.append(route)
        if route == "/responses":
            return 404, '{"detail": "Not Found"}'  # the route, not the model
        return 200, ""

    monkeypatch.setattr(harbor_cli, "_probe_route", _route)
    assert harbor_cli._probe_model("https://x/v1", "k", "m") == (200, "")
    assert seen == ["/responses", "/chat/completions"]

    # A model-level 404 on the first route is conclusive: do not keep probing.
    seen.clear()
    monkeypatch.setattr(
        harbor_cli,
        "_probe_route",
        lambda b, k, m, route, i: (
            seen.append(route),
            (404, '{"error": {"code": "DeploymentNotFound"}}'),
        )[1],
    )
    status, _ = harbor_cli._probe_model("https://x/v1", "k", "m")
    assert status == 404
    assert seen == ["/responses"]

    # Neither route served: inconclusive, so the run is allowed to proceed.
    seen.clear()
    monkeypatch.setattr(
        harbor_cli,
        "_probe_route",
        lambda b, k, m, route, i: (seen.append(route), (404, "<html>404</html>"))[1],
    )
    status, body = harbor_cli._probe_model("https://x/v1", "k", "m")
    assert status is None
    assert seen == ["/responses", "/chat/completions"]
    assert "not served by this upstream" in body


def test_preflight_does_not_block_on_inconclusive_upstream_failures(monkeypatch):
    from vero.harbor import cli as harbor_cli

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/openai/v1")

    for status, body in ((429, "rate limited"), (503, "unavailable"), (None, "timeout")):
        monkeypatch.setattr(
            harbor_cli, "_probe_model", lambda b, k, m, s=status, y=body: (s, y)
        )
        # Must not raise: a transient upstream blip cannot be allowed to stop a
        # run that would have succeeded.
        harbor_cli._preflight_models(_Build(_Gateway(["a"], ["b"])))

    # No credentials and no gateway are both no-ops rather than errors.
    monkeypatch.delenv("OPENAI_API_KEY")
    harbor_cli._preflight_models(_Build(_Gateway(["a"], ["b"])))
    harbor_cli._preflight_models(_Build(None))
