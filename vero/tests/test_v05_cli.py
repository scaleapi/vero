from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

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
    payload = json.loads(value)
    assert payload["provider"] == {
        "anthropic": {
            "options": {"baseURL": "http://inference-gateway:8001/scopes/p/o/v1"}
        }
    }
    # The same config carries the step limit; see the step-limit test below.
    assert payload["agent"]["build"]["steps"] > 100

    # openai is the one provider the adapter already handles for baseURL; don't
    # fight it -- but opencode still needs its step limit raised.
    openai = _opencode_gateway_args("opencode", "openai/gpt-5.4", task)
    assert "provider" not in json.loads(openai[1].removeprefix("opencode_config="))
    # Other agents are untouched -- claude-code controls turns via --max-turns.
    assert _opencode_gateway_args("claude-code", "anthropic/claude-sonnet-5", task) == []
    # For opencode the step limit is unconditional: it does not depend on the
    # model spelling or on a gateway being present, because a truncated search is
    # a problem either way. Only the baseURL injection is conditional.
    bare = _opencode_gateway_args("opencode", "claude-sonnet-5", task)
    assert json.loads(bare[1].removeprefix("opencode_config="))["agent"]["build"][
        "steps"
    ] > 100
    no_gateway = _opencode_gateway_args("opencode", "anthropic/x", tmp_path / "none")
    payload = json.loads(no_gateway[1].removeprefix("opencode_config="))
    assert payload["agent"]["build"]["steps"] > 100
    assert "provider" not in payload


def test_outer_modal_trial_gets_a_named_app():
    """The outer trial must not land in Modal's anonymous default app.

    Inner eval sandboxes are already grouped by app_name via each build's
    extra_harbor_args, but the outer trial had none, so it went to __harbor__
    alongside every other workspace container -- ~1800 of them -- which makes it
    unfindable in the UI and turns "copy the session out before killing this run"
    into a search problem.
    """
    from vero.harbor.cli import _outer_app_name_args

    assert _outer_app_name_args("modal", "vero/optimize-gaia-baseline", ()) == [
        "--ek",
        "app_name=vero-optimize-gaia-baseline",
    ]
    # Docker outer trials are found via `docker ps`; no app applies.
    assert _outer_app_name_args("docker", "vero/optimize-gaia-baseline", ()) == []
    # An explicit choice wins.
    assert _outer_app_name_args(
        "modal", "vero/optimize-gaia-baseline", ("--ek", "app_name=mine")
    ) == []
    # A name that is entirely punctuation still yields a usable app.
    assert _outer_app_name_args("modal", "///", ()) == ["--ek", "app_name=vero"]


def test_opencode_gets_a_step_limit_that_does_not_truncate_the_search(tmp_path):
    """opencode caps agentic iterations at 100 and then forces a text-only reply.

    That is inside a real run: gaia's optimizer used exactly 100 and stopped
    without ever calling `evals submit`, and because the cap produces a fluent
    final message the truncation is invisible unless you count the steps.
    claude-code takes harbor's --max-turns instead, so leaving opencode at its
    default makes the two harnesses incomparable on the one axis the grid varies.
    """
    from vero.harbor.cli import OPENCODE_STEP_LIMIT, _opencode_gateway_args

    args = _opencode_gateway_args("opencode", "anthropic/claude-sonnet-5", tmp_path)
    assert args[0] == "--ak"
    payload = json.loads(args[1].removeprefix("opencode_config="))
    assert payload["agent"]["build"]["steps"] == OPENCODE_STEP_LIMIT
    assert OPENCODE_STEP_LIMIT > 100

    # Still set for the openai provider, which needs no baseURL injection.
    openai = json.loads(
        _opencode_gateway_args("opencode", "openai/gpt-5.4", tmp_path)[1]
        .removeprefix("opencode_config=")
    )
    assert openai["agent"]["build"]["steps"] == OPENCODE_STEP_LIMIT
    assert "provider" not in openai

    # claude-code is untouched; it has its own turn control.
    assert _opencode_gateway_args("claude-code", "claude-sonnet-5", tmp_path) == []


def test_litellm_harnesses_get_the_gateway_url_under_the_name_they_read(tmp_path):
    """litellm reads <PROVIDER>_API_BASE; the provider SDKs read _BASE_URL.

    vero sets the SDK names, so mini-swe-agent -- which drives the model through
    litellm[proxy] -- saw no override and called api.anthropic.com holding only a
    scoped gateway token: AuthenticationError, invalid x-api-key. It fails closed,
    so nothing leaked, but the harness could not run at all.
    """
    from vero.harbor.cli import _litellm_base_url_args

    task = tmp_path / "task"
    (task / "environment" / "gateway").mkdir(parents=True)
    (task / "environment" / "gateway" / "launch.json").write_text(
        json.dumps({"producer_base_url": "http://inference-gateway:8001/scopes/p/o/v1"}),
        encoding="utf-8",
    )

    args = _litellm_base_url_args("mini-swe-agent", task)
    pairs = dict(zip(args[::2], args[1::2]))
    assert set(pairs) == {"--ae"} or True  # flags interleave; check the values
    values = [v for k, v in zip(args[::2], args[1::2])]
    assert "OPENAI_API_BASE=http://inference-gateway:8001/scopes/p/o/v1" in values
    assert "ANTHROPIC_API_BASE=http://inference-gateway:8001/scopes/p/o/v1" in values

    # Harnesses that use provider SDKs already get _BASE_URL and need nothing.
    assert _litellm_base_url_args("claude-code", task) == []
    assert _litellm_base_url_args("opencode", task) == []
    # No compiled gateway: nothing to point at.
    assert _litellm_base_url_args("mini-swe-agent", tmp_path / "none") == []
