"""The launch-time report naming where a run's W&B results will land.

`wandb.entity` is optional and omitting it is silent, so a cell reports into the
launching user's personal entity while looking, to anyone checking the shared
org, like a run that never started. These pin that the destination is always
stated and that the unset case says so loudly.
"""

from __future__ import annotations

from types import SimpleNamespace

import click
import pytest

from vero.harbor import cli as harbor_cli


def _config(entity=None, project="harness-engineering-bench", name="cell-1"):
    return SimpleNamespace(
        wandb=SimpleNamespace(entity=entity, project=project, name=name)
    )


def test_pinned_entity_is_reported(capsys) -> None:
    harbor_cli._report_wandb_destination(_config(entity="egp"))
    out = capsys.readouterr()
    assert "W&B: egp/harness-engineering-bench run=cell-1" in out.out
    assert "WARNING" not in out.out + out.err


def test_unset_entity_warns_and_names_the_fallback(capsys) -> None:
    harbor_cli._report_wandb_destination(_config(entity=None))
    err = capsys.readouterr().err
    assert "personal default" in err
    assert "WARNING" in err
    # The natural workaround has to be named, or the reader tries it and the run
    # still scatters: the sidecar owns wandb.init and never sees the variable.
    assert "WANDB_ENTITY will" in err and "NOT change this" in err
    assert "entity:" in err


def test_empty_string_entity_is_treated_as_unset(capsys) -> None:
    """`entity: ""` resolves from an unset ${param} and must not read as pinned."""

    harbor_cli._report_wandb_destination(_config(entity=""))
    assert "WARNING" in capsys.readouterr().err


def test_silent_when_wandb_is_not_configured(capsys) -> None:
    harbor_cli._report_wandb_destination(SimpleNamespace(wandb=None))
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_missing_run_name_does_not_crash_the_launch(capsys) -> None:
    """A diagnostic that raises would block the run it is meant to describe."""

    harbor_cli._report_wandb_destination(_config(entity="egp", name=None))
    assert "run=<default>" in capsys.readouterr().out


def test_harbor_run_warns_before_it_compiles_the_task(tmp_path, monkeypatch) -> None:
    """Ordering is the point: the line has to land before anything is spent.

    Proven by making compilation fail. If the report already ran, its warning is
    in the output even though the command died at compile, which is exactly the
    guarantee wanted: a misdirected destination is visible before a sandbox
    exists and before a single case is scored.
    """

    from click.testing import CliRunner

    from vero.cli import main
    from vero.harbor import build as harbor_build

    config_path = tmp_path / "build.yaml"
    config_path.write_text("name: org/task\n", encoding="utf-8")

    class _Config:
        harbor_requirement = "harbor[modal]==0.20.0"
        agent_env: dict[str, str] = {}
        optimizer_harbor_args: list[str] = []
        extra_harbor_args: list[str] = []
        name = "vero/stub-benchmark"
        wandb = SimpleNamespace(entity=None, project="heb", name="cell-1")

    def _explode(config, output):
        raise RuntimeError("compilation reached")

    monkeypatch.setattr(
        harbor_build, "load_harbor_build_config", lambda *a, **k: _Config()
    )
    monkeypatch.setattr(harbor_build, "compile_harbor_task", _explode)
    monkeypatch.setattr(harbor_cli.shutil, "which", lambda name: "/usr/bin/uvx")
    monkeypatch.setattr(harbor_cli, "_preflight_models", lambda config: None)

    result = CliRunner().invoke(
        main,
        [
            "harbor",
            "run",
            "--config",
            str(config_path),
            "--agent",
            "codex",
            "--model",
            "gpt-5.3-codex",
            "--yes",
        ],
    )

    # click >=8.2 splits the streams; the warning is deliberately on stderr.
    emitted = result.output + (result.stderr or "")
    assert "compilation reached" in str(result.exception)
    assert "WARNING" in emitted
    assert "personal" in emitted


def test_config_argument_is_required() -> None:
    """A diagnostic called with no config should fail loudly, not pass silently."""

    with pytest.raises(TypeError):
        harbor_cli._report_wandb_destination()


def test_config_without_a_wandb_attribute_is_a_no_op(capsys) -> None:
    """Degrade to silence, never to an exception.

    Several call sites and tests pass config shapes that carry no W&B settings at
    all. A diagnostic that raised on those would block the launch it exists to
    describe, which is a strictly worse failure than the one it prevents.
    """

    harbor_cli._report_wandb_destination(SimpleNamespace(name="org/task"))
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
