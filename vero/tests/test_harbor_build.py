"""Unit test for the `vero harbor build` compiler: a BuildConfig compiles to a
well-formed Harbor task directory whose ServeConfig validates and whose rendered
task.toml / compose / scripts parse. No Docker (that's the e2e)."""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

from vero.harbor.build import BuildConfig, compile_task
from vero.harbor.serve import ServeConfig


def _stub_vero(root: Path) -> Path:
    """A minimal stand-in for the vero source tree (compiler just copies it)."""
    d = root / "vero-src"
    (d / "src" / "vero").mkdir(parents=True)
    (d / "pyproject.toml").write_text("[project]\nname='scale-vero'\nversion='0'\n")
    (d / "README.md").write_text("vero\n")
    (d / "src" / "vero" / "__init__.py").write_text("")
    return d


def _agent_repo(root: Path) -> Path:
    d = root / "agent"
    (d / "src" / "gsm8k_agent").mkdir(parents=True)
    (d / "pyproject.toml").write_text(
        "[project]\nname='gsm8k-agent'\nversion='0'\n\n"
        '[tool.uv.sources]\nscale-vero = { path = "../../", editable = true }\n'
    )
    (d / "src" / "gsm8k_agent" / "agent.py").write_text("X = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    subprocess.run(["git", "add", "-A"], cwd=d, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"],
        cwd=d, check=True,
    )
    return d


def _dataset(root: Path) -> Path:
    from datasets import Dataset, DatasetDict

    ds = DatasetDict({
        "validation": Dataset.from_dict({"question": ["1+1?"], "answer": ["#### 2"]}),
        "test": Dataset.from_dict({"question": ["2+2?"], "answer": ["#### 4"]}),
    })
    p = root / "ds"
    ds.save_to_disk(str(p))
    return p


@pytest.fixture
def built(tmp_path, monkeypatch):
    monkeypatch.setenv("VERO_SKIP_SECRET_CHECK", "1")
    config = BuildConfig(
        name="vero/gsm8k-opt",
        description="optimize gsm8k",
        agent_repo=str(_agent_repo(tmp_path)),
        mode="A",
        task="gsm8k",
        task_module="gsm8k_agent.vero_tasks",
        dataset=str(_dataset(tmp_path)),
        splits=[
            {"split": "validation", "access": "non_viewable"},
            {"split": "test", "access": "no_access"},
        ],
        budgets=[{"split": "validation", "total_run_budget": 5}],
        reward_mode="auto_best",
        selection_split="validation",
        targets=[{"split": "test", "reward_key": "reward"}],
        read_only_paths=["src/gsm8k_agent/vero_tasks"],
        secrets=["OPENAI_API_KEY"],
    )
    out = compile_task(config, tmp_path / "task", vero_root=_stub_vero(tmp_path))
    return out


def test_structure(built):
    for rel in [
        "task.toml", "instruction.md",
        "environment/docker-compose.yaml", "environment/Dockerfile",
        "environment/sidecar/Dockerfile", "environment/sidecar/serve.json",
        "environment/main/seed.sh", "environment/vero/pyproject.toml",
        "environment/agent-baseline/.git", "environment/agent-seed/.git",
        "environment/sidecar/vero_home", "tests/test.sh", "solution/solve.sh",
    ]:
        assert (built / rel).exists(), f"missing {rel}"


def test_serve_config_validates(built):
    cfg = ServeConfig.from_file(built / "environment" / "sidecar" / "serve.json")
    assert cfg.repo_path == "/opt/agent-baseline"
    assert cfg.agent_repo_path == "/work/agent"
    assert cfg.task == "gsm8k"
    assert cfg.dataset_id  # registered
    assert cfg.base_commit  # baseline sha recorded for auto_best exclusion
    assert cfg.targets and cfg.targets[0].split == "test"
    assert cfg.budgets[0]["dataset_id"] == cfg.dataset_id


def test_score_baseline_reaches_serve_json(built):
    # Raw JSON on purpose: the key must be present in the compiler <-> serve
    # contract even where the local ServeConfig predates the field.
    raw = json.loads((built / "environment" / "sidecar" / "serve.json").read_text())
    assert raw["score_baseline"] is False  # default off


def test_score_baseline_true_emitted():
    # Through the actual YAML path (not just the BuildConfig constructor), so
    # the headline claim "reachable from build.yaml" is what is tested.
    from vero.harbor.build.compiler import _serve_config

    config = BuildConfig.model_validate(yaml.safe_load(
        "name: o/n\n"
        "agent_repo: .\n"
        "splits:\n"
        "  - {split: validation, access: non_viewable}\n"
        "score_baseline: true\n"
    ))
    assert config.score_baseline is True
    raw = _serve_config(config, "ds", "sha")
    assert raw["score_baseline"] is True


def test_rendered_files_parse(built):
    tomllib.loads((built / "task.toml").read_text())  # valid TOML
    compose = yaml.safe_load((built / "environment/docker-compose.yaml").read_text())
    assert "eval-sidecar" in compose["services"]
    assert compose["services"]["main"]["depends_on"]["eval-sidecar"]["condition"] == "service_healthy"
    # secret reaches the sidecar only, via host-resolved compose interpolation
    # with a fail-fast guard (${VAR:?msg}) so an unset host var aborts the run.
    sidecar_secret = compose["services"]["eval-sidecar"]["environment"]["OPENAI_API_KEY"]
    assert sidecar_secret.startswith("${OPENAI_API_KEY:?")
    assert "OPENAI_API_KEY" not in compose["services"]["main"].get("environment", {})


def test_vero_source_path_rewritten(built):
    pyproject = (built / "environment/agent-baseline/pyproject.toml").read_text()
    assert 'path = "/opt/vero"' in pyproject
    assert "../../" not in pyproject


def test_baseline_sha_shared(built):
    def head(p):
        return subprocess.run(
            ["git", "-C", str(built / p), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    assert head("environment/agent-baseline") == head("environment/agent-seed")
    cfg = json.loads((built / "environment/sidecar/serve.json").read_text())
    assert cfg["base_commit"] == head("environment/agent-baseline")


def test_baseline_archive_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("VERO_SKIP_SECRET_CHECK", "1")
    # An empty git repo with no commits: `git archive HEAD` exits nonzero.
    bad = tmp_path / "emptyrepo"
    (bad / "src").mkdir(parents=True)
    (bad / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
    subprocess.run(["git", "init", "-q"], cwd=bad, check=True)
    config = BuildConfig(
        name="vero/x", agent_repo=str(bad), mode="A", task="gsm8k",
        dataset=str(_dataset(tmp_path)),
        splits=[{"split": "validation", "access": "non_viewable"}],
    )
    with pytest.raises(RuntimeError, match="git archive failed"):
        compile_task(config, tmp_path / "task", vero_root=_stub_vero(tmp_path))


def test_missing_secret_fails_build(tmp_path, monkeypatch):
    monkeypatch.delenv("VERO_SKIP_SECRET_CHECK", raising=False)
    monkeypatch.delenv("DEFINITELY_MISSING_SECRET", raising=False)
    config = BuildConfig(
        name="vero/x", agent_repo=str(_agent_repo(tmp_path)), mode="A", task="gsm8k",
        dataset=str(_dataset(tmp_path)),
        splits=[{"split": "validation", "access": "non_viewable"}],
        secrets=["DEFINITELY_MISSING_SECRET"],
    )
    with pytest.raises(ValueError, match="missing from the host environment"):
        compile_task(config, tmp_path / "task", vero_root=_stub_vero(tmp_path))


def test_seed_documents_advisory_read_only(built):
    seed = (built / "environment/main/seed.sh").read_text()
    assert "ADVISORY ONLY" in seed
    assert "sidecar-side" in seed
