"""Invariants the checked-in benchmark build YAMLs must satisfy.

These read harness-engineering-bench, so they live on the branch that has it. If
that directory is missing the tests error rather than pass vacuously, which is
how their predecessors drifted unnoticed.

The one thing they do skip for is a benchmark whose tasks have not been vendored
yet -- see _require_vendored_task_source. That is a precondition of the machine,
not a property of the config, and skipping it names the missing directory rather
than quietly reporting green.

Assertions are derived from each config wherever possible. Pinning literal model
names is what broke the earlier version -- a benchmark switched target model and
the expectation was never updated -- so the rules below say what must hold
between fields, not what today's values happen to be.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vero.harbor import load_harbor_build_config

BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "harness-engineering-bench"

BENCHMARKS = ["gaia", "officeqa", "swe-atlas-qna", "tau3", "browsecomp-plus"]

# Names that would let a task reach the upstream provider directly, bypassing
# the gateway's allow-list and budget.
UPSTREAM_CREDENTIALS = {"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"}


def _require_vendored_task_source(path: Path) -> None:
    """Skip when a benchmark's task_source names a directory nobody fetched yet.

    officeqa and browsecomp-plus point task_source at a gitignored tasks/ tree
    built by their own scripts/. The loader only absolutizes a relative
    task_source when it exists, so on a fresh clone the path stays literal and
    is rejected as an unpinned registry reference -- these tests would fail on
    every checkout that had not run the script, which reads as a broken branch
    rather than a missing prerequisite.

    Deliberately narrower than skipping the whole module when BENCHMARK_ROOT is
    absent: that blanket guard is what let the earlier version rot.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    task_source = document.get("task_source")
    if not isinstance(task_source, str) or "@" in task_source or "${" in task_source:
        return  # a registry reference or a build parameter; nothing to vendor
    resolved = Path(task_source)
    if not resolved.is_absolute():
        resolved = path.parent / resolved
    if not resolved.exists():
        pytest.skip(
            f"{path.parent.parent.name} tasks are not vendored: {resolved} is "
            f"missing; run its scripts/ to fetch them"
        )


def _config(benchmark: str):
    path = BENCHMARK_ROOT / benchmark / "baseline" / "build.yaml"
    _require_vendored_task_source(path)
    return load_harbor_build_config(path)


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_benchmarks_route_all_inference_through_the_gateway(benchmark):
    """No benchmark may hand a task the raw upstream credential."""
    config = _config(benchmark)

    assert config.inference_gateway is not None, "benchmarks must meter inference"
    # The upstream credential is the gateway's alone. Declaring it as a task
    # secret would deliver it straight to the containers it is meant to bypass.
    assert not UPSTREAM_CREDENTIALS.intersection(config.secrets)
    assert config.inference_gateway.upstream_api_key_env == "OPENAI_API_KEY"
    assert config.inference_gateway.upstream_base_url_env == "OPENAI_BASE_URL"


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_target_model_is_the_only_model_the_evaluation_scope_allows(benchmark):
    """The measurement substrate is fixed: one target model, allow-listed.

    Derived from config.model rather than pinned, so switching a benchmark's
    target model cannot leave this test asserting the old one.
    """
    config = _config(benchmark)
    evaluation = config.inference_gateway.evaluation

    assert config.model is not None
    assert evaluation.allowed_models == [config.model]
    # Search is budgeted; an unbounded evaluation scope would make the token
    # cost of a run unbounded too.
    assert evaluation.max_requests is not None
    assert evaluation.max_tokens is not None


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_optimizer_and_target_models_are_separately_scoped(benchmark):
    """Producer and evaluation are distinct scopes, so neither can spend the other's budget."""
    gateway = _config(benchmark).inference_gateway

    assert gateway.producer.allowed_models
    # Whatever the two are set to, they are independent policies rather than one
    # shared pool; the compiler mints a separate token per scope.
    assert gateway.producer is not gateway.evaluation


def test_build_params_override_run_time_knobs_without_rebuild():
    path = BENCHMARK_ROOT / "gaia" / "baseline" / "build.yaml"

    default = load_harbor_build_config(path)
    assert default.environment_name == "modal"
    default_producer = default.inference_gateway.producer.allowed_models

    overridden = load_harbor_build_config(
        path, params={"optimizer_model": "gpt-5.5", "inner_env": "docker"}
    )
    assert overridden.environment_name == "docker"
    assert overridden.inference_gateway.producer.allowed_models == ["gpt-5.5"]
    assert overridden.inference_gateway.producer.allowed_models != default_producer
    # The rest of the measurement substrate is untemplated and stays fixed.
    assert overridden.model == default.model
    assert overridden.task_source == default.task_source
