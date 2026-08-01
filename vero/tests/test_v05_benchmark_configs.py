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

BENCHMARKS = [
    "gaia",
    "officeqa",
    "swe-atlas-qna",
    "tau3",
    "browsecomp-plus",
    "swe-bench-pro",
]

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
def test_upstream_rerouting_benchmarks_keep_their_agent_on_the_gateway(benchmark):
    """`task_services_use_upstream` obliges the target agent to read dedicated vars.

    That flag exists so an in-container grader or user-simulator can reach the
    provider, and it does that by pointing OPENAI_* at the real upstream. The
    candidate agent's metered, allow-listed credential then arrives only on
    VERO_AGENT_INFERENCE_*. An agent that does not read those falls through to
    OPENAI_* and runs on the raw upstream: unmetered, and with the pinned target
    model unenforced.

    The config-level test above cannot see this -- the build still declares a
    gateway, and vero injects the upstream deliberately -- so it passed while
    browsecomp-plus ran every evaluation off-gateway. This asserts against the
    agent source, which is where the contract is actually kept.
    """
    from vero.harbor.backend import (
        AGENT_INFERENCE_API_KEY_ENV,
        AGENT_INFERENCE_BASE_URL_ENV,
    )

    config = _config(benchmark)
    if not config.task_services_use_upstream:
        pytest.skip(f"{benchmark} does not reroute OPENAI_* to the upstream")

    sources = sorted(
        (BENCHMARK_ROOT / benchmark / "baseline" / "target" / "src").rglob("*.py")
    )
    assert sources, f"{benchmark} has no target agent source to check"
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    for name in (AGENT_INFERENCE_API_KEY_ENV, AGENT_INFERENCE_BASE_URL_ENV):
        assert name in text, (
            f"{benchmark} reroutes OPENAI_* to the upstream but its agent never "
            f"reads {name}, so its target inference bypasses the gateway"
        )


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


def test_terminal_bench_azure_variant_differs_only_by_the_model_alias():
    """build.azure.yaml must be build.yaml plus one alias, and nothing else.

    The variant exists only because codex cannot put a provider-qualified model
    id on the wire, so the gateway has to pin `gpt-5.6-sol` to a single Azure
    deployment on its behalf. Everything else -- budgets, timeouts, partitions,
    the pinned baseline -- has to stay identical, or the cell that uses this file
    stops being comparable to the nine that use build.yaml.

    There is no include/extends mechanism for these YAMLs, so the variant is a
    183-line copy. That is a drift hazard with no natural alarm: divergence in a
    timeout or a budget would change results and fail nothing. This test is the
    alarm. If you deliberately change build.yaml, mirror it here and the test
    passes again; if you forget, it does not.
    """
    baseline = BENCHMARK_ROOT / "terminal-bench" / "baseline"
    shared = yaml.safe_load((baseline / "build.yaml").read_text(encoding="utf-8"))
    variant = yaml.safe_load((baseline / "build.azure.yaml").read_text(encoding="utf-8"))

    alias = variant["inference_gateway"]["producer"].pop("model_aliases")
    assert alias == {"gpt-5.6-sol": "azure_ai/gpt-5.6-sol"}
    # Compared after popping the alias: the two documents must now be equal.
    assert variant == shared, (
        "build.azure.yaml has drifted from build.yaml beyond the model alias; "
        "mirror the change or the azure cell is no longer comparable"
    )

    # And the alias must actually reach the gateway for the cell that needs it,
    # while staying inert for a cell whose optimizer is something else.
    sol = load_harbor_build_config(
        baseline / "build.azure.yaml", params={"optimizer_model": "gpt-5.6-sol"}
    )
    other = load_harbor_build_config(
        baseline / "build.azure.yaml", params={"optimizer_model": "claude-opus-5"}
    )
    assert sol.inference_gateway.producer.allowed_models == ["gpt-5.6-sol"]
    assert sol.inference_gateway.producer.model_aliases == {
        "gpt-5.6-sol": "azure_ai/gpt-5.6-sol"
    }
    # Present but unreachable: the allow-list admits only claude-opus-5, and an
    # alias can fire only for a model the allow-list already passed.
    assert other.inference_gateway.producer.allowed_models == ["claude-opus-5"]
    assert "claude-opus-5" not in other.inference_gateway.producer.model_aliases
