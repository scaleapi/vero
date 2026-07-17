from __future__ import annotations

import pytest
from pydantic import ValidationError

from vero.config import AgentOptimizerConfig, VeroConfig, _producer, load_config


def _config(optimizer: dict) -> dict:
    return {
        "target": {"root": "."},
        "backend": {
            "harness_root": ".",
            "command": ["evaluate", "{workspace}", "{report}"],
        },
        "evaluations": [{"name": "default"}],
        "protocol": {"selection_evaluation": "default"},
        "objective": {"metric": "score", "direction": "maximize"},
        "optimizer": optimizer,
    }


def test_agent_optimizer_accepts_only_agent_fields():
    config = VeroConfig.model_validate(
        _config(
            {
                "kind": "vero",
                "instruction": "Improve the program",
                "model": "openai/gpt-5",
                "max_turns": 10,
            }
        )
    )

    assert isinstance(config.optimizer, AgentOptimizerConfig)
    assert config.optimizer.model == "openai/gpt-5"
    assert config.optimizer.max_turns == 10


def test_agent_optimizer_rejects_empty_model():
    with pytest.raises(ValidationError, match="model must not be empty"):
        VeroConfig.model_validate(_config({"kind": "vero", "model": " "}))


@pytest.mark.parametrize(
    ("kind", "model"),
    [("vero", "openai/gpt-5"), ("claude", "claude-opus-4-1")],
)
def test_agent_optimizer_applies_configured_model(kind, model):
    producer = _producer(AgentOptimizerConfig(kind=kind, model=model))

    configured = (
        producer.agent.model_str()
        if kind == "vero"
        else producer.agent.options.model
    )
    assert configured == model


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("root", "producer"),
        ("working_directory", "src"),
        ("environment", {"NAME": "value"}),
        ("passthrough_environment", ["TOKEN"]),
        ("timeout_seconds", 10),
        ("description", "ignored"),
        ("command", ["ignored"]),
    ],
)
def test_agent_optimizer_rejects_command_only_fields(field, value):
    optimizer = {"kind": "claude", field: value}

    with pytest.raises(ValidationError, match=field):
        VeroConfig.model_validate(_config(optimizer))


def test_config_rejects_case_ids_combined_with_range_start():
    value = _config({"kind": "vero"})
    value["evaluations"][0].update({"case_ids": ["a"], "case_start": 1})

    with pytest.raises(ValidationError, match="case_ids and case range"):
        VeroConfig.model_validate(value)


def test_config_wires_retry_policy_into_evaluation_limits():
    value = _config({"kind": "vero"})
    value["protocol"]["retry"] = {
        "max_attempts": 5,
        "initial_delay_seconds": 0.5,
    }

    config = VeroConfig.model_validate(value)

    assert config.protocol.to_limits().retry.max_attempts == 5
    assert config.protocol.to_limits().retry.initial_delay_seconds == 0.5


def test_config_wires_case_failure_and_error_rate_policy():
    value = _config({"kind": "vero"})
    value["protocol"]["error_rate_threshold"] = 0.25
    value["objective"].update(
        {"aggregation": "mean", "case_failure_value": 0.0}
    )

    config = VeroConfig.model_validate(value)

    assert config.protocol.to_limits().error_rate_threshold == 0.25
    assert config.objective.to_model().selector.case_failure_value == 0.0


def test_config_resolves_agent_context_inputs_with_staged_inputs(tmp_path):
    config_path = tmp_path / "vero.toml"
    config_path.write_text(
        """
[target]
root = "target"

[backend]
harness_root = "harness"
command = ["run", "{input:cases}"]

[backend.staged_inputs]
cases = "data/cases.json"

[backend.agent_context_inputs]
train = ["cases"]

[[evaluations]]
name = "train"

[protocol]
selection_evaluation = "train"

[objective]
metric = "score"
direction = "maximize"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.backend.agent_context_inputs == {"train": ["cases"]}
    assert config.backend.staged_inputs == {
        "cases": str((tmp_path / "data/cases.json").resolve())
    }
