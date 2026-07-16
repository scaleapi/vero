from __future__ import annotations

import pytest
from pydantic import ValidationError

from vero.config import AgentOptimizerConfig, VeroConfig


def _config(optimizer: dict) -> dict:
    return {
        "target": {"root": "."},
        "evaluation": {
            "harness_root": ".",
            "command": ["evaluate", "{workspace}", "{report}"],
        },
        "objective": {"metric": "score", "direction": "maximize"},
        "optimizer": optimizer,
    }


def test_agent_optimizer_accepts_only_agent_fields():
    config = VeroConfig.model_validate(
        _config(
            {
                "kind": "vero",
                "instruction": "Improve the program",
                "max_turns": 10,
            }
        )
    )

    assert isinstance(config.optimizer, AgentOptimizerConfig)
    assert config.optimizer.max_turns == 10


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
    value["evaluation"].update({"case_ids": ["a"], "case_start": 1})

    with pytest.raises(ValidationError, match="case_ids and case range"):
        VeroConfig.model_validate(value)


def test_config_wires_retry_policy_into_evaluation_limits():
    value = _config({"kind": "vero"})
    value["evaluation"]["retry"] = {
        "max_attempts": 5,
        "initial_delay_seconds": 0.5,
    }

    config = VeroConfig.model_validate(value)

    assert config.evaluation.to_limits().retry.max_attempts == 5
    assert config.evaluation.to_limits().retry.initial_delay_seconds == 0.5
