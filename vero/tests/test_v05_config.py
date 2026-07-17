from __future__ import annotations

import pytest
from pydantic import ValidationError

from vero.config import AgentOptimizerConfig, VeroConfig, load_config


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
