"""Compile a program-optimization setup into a runnable Harbor task.

Four modules, in the order a build passes through them: specs.py declares the
leaf models a build.yaml composes, config.py assembles them into
HarborBuildConfig and enforces the rules that span groups, loader.py reads the
YAML into one, and compiler.py lowers it into a task directory.
"""

from vero.harbor.build.compiler import compile_harbor_task
from vero.harbor.build.config import HarborBuildConfig
from vero.harbor.build.loader import load_harbor_build_config
from vero.harbor.build.specs import (
    AgentAccessSpec,
    CommandBackendSpec,
    InferenceBudgetSpec,
    InferenceGatewaySpec,
    VerificationTargetSpec,
    WandbSpec,
    WorkspaceOverlaySpec,
)

__all__ = [
    "AgentAccessSpec",
    "CommandBackendSpec",
    "HarborBuildConfig",
    "InferenceBudgetSpec",
    "InferenceGatewaySpec",
    "VerificationTargetSpec",
    "WandbSpec",
    "WorkspaceOverlaySpec",
    "compile_harbor_task",
    "load_harbor_build_config",
]
