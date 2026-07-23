"""Compile a program-optimization setup into a runnable Harbor task."""

from vero.harbor.build.compiler import compile_harbor_task
from vero.harbor.build.config import (
    AgentAccessSpec,
    HarborBuildConfig,
    InferenceBudgetSpec,
    InferenceGatewaySpec,
    VerificationTargetSpec,
    WandbSpec,
    WorkspaceOverlaySpec,
    load_harbor_build_config,
)

__all__ = [
    "AgentAccessSpec",
    "HarborBuildConfig",
    "InferenceBudgetSpec",
    "InferenceGatewaySpec",
    "VerificationTargetSpec",
    "WandbSpec",
    "WorkspaceOverlaySpec",
    "compile_harbor_task",
    "load_harbor_build_config",
]
