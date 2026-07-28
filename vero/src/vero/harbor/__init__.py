"""Harbor as VeRO's task substrate: compile a task, and evaluate on it.

``build`` compiles a benchmark's build YAML into a runnable Harbor task
directory. ``backend`` evaluates one candidate by nesting a ``harbor run`` per
case — a peer of the backends in ``vero.evaluation.backends``. ``deployment`` is
the standard sidecar factory, and the one module that binds the trusted runtime
in ``vero.sidecar`` to a Harbor-backed evaluation; a task with a command backend
uses the same runtime and no Harbor backend at all.

The trusted runtime itself lives in ``vero.sidecar``, and the inference gateway
in ``vero.gateway``. Neither is a Harbor concept.
"""

from vero.harbor.backend import HarborBackend, HarborBackendConfig, HarborCase
from vero.harbor.build import (
    AgentAccessSpec,
    CommandBackendSpec,
    HarborBuildConfig,
    InferenceBudgetSpec,
    InferenceGatewaySpec,
    VerificationTargetSpec,
    WandbSpec,
    WorkspaceOverlaySpec,
    compile_harbor_task,
    load_harbor_build_config,
)
from vero.harbor.deployment import HarborDeploymentConfig, build_harbor_components

__all__ = [
    "HarborBackend",
    "HarborBackendConfig",
    "HarborCase",
    "HarborDeploymentConfig",
    "AgentAccessSpec",
    "CommandBackendSpec",
    "HarborBuildConfig",
    "InferenceBudgetSpec",
    "InferenceGatewaySpec",
    "VerificationTargetSpec",
    "WandbSpec",
    "WorkspaceOverlaySpec",
    "build_harbor_components",
    "compile_harbor_task",
    "load_harbor_build_config",
]
