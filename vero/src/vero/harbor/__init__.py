"""Harbor integration: the sidecar-specific frontend over the shared
EvaluationEngine, plus Mode B (Harbor-delegated eval). The `harbor` SDK is an
optional extra, imported lazily (only registry enumeration / nested runs need it —
config, dataset compilation, and the sidecar handlers do not).
"""

from vero.harbor.config import HarborConfig
from vero.harbor.dataset import (
    build_harbor_dataset,
    enumerate_local_task_names,
    validate_partition,
)

__all__ = [
    "HarborConfig",
    "build_harbor_dataset",
    "enumerate_local_task_names",
    "validate_partition",
]
