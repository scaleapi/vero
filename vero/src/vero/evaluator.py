"""Back-compat shim. The implementation moved to ``vero.evaluation.evaluator``.

Prefer importing from ``vero.evaluation`` going forward; this module is kept so
existing ``from vero.evaluator import ...`` imports (examples, external code) keep
working.
"""

from vero.evaluation.evaluator import (  # noqa: F401
    Evaluator,
    _resolve_vero_dependency,
    isolate_project,
    run_evaluation,
)

__all__ = ["Evaluator", "isolate_project", "run_evaluation", "_resolve_vero_dependency"]
