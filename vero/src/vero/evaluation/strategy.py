"""The evaluation strategy seam.

The Evaluator handles the shared lifecycle (clean-tree check, result store, checkout,
ExperimentResult assembly) and delegates the mode-specific step — "produce per-sample
results for this candidate/split/sample_ids" — to an EvalStrategy.

The default (Mode A) path is the in-process ``task.utils`` subprocess, kept inline in
the Evaluator. A non-default strategy (e.g. Harbor Mode B, injected from ``vero.harbor``)
implements this Protocol; the Evaluator never imports the strategy's module, keeping
``vero.evaluation`` harbor-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vero.core.evaluation import EvaluationParameters
    from vero.workspace import Workspace


@runtime_checkable
class EvalStrategy(Protocol):
    async def produce_sample_results(
        self,
        *,
        workspace: Workspace,
        params: EvaluationParameters,
        result_dir: Path,
    ) -> None:
        """Run the evaluation for ``params.run`` (commit/split/sample_ids) against the
        checked-out ``workspace`` and persist per-sample ``SampleResult``s to the result
        store (so ``Evaluator`` can assemble them into an ``ExperimentResult``)."""
        ...
