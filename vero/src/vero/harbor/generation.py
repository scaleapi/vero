"""Harbor-backed candidate generation.

A :class:`~vero.optimization.protocols.GenerationBackend` that produces
candidates by delegating to a ``harbor run`` — symmetric to how
:class:`~vero.harbor.backend.HarborBackend` nests ``harbor run`` for *evaluation*.
This is the contained / untrusted / multi-adapter production path (codex, claude,
terminus, …); the lightweight in-process native producer is the Optimizer's
default backend.

Status: interface skeleton. The full wiring (drive a ``harbor run`` for the
proposal's agent, then import the resulting commit into the session candidate
repository by object identity) is a follow-on; it reuses the existing
:class:`~vero.sidecar.transport.GitCandidateTransport` and the ``harbor.backend``
nesting rather than re-hosting Harbor's sidecar/gateway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from vero.candidate import Candidate
from vero.evaluation import EvaluationRecord
from vero.optimization.models import (
    CandidateProposal,
    GenerationOutcome,
    OptimizationContext,
)


@dataclass
class HarborGenerationBackend:
    """Generate candidates by running a Harbor agent over the parent.

    Reuses ``GitCandidateTransport`` to import the untrusted candidate commit
    into the session repository by object identity, and returns the Harbor
    sidecar's disclosed-partition scores as the generation-time feedback in
    ``GenerationOutcome.trial_evaluations``. Selection/target scoring remains the
    Optimizer's responsibility.
    """

    # Populated when the harbor-run-as-production wiring lands (task source,
    # agent, transport, candidate repository, session dir, …).

    async def generate(
        self,
        *,
        proposal: CandidateProposal,
        parent: Candidate,
        context: OptimizationContext,
        evaluation_records: Sequence[EvaluationRecord],
    ) -> GenerationOutcome:
        raise NotImplementedError(
            "HarborGenerationBackend is an interface skeleton; the native "
            "in-process backend (Optimizer default) is the implemented path. "
            "Wire this to a `harbor run` + GitCandidateTransport import to enable "
            "contained/multi-adapter production."
        )
