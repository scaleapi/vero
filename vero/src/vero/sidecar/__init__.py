"""The trusted side of the evaluation boundary.

Owns held-out partitions, disclosure, budgets, and final scoring — everything the
optimizer must not be able to reach. Deployed as the ``eval-sidecar`` container
beside the optimizer's ``main`` container, which is where the name comes from.

Two invariants worth knowing:

- Nothing here is Harbor-specific. This package depends on ``vero.evaluation``
  only; ``vero.harbor.deployment`` is the single module that binds this stack to
  a Harbor-backed evaluation, and a compiled task can just as well use a command
  backend (see ``examples/harbor-circle-packing``).
- Co-location is load-bearing. HTTP carries the API, but candidate code, the
  ``.evals`` result context, and the admin token all move through volumes shared
  with ``main`` — so this is not, today, deployable away from the task it serves.

``sidecar`` is the agent-facing API and is transport-neutral; ``app`` is an
optional FastAPI transport over it; ``serve`` is the composition root that loads
a factory, builds the components, and runs them.
"""

from vero.sidecar.session import HarborSessionManifest
from vero.sidecar.sidecar import (
    EvaluationAccessError,
    EvaluationAccessStatus,
    EvaluationJobNotFoundError,
    EvaluationJobStatus,
    EvaluationSidecar,
    SidecarEvaluationJob,
    SidecarEvaluationPolicy,
    SidecarEvaluationRequest,
    SidecarEvaluationResult,
    SidecarStatus,
    Submission,
    SubmissionDisabledError,
)
from vero.sidecar.transport import (
    CandidateTransferError,
    CandidateTransport,
    GitCandidateTransport,
)
from vero.sidecar.verifier import (
    CanonicalVerifier,
    NoCandidateError,
    VerificationResult,
    VerificationSelection,
    VerificationTarget,
)

__all__ = [
    "CandidateTransferError",
    "CandidateTransport",
    "CanonicalVerifier",
    "EvaluationAccessError",
    "EvaluationAccessStatus",
    "EvaluationJobNotFoundError",
    "EvaluationJobStatus",
    "EvaluationSidecar",
    "GitCandidateTransport",
    "HarborSessionManifest",
    "NoCandidateError",
    "SidecarEvaluationJob",
    "SidecarEvaluationPolicy",
    "SidecarEvaluationRequest",
    "SidecarEvaluationResult",
    "SidecarStatus",
    "Submission",
    "SubmissionDisabledError",
    "VerificationResult",
    "VerificationSelection",
    "VerificationTarget",
]
