"""HarborConfig — the Mode-B configuration.

User-facing config that turns "evaluate my agent on a set of Harbor tasks" into a
`harbor run` invocation. A typed projection of the user-controllable `harbor run`
flags; the per-eval-derived flags (task selection, jobs dir, source/agent resolution)
are filled in by the runner, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HarborConfig:
    task_source: str  # registry ref "org/name[@ver]" OR a local path to a task dir/dataset
    agent_import_path: str  # module path to the candidate agent, e.g. "pkg.mod:Class"
    model: str | None = None
    environment: str = "modal"  # cloud provider (docker allowed for local testing)
    n_attempts: int = 1
    max_retries: int = 2
    reward_key: str | None = None  # primary reward; default pass -> reward -> mean
    # How to score a task when n_attempts > 1 produced several trials:
    #   "best": the existing behavior (clean trials preferred, then latest).
    #   "mean": average the reward across all scored attempts, dirty or clean
    #           (a timed-out attempt the verifier scored 0.0 still counts; only
    #           attempts with no rewards at all are excluded). This is the
    #           de-noising mode: noise shrinks ~1/sqrt(k), and the score
    #           estimates pass probability instead of pass@k (which "best"
    #           inflates toward).
    aggregate_attempts: str = "best"
    # Trusted source for the nested `harbor` CLI, as a uv requirement spec
    # (e.g. "harbor==0.1.17" or a pinned git URL). When set, the runner layers
    # it over the candidate env with `uv run --with`, whose ephemeral overlay
    # takes precedence for both the console script and sys.path — so the
    # orchestrator that scores the candidate resolves from THIS spec, not from
    # whatever the candidate's own pyproject/uv.lock pin (which the agent
    # controls, and could point at a fork that fabricates trial results
    # without running anything). None keeps the current behavior: the
    # candidate env supplies harbor, and is trusted to.
    harbor_requirement: str | None = None
    # Bounded within-eval retry for infra-destroyed samples. A sample whose
    # EVERY attempt died of a transient infrastructure cause (connection,
    # timeout, rate limit, 5xx) was never measured at all: re-run it after a
    # backoff instead of booking the outage as a permanent error. Measured
    # live: a 65-second host DNS blip killed 44 of 72 attempts of one eval
    # with ConnectionError, and nothing in the record distinguished the blip
    # from a bad candidate.
    #
    # OFF BY DEFAULT, and it must stay off when the candidate is an
    # adversarial optimizer. The qualifying predicate is built from exception
    # types raised inside candidate code, and agents are stochastic: a
    # candidate that raises an allowlisted exception whenever an attempt is
    # going badly loses nothing on partially-good samples (its fakes
    # zero-fill like honest failures) but converts every all-bad sample from
    # a booked 0.0 into a fresh re-roll. That is one-sided selection over
    # attempt sets, exactly what the zero-fill invariant exists to prevent.
    # Enable only for trusted-candidate evaluations (frozen agents,
    # operator-run matrices), where re-measuring an outage is pure signal
    # recovery. Candidate crashes and exhausted key budgets never retry
    # regardless (a crash is a result; a spent key cannot recover by
    # waiting), and recovered samples carry an ``infra_retry`` audit marker.
    infra_retry_rounds: int = 0
    # Backoff before retry round N is N times this many seconds. Transient
    # infra needs time, not immediacy: instant retries burned 6 of 8 run
    # attempts inside one live DNS blip.
    infra_retry_delay_s: float = 30.0
    extra_args: list[str] = field(default_factory=list)  # passthrough harbor run flags

    def __post_init__(self) -> None:
        # Only the exact string "mean" activates de-noising; without this check a
        # typo ("Mean", "avg") would silently run best-of-k with inflated scores.
        if self.aggregate_attempts not in ("best", "mean"):
            raise ValueError(
                f"aggregate_attempts must be 'best' or 'mean', got "
                f"{self.aggregate_attempts!r}"
            )

    @property
    def is_registry(self) -> bool:
        """Local if the source resolves to an existing path; otherwise a registry ref."""
        return not Path(self.task_source).expanduser().exists()

    def source_args(self) -> list[str]:
        """`harbor run` source selector: `-d <ref>` (registry) or `-p <path>` (local)."""
        if self.is_registry:
            return ["-d", self.task_source]
        return ["-p", str(Path(self.task_source).expanduser())]
