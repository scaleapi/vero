"""Canonical schema for interpretability analysis.

Every source adapter normalises into these types, so downstream code never sees a
harbor path, a tarball, or a git object. Adding a second producer of optimization
runs means writing an adapter, not touching anything below `artifacts/`.

Identity is content-addressed throughout. `Edit.id` in particular is derived from the
edit's own content rather than its position, so a cached label survives re-extraction,
re-ordering, and unrelated edits appearing earlier in the same candidate.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, Field


def content_id(*parts: str) -> str:
    """Stable 16-hex identity over the given parts."""
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8", "replace"))
    return digest.hexdigest()[:16]


class SymbolKind(StrEnum):
    """What kind of thing an edit landed in, resolved from the syntax tree."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    PROMPT_TEXT = "prompt_text"       # module-level str binding, long
    SCALAR_CONST = "scalar_const"     # module-level int/float/bool/short str
    COLLECTION = "collection"         # module-level tuple/list/dict/set
    REGEX = "regex"                   # module-level re.compile
    MODULE = "module"                 # module level, unattributed
    NON_PYTHON = "non_python"         # config, markdown, lockfiles


class CellRef(BaseModel):
    """One optimization run: a single (source, benchmark, cell) triple."""

    source: str                       # adapter name, e.g. "harbor"
    benchmark: str
    cell: str
    root: str                         # directory the adapter was pointed at
    cell_dir: str                     # resolved during discovery, not recomputed
    job_dir: str | None = None

    @property
    def key(self) -> str:
        return f"{self.source}/{self.benchmark}/{self.cell}"


class Candidate(BaseModel):
    """One state the optimizer produced. Position 0 is the seed."""

    sha: str
    parent_sha: str | None = None
    position: int
    subject: str = ""
    body: str = ""
    files: list[str] = Field(default_factory=list)
    insertions: int = 0
    deletions: int = 0
    tree_sha: str | None = None
    is_seed: bool = False
    is_shipped: bool = False


class EvalRecord(BaseModel):
    """One scoring of one candidate.

    A candidate can be scored more than once on the same partition; those repeats are
    the corpus's only direct measurement of evaluation noise, so they are kept as
    separate records rather than collapsed into a per-partition map.
    """

    candidate_sha: str
    partition: str
    score: float
    error_rate: float | None = None
    selection_kind: str | None = None   # "all" for a full partition, else a slice
    n_attempts: int | None = None
    started_at: str | None = None
    finished_at: str | None = None


class Edit(BaseModel):
    """A symbol-scoped change: the unit of analysis.

    Not the commit. A single candidate routinely bundles a bug fix with unrelated
    tuning under one subject line, so labelling per candidate assigns one category to
    several distinct modifications.
    """

    id: str
    cell_key: str
    candidate_sha: str
    path: str
    symbol: str                        # "<module>" when unattributed
    symbol_kind: SymbolKind
    added: int = 0
    removed: int = 0
    before_value: str | None = None    # scalar constants only
    after_value: str | None = None
    diff: str = ""                     # unified diff restricted to this symbol
    after_source: str | None = None    # the symbol's full text after the edit
    in_seed: bool = True               # did this symbol exist in the seed at all?
    prior_touches: int = 0             # earlier candidates in this cell that touched it
    # Whose defect was being repaired, decided by comparing trees rather than by
    # asking the model, which answers "own" almost everywhere. See edits.provenance.
    provenance: str = "unknown"

    @staticmethod
    def make_id(cell_key: str, sha: str, path: str, symbol: str, diff: str) -> str:
        return content_id(cell_key, sha, path, symbol, diff)


class EditLabel(BaseModel):
    """A model-assigned reading of one edit, plus the facets that were derived.

    `hinted` records whether the role came from a deterministic rule rather than the
    model, so agreement between the two can be measured instead of assumed.
    """

    edit_id: str
    action: str
    role: str
    # The model's own reading of provenance, kept only so the derived answer can be
    # audited against it. The report reads `Edit.provenance`; naming both fields
    # `provenance` is what let the model's version reach the figures unnoticed.
    model_provenance: str = "unknown"
    direction: str = "na"
    mechanism: str = ""              # one line, the model's own words
    confidence: float = 0.0
    hinted: bool = False
    model: str = ""
    taxonomy_version: str = ""


class Trajectory(BaseModel):
    """Everything known about one cell."""

    ref: CellRef
    candidates: list[Candidate] = Field(default_factory=list)
    evaluations: list[EvalRecord] = Field(default_factory=list)
    edits: list[Edit] = Field(default_factory=list)
    labels: list[EditLabel] = Field(default_factory=list)
    reward: float | None = None
    baseline_reward: float | None = None
    error_rate: float | None = None
    total_tokens: float | None = None

    @property
    def seed(self) -> Candidate | None:
        return next((c for c in self.candidates if c.is_seed), None)

    @property
    def shipped(self) -> Candidate | None:
        return next((c for c in self.candidates if c.is_shipped), None)

    @property
    def shipped_the_seed(self) -> bool:
        """True when the shipped tree is byte-identical to the seed tree.

        Decided on tree hashes, never on a commit message: messages saying "revert"
        routinely carry surviving behavioural change, and messages saying nothing of
        the sort are sometimes total reverts.
        """
        seed, shipped = self.seed, self.shipped
        if not (seed and shipped and seed.tree_sha and shipped.tree_sha):
            return False
        return seed.tree_sha == shipped.tree_sha


class Corpus(BaseModel):
    """A collated set of trajectories, usually one analysis scope."""

    trajectories: list[Trajectory] = Field(default_factory=list)

    def by_benchmark(self) -> dict[str, list[Trajectory]]:
        out: dict[str, list[Trajectory]] = {}
        for t in self.trajectories:
            out.setdefault(t.ref.benchmark, []).append(t)
        return out

    def edits(self):
        for t in self.trajectories:
            yield from t.edits
