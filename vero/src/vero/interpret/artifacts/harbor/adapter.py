"""Harbor adapter: raw job artifacts in, canonical `Trajectory` out.

Layout it expects, which is what `vero harbor run` writes:

    <root>/<benchmark>/<cell>/jobs/<timestamp>/task__*/verifier/
        finalization.json     the shipped candidate and its held-out reward
        session.tar.gz        the candidate repo and the sidecar evaluation records

`<root>` may be a benchmark directory or a tree of them; discovery handles both, so
callers can point at one benchmark or a whole runs/ tree without knowing the depth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from vero.interpret.artifacts.base import register
from vero.interpret.artifacts.harbor import session as session_mod
from vero.interpret.artifacts.harbor.repo import CandidateRepo
from vero.interpret.cache import Cache
from vero.interpret.models import Candidate, CellRef, EvalRecord, Trajectory

# `shipped AND error_rate == 0.0 AND tokens > 0` is the flag harbor's own tooling
# uses, but every benchmark's build config sets error_rate_threshold: 0.1, so the
# exact-zero test discards good runs that lost one case to a platform hiccup. The
# benchmark's own threshold is the defensible line.
DEFAULT_MAX_ERROR_RATE = 0.1


class HarborAdapter:
    name = "harbor"

    def __init__(self, cache: Cache) -> None:
        self.cache = cache

    # -- discovery ------------------------------------------------------------

    def discover(self, roots: Iterable[Path]) -> list[CellRef]:
        refs: list[CellRef] = []
        seen: set[tuple[str, str]] = set()
        for root in roots:
            root = Path(root)
            if not root.is_dir():
                continue
            for final in sorted(root.glob("**/jobs/*/task__*/verifier/finalization.json")):
                # .../<benchmark>/<cell>/jobs/<timestamp>/task__*/verifier/<file>
                job_dir = final.parents[2]
                cell_dir = final.parents[4]
                benchmark = final.parents[5].name
                identity = (benchmark, cell_dir.name)
                if identity in seen:
                    continue
                seen.add(identity)
                refs.append(
                    CellRef(
                        source=self.name,
                        benchmark=benchmark,
                        cell=cell_dir.name,
                        root=str(root),
                        cell_dir=str(cell_dir),
                        job_dir=str(job_dir),
                    )
                )
        return refs

    # -- loading --------------------------------------------------------------

    def load(self, ref: CellRef) -> Trajectory:
        cell_dir = Path(ref.cell_dir)
        finals = sorted(cell_dir.glob("jobs/*/task__*/verifier/finalization.json"))
        if not finals:
            return Trajectory(ref=ref)

        # Last job wins: a cell re-run in place leaves the earlier attempt behind.
        final_path = finals[-1]
        final = json.loads(final_path.read_text())
        metrics = (final.get("reward_metrics") or {}).get("reward", {}) or {}
        shipped_sha = ((final.get("candidate") or {}).get("id") or "")

        traj = Trajectory(
            ref=ref,
            reward=(final.get("rewards") or {}).get("reward"),
            baseline_reward=(final.get("baseline_rewards") or {}).get("reward"),
            error_rate=metrics.get("error_rate"),
            total_tokens=metrics.get("inference_total_tokens"),
        )

        archive = final_path.parent / "session.tar.gz"
        if not archive.is_file():
            return traj
        root = session_mod.unpack(archive, self.cache)
        if root is None:
            return traj

        repo_dir = session_mod.find_repo(root)
        if repo_dir is not None:
            traj.candidates = self._candidates(CandidateRepo(repo_dir), shipped_sha)
        traj.evaluations = self._evaluations(session_mod.read_evaluations(root))
        return traj

    def _candidates(self, repo: CandidateRepo, shipped_sha: str) -> list[Candidate]:
        out: list[Candidate] = []
        for position, (sha, subject, body) in enumerate(repo.log()):
            stats = repo.numstat(f"{sha}^", sha) if position else {}
            out.append(
                Candidate(
                    sha=sha,
                    parent_sha=repo.parent(sha),
                    position=position,
                    subject=subject,
                    body=body[:4000],
                    files=repo.files(sha),
                    insertions=sum(a for a, _ in stats.values()),
                    deletions=sum(r for _, r in stats.values()),
                    tree_sha=repo.tree_sha(sha),
                    is_seed=position == 0,
                    is_shipped=bool(shipped_sha) and sha.startswith(shipped_sha[:12]),
                )
            )
        return out

    @staticmethod
    def _evaluations(records: list[dict]) -> list[EvalRecord]:
        out: list[EvalRecord] = []
        for doc in records:
            request = doc.get("request") or {}
            report = doc.get("report") or {}
            eval_set = request.get("evaluation_set") or {}
            metrics = report.get("metrics") or {}
            sha = ((request.get("candidate") or {}).get("id") or "")
            partition = eval_set.get("partition")
            score = metrics.get("score")
            if not (sha and partition and score is not None):
                continue
            out.append(
                EvalRecord(
                    candidate_sha=sha,
                    partition=partition,
                    score=score,
                    error_rate=metrics.get("error_rate"),
                    selection_kind=(eval_set.get("selection") or {}).get("kind"),
                    n_attempts=(request.get("limits") or {}).get("n_attempts"),
                    started_at=report.get("started_at"),
                    finished_at=report.get("finished_at"),
                )
            )
        return out


def build(cache: Cache) -> HarborAdapter:
    return register(HarborAdapter(cache))
