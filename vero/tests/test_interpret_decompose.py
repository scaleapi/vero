"""Decomposition against a real candidate repository, not a mocked one.

The bugs these cover are all bugs of *ordering and attribution* — which commit is the
seed, which edits are ancestors of which, where a deletion lands — and none of them
reproduce against a stubbed repo, because the stub is where the wrong assumption gets
written down in the first place.
"""

from __future__ import annotations

import subprocess
import tarfile

import pytest

from vero.interpret.artifacts.harbor.repo import CandidateRepo
from vero.interpret.artifacts.harbor.session import _contained
from vero.interpret.edits.decompose import decompose
from vero.interpret.edits.provenance import provenance_of
from vero.interpret.labeling.taxonomy import Provenance
from vero.interpret.models import Candidate, SymbolKind

SEED = '''\
TIMEOUT = 30


def parse(text):
    return text.strip()


def audit(result):
    assert result
    return result


class Agent:
    def run(self, cmd):
        return cmd
'''


def _git(repo, *args, when: str | None = None):
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    if when:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = when
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                         text=True, env={**env, "PATH": "/usr/bin:/bin:/usr/local/bin"})
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def _commit(repo, source, message, when):
    (repo / "agent.py").write_text(source)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message, when=when)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def chain(tmp_path):
    """seed -> first -> second, committed out of date order.

    `second` carries an *earlier* timestamp than its own parent, which is what the
    optimizer produces when it reaches back and retries: sorting by commit date then
    puts a child ahead of its parent.
    """
    repo = tmp_path / "work"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    seed = _commit(repo, SEED, "seed", "2026-01-01T00:00:00Z")
    first = _commit(repo, SEED.replace("return text.strip()", "return text.strip().lower()"),
                    "lowercase the parse", "2026-01-09T00:00:00Z")
    second = _commit(repo, SEED.replace("return text.strip()", "return text.strip().lower() or ''")
                     .replace("TIMEOUT = 30", "TIMEOUT = 90"),
                     "empty-string fallback and a longer timeout",
                     "2026-01-05T00:00:00Z")
    return CandidateRepo(repo / ".git"), seed, first, second


def _candidate(repo, sha, position, parent):
    return Candidate(sha=sha, parent_sha=parent, position=position, subject="s", body="",
                     files=repo.files(sha), tree_sha=repo.tree_sha(sha),
                     is_seed=parent is None)


# -- ordering -----------------------------------------------------------------


def test_log_puts_every_parent_before_its_child(chain):
    """Date order would not: `second` is older than the `first` it descends from."""
    repo, seed, first, second = chain
    order = [sha for sha, _, _ in repo.log()]
    assert order.index(seed) < order.index(first) < order.index(second)


def test_seed_is_the_parentless_commit_not_the_first_row(chain):
    repo, seed, _, _ = chain
    assert repo.parent(seed) is None
    for sha, _, _ in repo.log():
        assert (sha == seed) == (repo.parent(sha) is None)


# -- deletion attribution -----------------------------------------------------


def test_deleted_function_lands_on_the_module_row_with_its_removals(tmp_path):
    """A deletion-only hunk must not be attributed to the symbol that survives it."""
    repo_dir = tmp_path / "del"
    repo_dir.mkdir()
    _git(repo_dir, "init", "-q", "-b", "main")
    seed = _commit(repo_dir, SEED, "seed", "2026-01-01T00:00:00Z")
    without_audit = SEED.replace("def audit(result):\n    assert result\n    return result\n\n\n", "")
    child = _commit(repo_dir, without_audit, "drop the audit pass", "2026-01-02T00:00:00Z")

    repo = CandidateRepo(repo_dir / ".git")
    edits = decompose(repo, "cell", _candidate(repo, child, 1, seed), seed_sha=seed)

    module = [e for e in edits if e.symbol == "<module>"]
    assert len(module) == 1, [e.symbol for e in edits]
    assert module[0].removed > 0
    assert module[0].symbol_kind is SymbolKind.MODULE
    # The surviving class must not have absorbed the removal.
    assert not [e for e in edits if e.symbol == "Agent" and e.removed]
    # ...and the deletion is still visible in what the labeller will read.
    assert "-    assert result" in module[0].diff


# -- provenance ---------------------------------------------------------------


def test_first_change_to_seed_code_is_seed_provenance(chain):
    repo, seed, first, _ = chain
    edits = decompose(repo, "cell", _candidate(repo, first, 1, seed), seed_sha=seed)
    parse = next(e for e in edits if e.symbol == "parse")
    assert parse.provenance == Provenance.SEED.value


def test_rewriting_the_optimizers_own_edit_is_own_provenance(chain):
    repo, seed, first, second = chain
    edits = decompose(repo, "cell", _candidate(repo, second, 2, first), seed_sha=seed)
    parse = next(e for e in edits if e.symbol == "parse")
    assert parse.provenance == Provenance.OWN.value
    # A symbol the optimizer has not touched before is still the seed's.
    timeout = next(e for e in edits if e.symbol == "TIMEOUT")
    assert timeout.provenance == Provenance.SEED.value


def test_provenance_of_matches_the_in_memory_comparison(chain):
    repo, seed, first, second = chain
    assert provenance_of(repo, seed, first, "agent.py", "parse") is Provenance.OWN
    assert provenance_of(repo, seed, seed, "agent.py", "parse") is Provenance.SEED
    assert provenance_of(repo, "", first, "agent.py", "parse") is Provenance.UNKNOWN


# -- archive containment ------------------------------------------------------


@pytest.mark.parametrize("name", [
    "../../candidates/repository.git/config",
    "/etc/candidates/repository.git/config",
    "a/../../../candidates/repository.git/HEAD",
])
def test_escaping_members_are_refused(tmp_path, name):
    """The name filter alone matches these: they contain the wanted prefix."""
    member = tarfile.TarInfo(name)
    member.type = tarfile.REGTYPE
    assert not _contained(member, tmp_path / "dest")


def test_ordinary_member_is_kept(tmp_path):
    member = tarfile.TarInfo("session/candidates/repository.git/HEAD")
    member.type = tarfile.REGTYPE
    assert _contained(member, tmp_path / "dest")


def test_links_are_refused(tmp_path):
    """`filter="data"` drops these on 3.12+; 3.11 has to be told."""
    member = tarfile.TarInfo("session/candidates/repository.git/HEAD")
    member.type = tarfile.SYMTYPE
    member.linkname = "/etc/passwd"
    assert not _contained(member, tmp_path / "dest")
