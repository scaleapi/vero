"""Tests for Policy budget/split-access reconciliation (vero.policy)."""

import pytest

from vero.core.budget import SplitBudget
from vero.core.dataset import (
    SplitAccess,
    SplitAccessLevel,
    default_split_accesses,
    resolve_split_access,
)
from vero.policy import Policy


class _StubSession:
    dataset_id = "ds1"


def _policy(split_accesses, budget):
    # Construct without __init__/initialize; set only what the two methods read.
    # sessions_dir/dataset_cache/session_id are read-only properties; we never
    # set them. _validate_budget_splits reads them only inside a try/except
    # (load_dataset), so the resulting AttributeError is swallowed and the tier
    # check still runs.
    p = Policy.__new__(Policy)
    p.budget = budget
    p.split_accesses = list(split_accesses)
    p.session = _StubSession()
    return p


def test_ensure_auto_tiers_unlisted_budgeted_split_non_viewable():
    p = _policy([], [SplitBudget(split="train", dataset_id="ds1", total_run_budget=5)])
    p._ensure_budgeted_splits_tiered()
    assert resolve_split_access("train", p.split_accesses) == SplitAccessLevel.non_viewable


def test_ensure_then_validate_passes_for_train_budget_default():
    p = _policy(
        list(default_split_accesses),
        [SplitBudget(split="train", dataset_id="ds1", total_run_budget=5)],
    )
    p._ensure_budgeted_splits_tiered()
    p._validate_budget_splits()  # no raise


def test_validate_rejects_explicit_viewable_budgeted_split():
    p = _policy(
        [SplitAccess.viewable("dev")],
        [SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=10)],
    )
    with pytest.raises(ValueError, match="non_viewable"):
        p._validate_budget_splits()


def test_validate_rejects_explicit_no_access_budgeted_split():
    p = _policy(
        [SplitAccess.no_access("dev")],
        [SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=10)],
    )
    with pytest.raises(ValueError, match="non_viewable"):
        p._validate_budget_splits()


def test_validate_accepts_non_viewable_budgeted_split():
    p = _policy(
        [SplitAccess.non_viewable("dev")],
        [SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=10)],
    )
    p._validate_budget_splits()  # no raise
