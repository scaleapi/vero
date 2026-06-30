"""Tests for resolve_split_access (vero.core.dataset.base): fail-closed tier resolution."""

from vero.core.dataset import SplitAccess, SplitAccessLevel, resolve_split_access


def test_resolve_listed_tiers():
    accesses = [
        SplitAccess.viewable("train"),
        SplitAccess.non_viewable("validation"),
        SplitAccess.no_access("test"),
    ]
    assert resolve_split_access("train", accesses) == SplitAccessLevel.viewable
    assert resolve_split_access("validation", accesses) == SplitAccessLevel.non_viewable
    assert resolve_split_access("test", accesses) == SplitAccessLevel.no_access


def test_resolve_unlisted_split_fails_closed():
    accesses = [SplitAccess.viewable("train")]
    assert resolve_split_access("holdout", accesses) == SplitAccessLevel.no_access


def test_resolve_empty_accesses_fails_closed():
    assert resolve_split_access("anything", []) == SplitAccessLevel.no_access
