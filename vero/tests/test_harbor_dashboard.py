"""Tests for vero.harbor.dashboard: the pure status-assembly layer.

The docker/HTTP collectors are thin shims; everything that decides what the
operator sees (key derivation, phase, verdicts, the meta/live/final join) is
pure and covered here without docker.
"""

import json

from vero.harbor.dashboard import (
    derive_phase,
    experiment_key,
    merge_status,
    normalize_rounds,
    scan_finals,
    verdict,
)


class TestExperimentKey:
    def test_sidecar_name_maps_to_task_key(self):
        assert experiment_key("gaia-exp11-task__nxkmtjb-eval-sidecar-1") == "gaia-exp11-task"

    def test_non_sidecar_containers_ignored(self):
        assert experiment_key("gaia-exp11-task__nxkmtjb-main-1") is None
        assert experiment_key("k8s_coredns_whatever") is None

    def test_sidecar_without_run_suffix_still_keys(self):
        assert experiment_key("plain-task-eval-sidecar-1") == "plain-task"


class TestPhase:
    def test_container_without_rounds_is_building(self):
        assert derive_phase(has_container=True, n_rounds=0, final=None) == "building"

    def test_container_with_rounds_is_running(self):
        assert derive_phase(has_container=True, n_rounds=3, final=None) == "running"

    def test_reward_json_wins_over_everything(self):
        assert derive_phase(has_container=False, n_rounds=0, final={"accuracy": 0.4}) == "done"
        assert derive_phase(has_container=True, n_rounds=9, final={"accuracy": 0.4}) == "done"

    def test_no_container_no_final_is_gone(self):
        # A crashed launch must be conspicuous, not silently absent.
        assert derive_phase(has_container=False, n_rounds=0, final=None) == "gone"


class TestVerdict:
    def test_win_when_above_bar(self):
        assert verdict({"accuracy": 0.6}, {"win_bar": 0.41}) == "WIN"

    def test_miss_when_at_or_below_bar(self):
        assert verdict({"accuracy": 0.40}, {"win_bar": 0.41}) == "MISS"
        assert verdict({"accuracy": 0.41}, {"win_bar": 0.41}) == "MISS"  # bar is strict

    def test_no_bars_no_verdict(self):
        assert verdict({"accuracy": 0.6}, None) is None
        assert verdict(None, {"win_bar": 0.41}) is None


class TestNormalizeRounds:
    def test_rows_truncate_commit_and_carry_scores(self):
        payload = {"experiments": [
            {"commit": "d5a5facb994912345678", "split": "train",
             "mean_score": 0.5555, "error_rate": 0.0, "created_at": "t1"},
            {"commit": None, "split": None, "mean_score": None, "error_rate": None},
        ]}
        rows = normalize_rounds(payload)
        assert rows[0]["commit"] == "d5a5facb99"
        assert rows[0]["score"] == 0.5555
        assert rows[1]["commit"] == ""  # tolerant of null ledger rows
        assert normalize_rounds(None) == []


class TestMergeStatus:
    META = {
        "experiments": [
            {"key": "gaia-exp11-task", "title": "Exp11: floor dogfood",
             "question": "Does the floor revert harm?",
             "bars": {"baseline": 0.317, "win_bar": 0.454}},
            {"key": "gaia-exp12-task", "title": "Exp12: queued"},
        ],
        "history": [{"exp": "7", "question": "positive control", "final": "0.6", "verdict": "WIN"}],
    }

    def test_meta_live_and_final_join_by_key(self):
        sidecars = {"gaia-exp11-task": {
            "experiments": {"experiments": [
                {"commit": "abc1234500", "split": "train", "mean_score": 0.25, "error_rate": 0.0}]},
            "status": {"splits": [{"split": "train", "remaining_run_budget": 5}]},
        }}
        out = merge_status(meta=self.META, sidecars=sidecars, finals={})
        e11 = next(e for e in out["experiments"] if e["key"] == "gaia-exp11-task")
        assert e11["phase"] == "running"
        assert e11["rounds"][0]["score"] == 0.25
        assert e11["budget"] == [{"split": "train", "remaining_run_budget": 5}]
        # meta-only experiment shows as gone (conspicuous, not hidden)
        e12 = next(e for e in out["experiments"] if e["key"] == "gaia-exp12-task")
        assert e12["phase"] == "gone"
        assert out["history"] == self.META["history"]

    def test_unlisted_live_trial_still_appears(self):
        # A running trial the meta forgot must never be invisible.
        sidecars = {"gaia-expX-task": {"experiments": {"experiments": []}, "status": {}}}
        out = merge_status(meta=self.META, sidecars=sidecars, finals={})
        ex = next(e for e in out["experiments"] if e["key"] == "gaia-expX-task")
        assert ex["phase"] == "building"
        assert ex["title"] == "gaia-expX-task"  # falls back to the key

    def test_final_produces_done_and_verdict(self):
        finals = {"gaia-exp11-task": {"accuracy": 0.5}}
        out = merge_status(meta=self.META, sidecars={}, finals=finals)
        e11 = next(e for e in out["experiments"] if e["key"] == "gaia-exp11-task")
        assert e11["phase"] == "done"
        assert e11["final"] == {"accuracy": 0.5}
        assert e11["verdict"] == "WIN"  # 0.5 > 0.454


class TestScanFinals:
    def test_reward_json_keyed_by_trial_dir_task_name(self, tmp_path):
        trial = tmp_path / "jobs" / "2026-07-04__x" / "gaia-exp11-task__AbCdEf" / "verifier"
        trial.mkdir(parents=True)
        (trial / "reward.json").write_text(json.dumps({"accuracy": 0.4}))
        finals = scan_finals([tmp_path / "jobs"])
        assert finals == {"gaia-exp11-task": {"accuracy": 0.4}}

    def test_missing_dir_is_empty(self, tmp_path):
        assert scan_finals([tmp_path / "nope"]) == {}
