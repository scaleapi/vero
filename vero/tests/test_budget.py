"""Tests for BudgetLedger (vero.core.budget)."""

import json

import pytest

from vero.core.budget import BudgetLedger, SplitBudget
from vero.exceptions import ExperimentBudgetExceeded, InvalidSplitError


def _ledger(**kwargs):
    return BudgetLedger(
        [
            SplitBudget(
                split="dev", dataset_id="ds1", total_sample_budget=100, total_run_budget=3
            )
        ],
        **kwargs,
    )


class TestAllowlist:
    def test_validate_allows_configured_pair(self):
        _ledger().validate("ds1", "dev")  # no raise

    def test_validate_rejects_unknown_pair(self):
        with pytest.raises(InvalidSplitError):
            _ledger().validate("ds1", "test")
        with pytest.raises(InvalidSplitError):
            _ledger().validate("other", "dev")


class TestCheck:
    def test_check_passes_within_budget(self):
        _ledger().check("ds1", "dev", 50)

    def test_check_rejects_over_sample_budget(self):
        with pytest.raises(ExperimentBudgetExceeded):
            _ledger().check("ds1", "dev", 101)

    def test_check_rejects_no_runs_left(self):
        led = BudgetLedger([SplitBudget(split="dev", dataset_id="ds1", total_run_budget=1)])
        led.record("ds1", "dev", 0)  # consume the one run
        with pytest.raises(ExperimentBudgetExceeded):
            led.check("ds1", "dev", 0)

    def test_check_rejects_over_per_run(self):
        led = BudgetLedger(
            [SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=100, max_samples_per_run=10)]
        )
        with pytest.raises(ExperimentBudgetExceeded):
            led.check("ds1", "dev", 11)


class TestRecord:
    def test_record_decrements(self):
        led = _ledger()
        b = led.record("ds1", "dev", 30)
        assert b.remaining_sample_budget == 70
        assert b.remaining_run_budget == 2


class TestReserve:
    @pytest.mark.asyncio
    async def test_reserve_checks_then_decrements(self):
        led = _ledger()
        b = await led.reserve("ds1", "dev", 40)
        assert b.remaining_sample_budget == 60
        assert b.remaining_run_budget == 2

    @pytest.mark.asyncio
    async def test_reserve_rejects_without_decrementing(self):
        led = _ledger()
        with pytest.raises(ExperimentBudgetExceeded):
            await led.reserve("ds1", "dev", 101)
        # rejected request costs nothing
        assert led.get("ds1", "dev").remaining_sample_budget == 100
        assert led.get("ds1", "dev").remaining_run_budget == 3


class TestPersistence:
    def test_flush_writes_durable_json(self, tmp_path):
        path = tmp_path / "ledger.json"
        led = _ledger(persist_path=path)
        led.record("ds1", "dev", 25)
        data = json.loads(path.read_text())
        entry = next(e for e in data if e["split"] == "dev" and e["dataset_id"] == "ds1")
        assert entry["remaining_sample_budget"] == 75
        assert entry["remaining_run_budget"] == 2

    def test_no_flush_when_in_memory(self, tmp_path):
        led = _ledger()  # persist_path=None
        led.record("ds1", "dev", 25)  # no file written, no error
        assert not list(tmp_path.iterdir())
