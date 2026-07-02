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


class TestReserveFlushOffLoop:
    @pytest.mark.asyncio
    async def test_reserve_flushes_durably(self, tmp_path):
        path = tmp_path / "ledger.json"
        led = _ledger(persist_path=path)
        await led.reserve("ds1", "dev", 25)
        data = json.loads(path.read_text())
        entry = next(e for e in data if e["split"] == "dev")
        assert entry["remaining_sample_budget"] == 75
        assert entry["remaining_run_budget"] == 2

    @pytest.mark.asyncio
    async def test_reserve_flush_goes_through_to_thread(self, tmp_path, monkeypatch):
        import asyncio as _asyncio

        path = tmp_path / "ledger.json"
        led = _ledger(persist_path=path)
        seen = []
        real_to_thread = _asyncio.to_thread

        async def _spy(func, *args, **kwargs):
            seen.append(func)
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr("vero.core.budget.asyncio.to_thread", _spy)
        await led.reserve("ds1", "dev", 10)
        # the durable flush ran off the event loop
        assert led._flush in seen
        # and it actually persisted
        data = json.loads(path.read_text())
        entry = next(e for e in data if e["split"] == "dev")
        assert entry["remaining_sample_budget"] == 90

    @pytest.mark.asyncio
    async def test_reserve_no_to_thread_when_in_memory(self, monkeypatch):
        import asyncio as _asyncio

        led = _ledger()  # persist_path=None
        called = []
        real_to_thread = _asyncio.to_thread

        async def _spy(func, *args, **kwargs):
            called.append(func)
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr("vero.core.budget.asyncio.to_thread", _spy)
        await led.reserve("ds1", "dev", 10)
        assert called == []  # no flush path when in-memory
        assert led.get("ds1", "dev").remaining_sample_budget == 90

    @pytest.mark.asyncio
    async def test_concurrent_reserves_do_not_overspend(self, tmp_path):
        import asyncio as _asyncio

        path = tmp_path / "ledger.json"
        # run budget of 3: only 3 of 10 concurrent reserves may succeed
        led = BudgetLedger(
            [SplitBudget(split="dev", dataset_id="ds1", total_run_budget=3)],
            persist_path=path,
        )

        async def _try():
            try:
                await led.reserve("ds1", "dev", 0)
                return True
            except ExperimentBudgetExceeded:
                return False

        results = await _asyncio.gather(*[_try() for _ in range(10)])
        assert sum(results) == 3
        assert led.get("ds1", "dev").remaining_run_budget == 0
