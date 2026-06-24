"""Tests for vero.harbor.cli — the agent/verifier CLI clients (mocked httpx)."""

import json

from click.testing import CliRunner

from vero.harbor.cli import harbor


class _Resp:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data
        self.text = json.dumps(data)

    def json(self):
        return self._data


def _patch_httpx(monkeypatch, resp, capture):
    import httpx

    def fake_request(method, url, *, json=None, headers=None, timeout=None):
        capture.update(method=method, url=url, json=json, headers=headers)
        return resp

    monkeypatch.setattr(httpx, "request", fake_request)


def test_eval_posts_and_prints(monkeypatch):
    monkeypatch.setenv("VERO_EVAL_URL", "http://sidecar:8000")
    cap: dict = {}
    _patch_httpx(monkeypatch, _Resp(200, {"mean_score": 0.5}), cap)

    result = CliRunner().invoke(
        harbor, ["eval", "--dataset-id", "ds", "--split", "train", "--num-samples", "3"]
    )
    assert result.exit_code == 0
    assert cap["method"] == "POST" and cap["url"].endswith("/eval")
    assert cap["json"] == {"dataset_id": "ds", "split": "train", "num_samples": 3}
    assert json.loads(result.output)["mean_score"] == 0.5


def test_eval_error_status_raises(monkeypatch):
    monkeypatch.setenv("VERO_EVAL_URL", "http://sidecar:8000")
    _patch_httpx(monkeypatch, _Resp(429, {"error": "no budget"}), {})
    result = CliRunner().invoke(harbor, ["eval", "--dataset-id", "ds", "--split", "train"])
    assert result.exit_code != 0
    assert "429" in result.output


def test_eval_missing_url_errors():
    result = CliRunner(env={"VERO_EVAL_URL": ""}).invoke(
        harbor, ["eval", "--dataset-id", "ds", "--split", "train"]
    )
    assert result.exit_code != 0
    assert "VERO_EVAL_URL" in result.output


def test_status_get(monkeypatch):
    monkeypatch.setenv("VERO_EVAL_URL", "http://sidecar:8000")
    cap: dict = {}
    _patch_httpx(monkeypatch, _Resp(200, {"submit_enabled": True}), cap)
    result = CliRunner().invoke(harbor, ["status"])
    assert result.exit_code == 0 and cap["method"] == "GET" and cap["url"].endswith("/status")


def test_finalize_uses_token_and_writes_reward(monkeypatch, tmp_path):
    monkeypatch.setenv("VERO_EVAL_URL", "http://sidecar:8000")
    token_file = tmp_path / "tok"
    token_file.write_text("T0KEN")
    out = tmp_path / "reward.json"
    cap: dict = {}
    _patch_httpx(monkeypatch, _Resp(200, {"reward": 1.0}), cap)

    result = CliRunner().invoke(
        harbor, ["finalize", "--token-file", str(token_file), "--output", str(out)]
    )
    assert result.exit_code == 0
    assert cap["url"].endswith("/finalize")
    assert cap["headers"]["Authorization"] == "Bearer T0KEN"
    assert json.loads(out.read_text()) == {"reward": 1.0}
