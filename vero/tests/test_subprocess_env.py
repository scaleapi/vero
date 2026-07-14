"""Tests for subprocess environment building."""

from __future__ import annotations

import os

from vero.utils.subprocess_env import build_subprocess_env, load_env_file, apply_env_file


class TestBuildSubprocessEnv:
    def test_includes_system_defaults(self):
        env = build_subprocess_env()
        assert "PATH" in env
        assert "HOME" in env

    def test_forwards_extra_vars(self, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "test-key")
        env = build_subprocess_env(source=["MY_API_KEY"])
        assert env["MY_API_KEY"] == "test-key"

    def test_callable_spec(self, monkeypatch):
        monkeypatch.setenv("BASE_URL", "https://proxy.example.com/")
        env = build_subprocess_env(source=[
            ("BASE_URL", lambda: "https://proxy.example.com/v1"),
        ])
        assert env["BASE_URL"] == "https://proxy.example.com/v1"

    def test_callable_returning_none_excluded(self):
        env = build_subprocess_env(source=[
            ("MISSING", lambda: None),
        ])
        assert "MISSING" not in env

    def test_missing_string_var_excluded(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT", raising=False)
        env = build_subprocess_env(source=["NONEXISTENT"])
        assert "NONEXISTENT" not in env

    def test_no_extra_env_leakage(self, monkeypatch):
        monkeypatch.setenv("SECRET_THING", "should-not-leak")
        env = build_subprocess_env()
        assert "SECRET_THING" not in env

    def test_uv_index_forwarded_by_default(self, monkeypatch):
        monkeypatch.setenv("UV_INDEX", "https://index.example.com/")
        env = build_subprocess_env()
        assert env["UV_INDEX"] == "https://index.example.com/"

    def test_uv_cache_forwarded_by_default(self, monkeypatch):
        monkeypatch.setenv("UV_CACHE_DIR", "/tmp/uv-cache")
        env = build_subprocess_env()
        assert env["UV_CACHE_DIR"] == "/tmp/uv-cache"

    def test_mixed_string_and_callable(self, monkeypatch):
        monkeypatch.setenv("KEY_A", "value-a")
        env = build_subprocess_env(source=[
            "KEY_A",
            ("KEY_B", lambda: "computed-b"),
        ])
        assert env["KEY_A"] == "value-a"
        assert env["KEY_B"] == "computed-b"


class TestEnvFile:
    def test_load_env_file(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("API_KEY=secret123\nBASE_URL=https://example.com\n")
        env = load_env_file(env_file)
        assert env == {"API_KEY": "secret123", "BASE_URL": "https://example.com"}

    def test_load_env_file_strips_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('API_KEY="secret123"\nOTHER=\'value\'\n')
        env = load_env_file(env_file)
        assert env["API_KEY"] == "secret123"
        assert env["OTHER"] == "value"

    def test_load_env_file_skips_comments_and_blanks(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nKEY=val\n  # another comment\n")
        env = load_env_file(env_file)
        assert env == {"KEY": "val"}

    def test_build_subprocess_env_from_path(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_VAR=from_file\n")
        env = build_subprocess_env(source=env_file)
        assert env["MY_VAR"] == "from_file"
        assert "PATH" in env  # system defaults still present

    def test_build_subprocess_env_from_str_path(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_VAR=from_file\n")
        env = build_subprocess_env(source=str(env_file))
        assert env["MY_VAR"] == "from_file"

    def test_apply_env_file_does_not_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EXISTING", "original")
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING=overwritten\nNEW_VAR=new\n")
        apply_env_file(env_file)
        assert os.environ["EXISTING"] == "original"  # not overwritten
        assert os.environ["NEW_VAR"] == "new"

    def test_load_env_file_not_found(self, tmp_path):
        import pytest
        with pytest.raises(FileNotFoundError):
            load_env_file(tmp_path / "nonexistent.env")
