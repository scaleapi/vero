"""Tests for SessionLogger: event logging, general logging, console rendering."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from vero.logging import SessionLogger


class TestEventLog:
    def test_writes_per_turn_files(self, tmp_path: Path):
        logger = SessionLogger(session_dir=tmp_path)
        logger({"role": "assistant", "content": "hello"})
        logger({"role": "tool", "name": "read_file", "content": "file contents"})
        logger.close()

        trace_dir = tmp_path / "agent_trace"
        assert trace_dir.exists()
        files = sorted(trace_dir.glob("turn_*.json"))
        assert len(files) == 2

        event0 = json.loads(files[0].read_text())
        assert event0["turn"] == 0
        assert event0["role"] == "assistant"
        assert "ts" in event0

        event1 = json.loads(files[1].read_text())
        assert event1["turn"] == 1
        assert event1["role"] == "tool"

    def test_disabled(self, tmp_path: Path):
        logger = SessionLogger(session_dir=tmp_path, enable_event_log=False)
        logger({"role": "assistant", "content": "hello"})
        logger.close()
        assert not (tmp_path / "agent_trace").exists()

    def test_readable_before_close(self, tmp_path: Path):
        logger = SessionLogger(session_dir=tmp_path, enable_console=False)
        logger({"role": "assistant", "content": "hello"})
        # File should exist immediately (no buffering)
        path = tmp_path / "agent_trace" / "turn_0000.json"
        assert path.exists()
        content = json.loads(path.read_text())
        assert content["content"] == "hello"
        logger.close()

    def test_filenames_are_zero_padded(self, tmp_path: Path):
        logger = SessionLogger(session_dir=tmp_path, enable_console=False)
        for i in range(12):
            logger({"turn_index": i})
        logger.close()

        files = sorted((tmp_path / "agent_trace").glob("turn_*.json"))
        assert files[0].name == "turn_0000.json"
        assert files[11].name == "turn_0011.json"


class TestGeneralLog:
    def test_captures_python_logging(self, tmp_path: Path):
        session_logger = SessionLogger(session_dir=tmp_path, enable_console=False)
        test_logger = logging.getLogger("vero.test_session_logging")
        test_logger.info("test info message")
        test_logger.warning("test warning")
        session_logger.close()

        log_text = (tmp_path / "session.log").read_text()
        assert "test info message" in log_text
        assert "test warning" in log_text

    def test_disabled(self, tmp_path: Path):
        session_logger = SessionLogger(session_dir=tmp_path, enable_general_log=False)
        logging.getLogger("vero.test").info("should not appear")
        session_logger.close()
        assert not (tmp_path / "session.log").exists()

    def test_handler_removed_on_close(self, tmp_path: Path):
        root = logging.getLogger()
        initial_handlers = len(root.handlers)
        session_logger = SessionLogger(session_dir=tmp_path)
        assert len(root.handlers) == initial_handlers + 1
        session_logger.close()
        assert len(root.handlers) == initial_handlers


class TestConsoleRendering:
    def test_does_not_crash(self, tmp_path: Path):
        """Console rendering should not raise on valid events."""
        logger = SessionLogger(session_dir=tmp_path, enable_console=True)
        logger({"type": "run_item_stream_event", "content": "thinking..."})
        logger({"role": "assistant", "content": "4"})
        logger.close()

    def test_disabled(self, tmp_path: Path):
        """Should work fine with console disabled."""
        logger = SessionLogger(session_dir=tmp_path, enable_console=False)
        logger({"role": "assistant", "content": "hello"})
        logger.close()

    def test_compact_claude_events(self, tmp_path: Path):
        """Compact mode handles Claude SDK event format."""
        logger = SessionLogger(session_dir=tmp_path, enable_console=True, console_verbose=False)
        logger({"content": [{"text": "Let me analyze this."}]})
        logger({"content": [{"name": "Read", "input": {"file_path": "main.py"}}]})
        logger({"content": [{"tool_use_id": "123", "content": "file contents here"}]})
        logger({"content": [{"tool_use_id": "456", "content": "error", "is_error": True}]})
        logger({"content": [{"thinking": "I should check the traces first."}]})
        logger.close()

    def test_compact_oai_events(self, tmp_path: Path):
        """Compact mode handles OAI SDK event format."""
        logger = SessionLogger(session_dir=tmp_path, enable_console=True, console_verbose=False)
        logger({"role": "assistant", "content": "Hello"})
        logger({"role": "tool", "name": "read_file", "content": "file data"})
        logger({"type": "agent_updated_stream_event"})
        logger.close()

    def test_compact_unknown_events(self, tmp_path: Path):
        """Compact mode handles unknown event formats gracefully."""
        logger = SessionLogger(session_dir=tmp_path, enable_console=True, console_verbose=False)
        logger({"some_unknown_key": "value"})
        logger({})
        logger.close()


class TestContextManager:
    def test_context_manager(self, tmp_path: Path):
        with SessionLogger(session_dir=tmp_path, enable_console=False) as logger:
            logger({"role": "assistant", "content": "hello"})

        files = sorted((tmp_path / "agent_trace").glob("turn_*.json"))
        assert len(files) == 1
