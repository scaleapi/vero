"""Tests for web tools (WebSearch, WebFetch)."""

import json
import os

import pytest

# Check if Serper API key is available
SERPER_API_KEY_AVAILABLE = bool(os.getenv("SERPER_KEY_ID") or os.getenv("SERPER_API_KEY"))

requires_serper_key = pytest.mark.skipif(
    not SERPER_API_KEY_AVAILABLE,
    reason="SERPER_KEY_ID or SERPER_API_KEY not set",
)


class TestWebSearch:
    """Tests for WebSearch tool."""

    def test_init_without_api_key(self, monkeypatch):
        """Test that WebSearch raises ValueError if no API key is set."""
        from vero.tools.web import WebSearch

        # Remove both possible API key env vars
        monkeypatch.delenv("SERPER_KEY_ID", raising=False)
        monkeypatch.delenv("SERPER_API_KEY", raising=False)

        with pytest.raises(ValueError, match="SERPER_KEY_ID.*SERPER_API_KEY"):
            WebSearch()

    @requires_serper_key
    def test_init_with_api_key(self):
        """Test that WebSearch initializes successfully with API key."""
        from vero.tools.web import WebSearch

        search = WebSearch()
        assert search.max_results == 10
        assert search.max_retries == 3

    @requires_serper_key
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """Test that WebSearch returns valid results."""
        from vero.tools.web import WebSearch

        search = WebSearch(fetch_full_content=False)  # Faster - just snippets
        result = await search("Python programming language")

        # Parse the JSON result
        parsed = json.loads(result)

        # Should not have an error
        assert "error" not in parsed or parsed.get("results") is not None

        # Should be a list of results
        assert isinstance(parsed, list)
        assert len(parsed) > 0

        # Each result should have required fields
        for item in parsed:
            assert "title" in item
            assert "url" in item
            assert "content" in item

    @requires_serper_key
    @pytest.mark.asyncio
    async def test_search_with_full_content(self):
        """Test that WebSearch can fetch full page content."""
        from vero.tools.web import WebSearch

        search = WebSearch(fetch_full_content=True, max_results=2)
        result = await search("Python official website")

        parsed = json.loads(result)
        assert isinstance(parsed, list)

        # At least one result should have substantial content
        has_content = any(len(item.get("content", "")) > 100 for item in parsed)
        assert has_content, "Expected at least one result with fetched content"

    @requires_serper_key
    @pytest.mark.asyncio
    async def test_search_chinese_query(self):
        """Test that WebSearch handles Chinese queries."""
        from vero.tools.web import WebSearch

        search = WebSearch(fetch_full_content=False, max_results=3)
        result = await search("Python 编程语言")

        parsed = json.loads(result)
        assert isinstance(parsed, list)
        # Should return some results (might be empty depending on search)

    @requires_serper_key
    @pytest.mark.asyncio
    async def test_search_no_results_query(self):
        """Test that WebSearch handles queries with no results gracefully."""
        from vero.tools.web import WebSearch

        search = WebSearch(fetch_full_content=False)
        # Very unlikely to have results
        result = await search("xyzzy123456789abcdefghijklmnop")

        parsed = json.loads(result)
        # Should either be empty list or have error message
        assert isinstance(parsed, (list, dict))


class TestWebFetch:
    """Tests for WebFetch tool."""

    @pytest.mark.asyncio
    async def test_fetch_webpage(self):
        """Test fetching a simple webpage."""
        from vero.tools.web import WebFetch

        fetch = WebFetch()
        # Use a reliable, stable URL
        content = await fetch("https://example.com")

        assert isinstance(content, str)
        assert len(content) > 0
        assert "Example Domain" in content or "example" in content.lower()

    @pytest.mark.asyncio
    async def test_fetch_with_offset(self):
        """Test fetching with offset."""
        from vero.tools.web import WebFetch

        fetch = WebFetch()
        full_content = await fetch("https://example.com")
        offset_content = await fetch("https://example.com", offset=10)

        # The offset content should be the full content minus the first 10 chars
        assert offset_content == full_content[10:]
        assert len(offset_content) == len(full_content) - 10

    @pytest.mark.asyncio
    async def test_fetch_invalid_url(self):
        """Test that fetching invalid URL raises error."""
        from vero.tools.web import WebFetch

        fetch = WebFetch()

        with pytest.raises(RuntimeError, match="Failed to fetch"):
            await fetch("https://this-url-definitely-does-not-exist-12345.com")

    @pytest.mark.asyncio
    async def test_fetch_with_max_chars(self):
        """Test that max_chars truncates content."""
        from vero.tools.web import WebFetch

        fetch = WebFetch()
        content = await fetch("https://example.com", max_chars=50)

        assert len(content) <= 50
