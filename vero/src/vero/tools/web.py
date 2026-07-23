from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field

from async_lru import alru_cache

from vero.tools.registry import ToolRegistry

SCRAPING_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"


def _fetch_and_parse_webpage_trafilatura(url: str) -> str | None:
    """Fetch and parse a webpage using trafilatura."""
    import trafilatura
    from trafilatura.settings import use_config

    trafilatura_config = use_config()
    trafilatura_config["DEFAULT"]["USER_AGENT"] = SCRAPING_AGENT

    downloaded = trafilatura.fetch_url(url, config=trafilatura_config, no_ssl=True)
    if downloaded is None:
        return None

    return trafilatura.extract(downloaded)


async def _fetch_and_parse_webpage_beautifulsoup(url: str) -> str | None:
    import httpx
    from bs4 import BeautifulSoup

    try:
        async with httpx.AsyncClient(verify=False, follow_redirects=True, timeout=30.0) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": SCRAPING_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )
            response.raise_for_status()
            downloaded = response.text
            soup = BeautifulSoup(downloaded, "lxml")
            for script in soup(["script", "style"]):
                script.extract()
            content = soup.get_text(separator="\n", strip=True)
            return content
    except (httpx.HTTPError, Exception):
        return None


async def _fetch_and_parse_pdf(url: str) -> str | None:
    """Fetch and parse a PDF document."""
    import httpx
    from pypdf import PdfReader

    try:
        async with httpx.AsyncClient(verify=False, follow_redirects=True, timeout=30.0) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": SCRAPING_AGENT,
                    "Accept": "application/pdf",
                },
            )
            response.raise_for_status()

            # Check if content is actually a PDF
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                return None

            # Parse PDF
            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)

            # Extract text from all pages
            text_content = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)

            return "\n\n".join(text_content) if text_content else None

    except Exception:
        return None


@alru_cache(maxsize=128)
async def _fetch_and_parse_webpage(url: str) -> str | None:
    """Fetch and parse a webpage or PDF using multiple methods with fallbacks."""
    # Check if URL likely points to a PDF
    if url.lower().endswith(".pdf"):
        content = await _fetch_and_parse_pdf(url)
        if content is not None:
            return content

    # Try HTML parsing first
    content = _fetch_and_parse_webpage_trafilatura(url)
    if content is not None:
        return content

    content = await _fetch_and_parse_webpage_beautifulsoup(url)
    if content is not None:
        return content

    # If HTML parsing failed, try PDF as fallback (might be PDF with non-.pdf extension)
    content = await _fetch_and_parse_pdf(url)
    if content is not None:
        return content

    return None


@dataclass
class WebFetch:
    """Fetches and extracts text content from a webpage given a URL."""

    exclude_tools: list[str] = field(default_factory=list)
    max_chars: int = 50_000

    async def __call__(self, url: str, offset: int = 0, max_chars: int = 50_000) -> str:
        """
        Fetches the content of a webpage and returns the extracted text.

        Args:
            url: The URL of the webpage to fetch

        Returns:
            Extracted text content from the webpage
        """

        if offset < 0:
            raise ValueError("offset must be greater than or equal to 0.")

        if max_chars > self.max_chars:
            raise ValueError(
                f"max_chars must be less than or equal to {self.max_chars}. Use pagination to fetch more content."
            )

        content = await _fetch_and_parse_webpage(url)

        if content is None:
            raise RuntimeError(f"Failed to fetch content from {url}")

        if offset > len(content):
            raise ValueError(f"Offset was greater than the length of the content {len(content)}")

        # Apply offset and max_chars
        content = content[offset : offset + max_chars]

        return content


def _contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return any("\u4E00" <= char <= "\u9FFF" for char in text)


@dataclass
class WebSearch:
    """Searches the web for information using the Serper API (google.serper.dev)."""

    exclude_tools: list[str] = field(default_factory=list)
    max_chars_per_result: int = 10_000
    max_results: int = 10
    max_retries: int = 3
    fetch_full_content: bool = True
    base_url: str = "https://google.serper.dev/search"

    def __post_init__(self):
        api_key = os.getenv("SERPER_KEY_ID") or os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError(
                "`SERPER_KEY_ID` or `SERPER_API_KEY` must be set as an environment variable to use `WebSearch` tool."
            )

    async def __call__(self, query: str) -> str:  # noqa: C901
        """
        Searches the web for information and returns a list of results with URL, title, and content.

        If fetch_full_content is True (default), fetches and parses full webpage content.
        Otherwise, returns only snippets from the search results.

        Args:
            query: The query string to search for
        Returns:
            JSON string containing search results (URL, Title, Content/Snippet)
        """
        api_key = os.getenv("SERPER_KEY_ID") or os.getenv("SERPER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "`SERPER_KEY_ID` or `SERPER_API_KEY` must be set as an environment variable to use `WebSearch` tool."
            )

        # Configure payload based on language
        if _contains_chinese(query):
            payload = {
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn",
                "num": min(self.max_results, 10),
            }
        else:
            payload = {
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en",
                "num": min(self.max_results, 10),
            }

        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }

        import httpx

        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        self.base_url,
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()

                results = response.json()

                if "organic" not in results:
                    return json.dumps(
                        {
                            "error": f"No results found for query: '{query}'",
                            "suggestion": "Try a less specific query",
                            "results": [],
                        }
                    )

                # Parse and optionally fetch full content
                parsed_results = []
                for item in results["organic"][: self.max_results]:
                    result_item = {
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                    }

                    if self.fetch_full_content:
                        # Fetch full page content
                        content = await _fetch_and_parse_webpage(result_item["url"])
                        if content is not None:
                            if len(content) > self.max_chars_per_result:
                                content = content[: self.max_chars_per_result] + "... (truncated)"
                            result_item["content"] = content
                        else:
                            # Fall back to snippet if full content fetch fails
                            result_item["content"] = item.get("snippet", "")
                    else:
                        # Just use the snippet
                        result_item["content"] = item.get("snippet", "")

                    # Add optional metadata
                    if "date" in item:
                        result_item["date"] = item["date"]
                    if "source" in item:
                        result_item["source"] = item["source"]

                    parsed_results.append(result_item)

                return json.dumps(parsed_results, indent=4)

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    continue
            except httpx.HTTPStatusError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    continue
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    continue

        # All retries failed
        return json.dumps(
            {
                "error": f"Search failed after {self.max_retries} attempts: {str(last_error)}",
                "results": [],
            }
        )


# These tools have optional scraping dependencies, so register them only when
# the web module is imported.
ToolRegistry.register(WebFetch)
ToolRegistry.register(WebSearch)
