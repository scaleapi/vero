import asyncio
from typing import Any

from bs4 import BeautifulSoup
from httpx import AsyncClient

# Semaphore to limit concurrent requests
_semaphore: asyncio.Semaphore | None = None
MAX_CONCURRENT_REQUESTS = 5

WIKI_API_URL = "http://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "wikipedia (https://github.com/goldsmith/Wikipedia/)"}


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the semaphore for the current event loop."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _semaphore


async def _wiki_request(params: dict[str, str], max_retries: int = 3) -> dict[str, Any]:
    """
    Make a request to the Wikipedia API using the given search parameters.

    Args:
        params (dict[str, str]): The search parameters to use.
        max_retries (int): Maximum number of retries on failure.

    Returns:
        dict[str, Any]: A parsed dict of the JSON response.
    """

    params["format"] = "json"
    if "action" not in params:
        params["action"] = "query"

    last_error = None
    for attempt in range(max_retries):
        async with _get_semaphore():
            try:
                async with AsyncClient(follow_redirects=True, headers=WIKI_HEADERS, timeout=30.0) as client:
                    r = await client.get(WIKI_API_URL, params=params)

                if not r.text:
                    raise Exception(f"Empty response from Wikipedia API (status={r.status_code})")

                return r.json()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff

    raise last_error


async def search(query: str, results: int = 10, suggestion: bool = False) -> tuple[list[str], str | None]:
    """
    Do a Wikipedia search for `query`.

    Args:
        query (str): The search query to use.
        results (int): The maximum number of results to return.
        suggestion (bool): If True, return results and suggestion (if any) in a tuple.

    Returns:
        tuple[list[str], str | None]: A tuple of the search results and suggestion (if any).
    """

    search_params = {"list": "search", "srprop": "", "srlimit": results, "limit": results, "srsearch": query}

    if suggestion:
        search_params["srinfo"] = "suggestion"

    raw_results = await _wiki_request(search_params)

    if "error" in raw_results:
        if raw_results["error"]["info"] in ("HTTP request timed out."):
            raise TimeoutError(query)
        else:
            raise Exception(raw_results["error"]["info"])

    search_results = [d["title"] for d in raw_results["query"]["search"]]

    if suggestion:
        if raw_results["query"].get("searchinfo"):
            return search_results, raw_results["query"]["searchinfo"]["suggestion"]
        else:
            return search_results, None

    return search_results


class WikipediaPage:
    """
    Contains data from a Wikipedia page.
    Uses property methods to filter data from the raw HTML.
    """

    def __init__(
        self, title: str | None = None, pageid: int | None = None, redirect: bool = True, original_title: str = ""
    ):
        if title is not None:
            self.title = title
            self.original_title = original_title or title
        elif pageid is not None:
            self.pageid = pageid
        else:
            raise ValueError("Either a title or a pageid must be specified")

        self.redirect = redirect
        self._loaded = False

    def __repr__(self):
        return f"WikipediaPage '{self.title}'"

    def __eq__(self, other: "WikipediaPage"):
        try:
            return self.pageid == other.pageid and self.title == other.title and self.url == other.url
        except Exception:
            return False

    async def __load(self, redirect: bool = True):
        """Load basic information from Wikipedia. Confirm that page exists and is not a disambiguation/redirect."""

        query_params = {
            "prop": "info|pageprops",
            "inprop": "url",
            "ppprop": "disambiguation",
            "redirects": "",
        }
        if not getattr(self, "pageid", None):
            query_params["titles"] = self.title
        else:
            query_params["pageids"] = self.pageid

        request = await _wiki_request(query_params)

        query = request["query"]
        pageid = list(query["pages"].keys())[0]
        page = query["pages"][pageid]

        # missing is present if the page is missing
        if "missing" in page:
            if hasattr(self, "title"):
                raise Exception(f"Page not found: {self.title}")
            else:
                raise Exception(f"Page not found: {self.pageid}")

        # same thing for redirect, except it shows up in query instead of page for
        # whatever silly reason
        elif "redirects" in query:
            if redirect:
                redirects = query["redirects"][0]

                if "normalized" in query:
                    normalized = query["normalized"][0]
                    assert normalized["from"] == self.title

                    from_title = normalized["to"]

                else:
                    from_title = self.title

                assert redirects["from"] == from_title

                # change the title and reload the whole object
                self.__init__(title=redirects["to"], redirect=redirect)
                await self._WikipediaPage__load(redirect=redirect)

            else:
                raise Exception(f"Page redirected: {getattr(self, 'title', page['title'])}")

        # since we only asked for disambiguation in ppprop,
        # if a pageprop is returned,
        # then the page must be a disambiguation page
        elif "pageprops" in page:
            query_params = {"prop": "revisions", "rvprop": "content", "rvparse": "", "rvlimit": 1}
            if hasattr(self, "pageid"):
                query_params["pageids"] = self.pageid
            else:
                query_params["titles"] = self.title
            request = await _wiki_request(query_params)
            html = request["query"]["pages"][pageid]["revisions"][0]["*"]

            lis = BeautifulSoup(html, "lxml").find_all("li")
            filtered_lis = [li for li in lis if "tocsection" not in "".join(li.get("class", []))]
            [li.a.get_text() for li in filtered_lis if li.a]

            raise Exception(f"Page is a disambiguation: {getattr(self, 'title', page['title'])}")

        else:
            self.pageid = pageid
            self.title = page["title"]
            self.url = page["fullurl"]

        self._loaded = True

    async def content(self):
        """Plain text content of the page, excluding images, tables, and other data."""

        if not self._loaded:
            await self.__load()

        if not getattr(self, "_content", False):
            query_params = {"prop": "extracts|revisions", "explaintext": "", "rvprop": "ids"}
            if getattr(self, "title", None) is not None:
                query_params["titles"] = self.title
            else:
                query_params["pageids"] = self.pageid
            request = await _wiki_request(query_params)
            self._content = request["query"]["pages"][self.pageid]["extract"]
            self._revision_id = request["query"]["pages"][self.pageid]["revisions"][0]["revid"]
            self._parent_id = request["query"]["pages"][self.pageid]["revisions"][0]["parentid"]

        return self._content


async def get_page(
    title: str | None = None, pageid: int | None = None, auto_suggest: bool = True, redirect: bool = True
) -> str:
    """
    Get a WikipediaPage object for the page with title `title` or the pageid `pageid` (mutually exclusive).

    Args
        title (str): The title of the page to load
        pageid (int): The numeric pageid of the page to load
        auto_suggest (bool): Let Wikipedia find a valid page title for the query
        redirect (bool): Allow redirection without raising RedirectError

    Returns:
        str: The content of the page
    """

    if title is not None:
        if auto_suggest:
            results, suggestion = await search(title, results=1, suggestion=True)
            try:
                title = suggestion or results[0]
            except IndexError:
                # if there is no suggestion or search results, the page doesn't exist
                raise Exception(f"Page not found: {title}") from None
        wiki_page = WikipediaPage(title=title, redirect=redirect)
        return await wiki_page.content()
    elif pageid is not None:
        wiki_page = WikipediaPage(pageid=pageid)
        return await wiki_page.content()
    else:
        raise Exception("Either a title or a pageid must be specified")
