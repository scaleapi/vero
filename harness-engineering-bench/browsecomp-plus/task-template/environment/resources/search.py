#!/usr/bin/env python3
"""Command-line access to the pinned BrowseComp-Plus BM25 index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pyserini.search.lucene import LuceneSearcher

INDEX = Path("/opt/browsecomp/indexes/bm25")
DEFAULT_SNIPPET_WORDS = 512


def _searcher() -> LuceneSearcher:
    if not INDEX.is_dir():
        raise RuntimeError(f"BrowseComp-Plus index is missing: {INDEX}")
    return LuceneSearcher(str(INDEX))


def _raw_document(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    return {"docid": str(value.get("id") or ""), "text": str(value["contents"])}


def search(query: str, k: int, snippet_words: int) -> list[dict[str, Any]]:
    engine = _searcher()
    results: list[dict[str, Any]] = []
    for hit in engine.search(query, k):
        document = _raw_document(hit.lucene_document.get("raw"))
        words = document["text"].split()
        snippet = " ".join(words[:snippet_words])
        results.append({"docid": hit.docid, "score": hit.score, "snippet": snippet})
    return results


def get_document(docid: str) -> dict[str, Any]:
    engine = _searcher()
    document = engine.doc(docid)
    if document is None:
        return {"error": f"document {docid!r} was not found"}
    value = _raw_document(document.raw())
    value["docid"] = docid
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--k", type=int, default=5, choices=range(1, 21))
    search_parser.add_argument(
        "--snippet-words", type=int, default=DEFAULT_SNIPPET_WORDS
    )
    document_parser = subparsers.add_parser("get-document")
    document_parser.add_argument("--docid", required=True)
    args = parser.parse_args()

    if args.command == "search":
        result = search(args.query, args.k, args.snippet_words)
    else:
        result = get_document(args.docid)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
