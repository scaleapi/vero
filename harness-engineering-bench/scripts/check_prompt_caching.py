#!/usr/bin/env python3
"""Check whether the proxy serves cached prompt prefixes for a target model.

Our per-cell cost estimates weight rates at ~90% cache-read, so if a target's
prompts are not actually being cached the real cost is several times the
estimate. This sends the same long prompt twice and reads the reported cache
hit:

    python3 harness-engineering-bench/scripts/check_prompt_caching.py \
        vero/heb.secrets.env --model xai/grok-build-0.1

Measured on xai/grok-build-0.1 (2026-07-31): request 1 cached=128, request 2
cached=5,184 of a 5,192-token prompt, with and without a tool block. Caching
works and arrives in `prompt_tokens_details.cached_tokens`, which is the field
vero's gateway metering already reads.

Two things this encodes that a naive version gets wrong:

- **A cold request reports a constant 128 cached tokens.** Testing
  `cached_tokens > 0` passes on a cold request, so compare against the prompt
  size instead.
- **A warm request can miss** (cache write not landed, or another replica), so
  a single miss is not evidence of no caching. Hence the retry.

Prints no secret material: the key appears as a short fingerprint only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Comfortably past the ~1024-token minimum prefix providers commonly require.
PREFIX_TOKENS = 5000
# A pass recovers most of the prompt. Providers cache in blocks and leave a
# remainder, so the bar is well short of 1.0 but far above the 128-token floor.
HIT_FLOOR = 0.5
ATTEMPTS = 3
RETRY_DELAY_SEC = 3.0

TOOL_BLOCK = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command in the container.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def load_env(path: Path) -> dict[str, str]:
    """Parse a `*.secrets.env` file. Mirrors check_keys.py."""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^export\s+", "", line)
        name, sep, value = line.partition("=")
        if sep:
            values[name.strip()] = value.strip().strip("\"'")
    return values


def ask(base_url: str, api_key: str, model: str, prefix: str, tools: bool):
    """One request. Returns (prompt_tokens, cached_tokens) or raises."""
    body: dict = {
        "model": model,
        "max_tokens": 8,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": "hi"},
        ],
    }
    if tools:
        body["tools"] = TOOL_BLOCK
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        usage = json.loads(response.read()).get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return (
        int(usage.get("prompt_tokens") or 0),
        int(details.get("cached_tokens") or 0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("env_file", type=Path,
                        help="a *.secrets.env with OPENAI_API_KEY / OPENAI_BASE_URL")
    parser.add_argument("--model", default="xai/grok-build-0.1")
    parser.add_argument("--no-tools", action="store_true",
                        help="omit the tool block; on by default because our "
                             "agents send one and it is part of the prefix")
    args = parser.parse_args()

    if not args.env_file.is_file():
        print(f"no such env file: {args.env_file}")
        return 2
    env = load_env(args.env_file)
    api_key, base_url = env.get("OPENAI_API_KEY"), env.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        print(f"{args.env_file} is missing OPENAI_API_KEY or OPENAI_BASE_URL")
        return 2

    # Deterministic, so the prefix is identical across the two requests -- that
    # is the whole experiment.
    prefix = "filler. " * (PREFIX_TOKENS // 2)

    print(f"model: {args.model}")
    print(f"key:   {hashlib.sha256(api_key.encode()).hexdigest()[:8]} "
          f"(from {args.env_file.name})")
    print(f"tools: {'omitted' if args.no_tools else 'one function in the request'}")

    prompt = cached = 0
    try:
        for attempt in range(1, ATTEMPTS + 1):
            prompt, cached = ask(base_url, api_key, args.model, prefix,
                                 not args.no_tools)
            share = cached / prompt if prompt else 0.0
            print(f"request {attempt}: prompt={prompt:6,}  cached={cached:6,}  "
                  f"({share:.1%})")
            if attempt > 1 and share >= HIT_FLOOR:
                break
            if attempt < ATTEMPTS:
                time.sleep(RETRY_DELAY_SEC)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:200]
        print(f"request failed: HTTP {error.code}: {detail}")
        return 2
    except Exception as error:  # noqa: BLE001 - a failed request is not a result
        print(f"request failed: {error}")
        return 2

    share = cached / prompt if prompt else 0.0
    if share >= HIT_FLOOR:
        print(f"\nCACHED: the repeated prompt recovered {share:.1%} of its "
              f"{prompt:,} tokens from cache.")
        return 0
    print(
        f"\nNOT CACHED: the repeated prompt recovered only {cached:,} of "
        f"{prompt:,} tokens after {ATTEMPTS} attempts.\n"
        "Cost estimates weighting cache reads at ~90% understate this model's "
        "real cost -- uncached input is priced several times a cache read."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
