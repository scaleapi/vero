#!/usr/bin/env python3
"""Report the rate limits LiteLLM applies to each credential file.

Concurrency planning is bounded by the per-key bucket, so this prints what each
`*.secrets.env` actually gets rather than what we assume it gets:

    python3 harness-engineering-bench/scripts/check_keys.py vero/

Sends one 1-token request per key and reads the response headers. Note the
`llm_provider-*` headers are the provider's limits echoed per request, NOT a
depleting shared bucket — see the concurrency note (§) in CONFIGURATION.md — so
only the `x-ratelimit-api_key-*` values below bound how many runs fit.

**A raised limit can take ~15 minutes to propagate.** A key still reporting its
old ceiling right after a change is probably stale, not unchanged — re-run before
concluding anything or re-planning concurrency around it.

Prints no secret material: keys are shown as a short fingerprint only.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# One officeqa run at max_concurrency 24, measured: peak sustained draw.
RUN_TPM = 2_900_000
RUN_RPM = 182
PROBE_MODEL = "fireworks_ai/deepseek-v4-flash"


def load_env(path: Path) -> dict[str, str]:
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


def probe(base_url: str, api_key: str) -> dict[str, str]:
    body = json.dumps(
        {
            "model": PROBE_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as error:
        return {"__error__": f"HTTP {error.code}", **{k.lower(): v for k, v in error.headers.items()}}
    except Exception as error:  # noqa: BLE001 - report and continue to the next key
        return {"__error__": str(error)[:80]}


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    files = sorted(root.glob("*.secrets.env")) + sorted(root.glob("secrets.env"))
    if not files:
        print(f"no *.secrets.env or secrets.env under {root}")
        return 1

    print(f"{'file':28s} {'key':10s} {'TPM limit':>12s} {'RPM limit':>10s} "
          f"{'runs/key':>9s}  note")
    total = 0
    for path in files:
        env = load_env(path)
        key, base = env.get("OPENAI_API_KEY"), env.get("OPENAI_BASE_URL")
        if not key or not base:
            print(f"{path.name:28s} {'-':10s} {'-':>12s} {'-':>10s} {'-':>9s}  "
                  "missing OPENAI_API_KEY/OPENAI_BASE_URL")
            continue
        fingerprint = hashlib.sha256(key.encode()).hexdigest()[:8]
        headers = probe(base, key)
        if "__error__" in headers and "x-ratelimit-api_key-limit-tokens" not in headers:
            print(f"{path.name:28s} {fingerprint:10s} {'-':>12s} {'-':>10s} {'-':>9s}  "
                  f"{headers['__error__']}")
            continue
        tpm = int(headers.get("x-ratelimit-api_key-limit-tokens", 0) or 0)
        rpm = int(headers.get("x-ratelimit-api_key-limit-requests", 0) or 0)
        by_tpm = tpm // RUN_TPM if tpm else 0
        by_rpm = rpm // RUN_RPM if rpm else 0
        fits = min(x for x in (by_tpm, by_rpm) if x) if (by_tpm or by_rpm) else 0
        bound = "TPM-bound" if by_tpm <= by_rpm else "RPM-bound"
        total += fits
        print(f"{path.name:28s} {fingerprint:10s} {tpm:12,} {rpm:10,} {fits:9d}  {bound}")

    print(f"\nOne run draws ~{RUN_TPM:,} TPM / ~{RUN_RPM} RPM (officeqa at "
          f"max_concurrency 24, measured peak).")
    print(f"Combined ceiling across these keys: ~{total} concurrent runs.")
    print("Distinct keys only help if each run is launched with its own "
          "--env-file; runs sharing a file share its bucket.")
    print("A raised limit can take ~15 min to propagate; an unexpectedly low "
          "reading right after a change is probably stale.")
    print("Local capacity is a separate and usually tighter ceiling, and "
          "container memory binds well before CPU does: each run is 3 "
          "containers, and a Docker VM that looks idle on load average will "
          "still OOM-kill a run once their combined footprint exceeds it. "
          "Measure your own VM's limit rather than sizing off core count; the "
          "OOM surfaces misleadingly as a rate-limit error. Put the outer "
          "trial on Modal to get past it. Modal sandbox concurrency (runs x "
          "max_concurrency) is a third ceiling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
