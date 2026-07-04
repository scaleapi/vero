"""Local dashboard for watching Harbor optimization trials.

``vero harbor dashboard`` serves a single-page UI (stdlib HTTP server, no new
dependencies) that answers, at a glance: which experiments are running, what
each one is doing right now, and how finished ones scored against their
pre-registered bars.

Data sources, all local to the operator's machine:

- ``docker ps`` discovers live eval-sidecar containers (one per running trial).
- Each sidecar's admin ``/experiments`` endpoint (via ``docker exec``, using the
  in-container admin token) provides the live ledger: one row per recorded
  eval (commit, split, score, error rate). This is the same admin-observability
  surface the ``/experiments`` endpoint was added for; the dashboard never
  reaches into agent volumes.
- ``/status`` (agent-facing, unauthenticated) provides remaining budget.
- Harbor jobs directories are scanned for ``reward.json`` (the trial's final).
- An optional ``--meta`` JSON file supplies display context the containers
  cannot know: the experiment's question, pre-registered bars (baseline /
  win bar / healthy ceiling), notes, and a scorecard of finished experiments.

The server binds 127.0.0.1 by default: ledger rows include hidden-split scores,
so the page is for the operator's eyes, not the network's.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)

SIDECAR_SUFFIX = "-eval-sidecar-1"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no docker, no filesystem)
# ---------------------------------------------------------------------------


def experiment_key(container_name: str) -> str | None:
    """``<task>__<runid>-eval-sidecar-1`` -> ``<task>`` (None if not a sidecar).

    Harbor composes container names as ``<project>-eval-sidecar-1`` where the
    project is ``<task-dir-name>__<random>``; the task dir name is the stable
    key an operator recognizes (e.g. ``gaia-exp11-task``).
    """
    if not container_name.endswith(SIDECAR_SUFFIX):
        return None
    project = container_name[: -len(SIDECAR_SUFFIX)]
    return project.split("__")[0] if "__" in project else project


def derive_phase(*, has_container: bool, n_rounds: int, final: dict | None) -> str:
    """One word for where a trial is: building -> running -> done (or gone).

    ``gone`` = no live container and no reward.json: either torn down abnormally
    or never started; the operator should look at the launch log.
    """
    if final is not None:
        return "done"
    if has_container:
        return "running" if n_rounds > 0 else "building"
    return "gone"


def verdict(final: dict | None, bars: dict | None) -> str | None:
    """WIN / MISS against the pre-registered win bar, when both are known."""
    if not final or not bars or "win_bar" not in bars:
        return None
    scores = [v for v in final.values() if isinstance(v, (int, float))]
    if not scores:
        return None
    return "WIN" if scores[0] > bars["win_bar"] else "MISS"


def normalize_rounds(payload: dict | None) -> list[dict]:
    """/experiments payload -> display rows (newest last), tolerant of Nones."""
    rows = []
    for e in (payload or {}).get("experiments", []):
        rows.append(
            {
                "commit": str(e.get("commit") or "")[:10],
                "split": e.get("split"),
                "score": e.get("mean_score"),
                "error_rate": e.get("error_rate"),
                "created_at": e.get("created_at"),
            }
        )
    return rows


def merge_status(
    *,
    meta: dict,
    sidecars: dict[str, dict],
    finals: dict[str, dict],
) -> dict:
    """Join meta-declared experiments with live containers and finished rewards.

    Experiments the meta file does not mention still appear (key only), so a
    running trial is never invisible just because the operator forgot to list
    it. Meta-only entries appear too, phase ``gone``, so a crashed launch is
    conspicuous rather than silently absent.
    """
    keys = list(
        dict.fromkeys(
            [e["key"] for e in meta.get("experiments", [])]
            + list(sidecars)
            + list(finals)
        )
    )
    by_key = {e["key"]: e for e in meta.get("experiments", [])}
    out = []
    for key in keys:
        m = by_key.get(key, {})
        live = sidecars.get(key)
        final = finals.get(key)
        rounds = normalize_rounds(live.get("experiments") if live else None)
        out.append(
            {
                "key": key,
                "title": m.get("title", key),
                "question": m.get("question"),
                "bars": m.get("bars"),
                "note": m.get("note"),
                "phase": derive_phase(
                    has_container=live is not None,
                    n_rounds=len(rounds),
                    final=final,
                ),
                "rounds": rounds,
                "budget": (live or {}).get("status", {}).get("splits"),
                "final": final,
                "verdict": verdict(final, m.get("bars")),
            }
        )
    return {"experiments": out, "history": meta.get("history", [])}


# ---------------------------------------------------------------------------
# Collectors (thin side-effecting shims around docker + the filesystem)
# ---------------------------------------------------------------------------


def _docker(*args: str, timeout: int = 15) -> str:
    res = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout
    )
    return res.stdout if res.returncode == 0 else ""


def discover_sidecars() -> dict[str, str]:
    """{experiment key: container name} for every live eval sidecar."""
    out = {}
    for name in _docker("ps", "--format", "{{.Names}}").splitlines():
        key = experiment_key(name.strip())
        if key:
            out[key] = name.strip()
    return out


_TOKEN_SNIPPET = (
    "import json;"
    "print(open(json.load(open('/opt/serve.json'))['admin_token_path']).read().strip())"
)


def snapshot_sidecar(container: str) -> dict:
    """Live ledger + budget for one sidecar, via docker exec (admin-side)."""
    snap: dict = {}
    token = _docker("exec", container, "python3", "-c", _TOKEN_SNIPPET).strip()
    if token:
        raw = _docker(
            "exec", container, "curl", "-s",
            "-H", f"Authorization: Bearer {token}",
            "http://localhost:8000/experiments",
        )
        try:
            snap["experiments"] = json.loads(raw)
        except json.JSONDecodeError:
            pass
    raw = _docker("exec", container, "curl", "-s", "http://localhost:8000/status")
    try:
        snap["status"] = json.loads(raw)
    except json.JSONDecodeError:
        snap["status"] = {}
    return snap


def scan_finals(jobs_dirs: list[Path]) -> dict[str, dict]:
    """{experiment key: reward dict} from every reward.json under the jobs dirs.

    The experiment key is recovered from the trial directory name harbor writes
    (``<task>__<runid>``), so finals keep matching after containers are gone.
    """
    finals: dict[str, dict] = {}
    for jd in jobs_dirs:
        for rj in Path(jd).glob("**/reward.json"):
            key = None
            for part in rj.parts:
                if "__" in part:
                    key = part.split("__")[0]
            if key is None:
                key = Path(jd).name
            try:
                finals[key] = json.loads(rj.read_text())
            except (OSError, json.JSONDecodeError):
                continue
    return finals


def collect(meta_path: Path | None, jobs_dirs: list[Path]) -> dict:
    meta: dict = {}
    if meta_path and Path(meta_path).exists():
        try:
            meta = json.loads(Path(meta_path).read_text())
        except json.JSONDecodeError:
            logger.warning("meta file %s is not valid JSON; ignoring", meta_path)
    sidecars = {
        key: snapshot_sidecar(container)
        for key, container in discover_sidecars().items()
    }
    return merge_status(meta=meta, sidecars=sidecars, finals=scan_finals(jobs_dirs))


# ---------------------------------------------------------------------------
# HTTP server + page
# ---------------------------------------------------------------------------

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>vero harbor: experiments</title>
<style>
  :root { color-scheme: dark; }
  body { font: 14px/1.5 -apple-system, "Segoe UI", sans-serif; background:#111418;
         color:#dfe3e8; margin:0; padding:24px; }
  h1 { font-size:18px; margin:0 0 4px; } .sub { color:#8a94a1; margin-bottom:20px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(430px,1fr)); gap:16px; }
  .card { background:#1a1f26; border:1px solid #2a3038; border-radius:10px; padding:16px; }
  .hdr { display:flex; justify-content:space-between; align-items:center; }
  .name { font-weight:600; font-size:15px; }
  .chip { font-size:11px; padding:2px 10px; border-radius:999px; font-weight:600; }
  .running { background:#0d3a2e; color:#4ade80; } .building { background:#3a2e0d; color:#fbbf24; }
  .done { background:#12304d; color:#60a5fa; } .gone { background:#3d1a1a; color:#f87171; }
  .q { color:#aeb7c2; margin:8px 0; font-size:13px; }
  .bars { color:#8a94a1; font-size:12px; margin-bottom:8px; }
  table { width:100%; border-collapse:collapse; font-size:12px; margin-top:6px; }
  th { text-align:left; color:#8a94a1; font-weight:500; padding:2px 6px; }
  td { padding:2px 6px; border-top:1px solid #232932; font-variant-numeric:tabular-nums; }
  .final { margin-top:10px; padding:8px 10px; border-radius:8px; font-weight:600; }
  .WIN { background:#0d3a2e; color:#4ade80; } .MISS { background:#3d2a12; color:#fbbf24; }
  .noverdict { background:#12304d; color:#60a5fa; }
  .hist { margin-top:28px; } .hist table { font-size:12px; }
  .note { color:#8a94a1; font-size:12px; margin-top:6px; }
  .ts { color:#5c6672; font-size:11px; margin-top:16px; }
</style></head>
<body>
<h1>vero harbor: experiments</h1>
<div class="sub">live trials, pre-registered bars, and the campaign scorecard. Auto-refreshes.</div>
<div class="grid" id="grid"></div>
<div class="hist" id="hist"></div>
<div class="ts" id="ts"></div>
<script>
const fmt = v => (v === null || v === undefined) ? "-" : (typeof v === "number" ? v.toFixed(3) : v);
async function refresh() {
  const r = await fetch("/api/status"); const d = await r.json();
  document.getElementById("grid").innerHTML = d.experiments.map(e => {
    const rounds = e.rounds.slice(-10).map(x =>
      `<tr><td>${x.commit}</td><td>${x.split ?? "-"}</td><td>${fmt(x.score)}</td><td>${fmt(x.error_rate)}</td></tr>`).join("");
    const bars = e.bars ? `baseline ${fmt(e.bars.baseline)} &middot; win &gt; ${fmt(e.bars.win_bar)}` +
      (e.bars.healthy !== undefined ? ` &middot; healthy ${fmt(e.bars.healthy)}` : "") : "";
    const budget = (e.budget || []).map(s =>
      s.remaining_run_budget !== undefined && s.remaining_run_budget !== null
        ? `${s.split}: ${s.remaining_run_budget} evals left` : "").filter(Boolean).join(" &middot; ");
    const final = e.final ? `<div class="final ${e.verdict ?? "noverdict"}">final ${
      Object.entries(e.final).map(([k,v]) => k+" = "+fmt(v)).join(", ")}${
      e.verdict ? " &middot; " + e.verdict : ""}</div>` : "";
    return `<div class="card">
      <div class="hdr"><span class="name">${e.title}</span><span class="chip ${e.phase}">${e.phase}</span></div>
      ${e.question ? `<div class="q">${e.question}</div>` : ""}
      ${bars ? `<div class="bars">${bars}</div>` : ""}
      ${budget ? `<div class="bars">${budget}</div>` : ""}
      ${e.rounds.length ? `<table><tr><th>commit</th><th>split</th><th>score</th><th>err</th></tr>${rounds}</table>` : ""}
      ${final}
      ${e.note ? `<div class="note">${e.note}</div>` : ""}
    </div>`;
  }).join("");
  const h = d.history || [];
  document.getElementById("hist").innerHTML = h.length ? `<h1>scorecard: finished experiments</h1>
    <div class="sub">baseline = the unmodified agent's score before optimization &middot;
    bar = the pre-registered score the optimized agent had to beat &middot;
    final = what the optimized agent actually scored on the hidden test set</div><table>
    <tr><th>exp</th><th>question</th><th>baseline</th><th>bar</th><th>final</th><th>outcome</th><th>what happened</th></tr>` +
    h.map(x => `<tr><td>${x.exp}</td><td>${x.question}</td><td>${x.baseline ?? "-"}</td>
      <td>${x.bar ?? "-"}</td><td>${x.final ?? "-"}</td><td>${x.verdict ?? "-"}</td>
      <td>${x.summary ?? "-"}</td></tr>`).join("") + "</table>" : "";
  document.getElementById("ts").textContent = "updated " + new Date().toLocaleTimeString();
}
refresh(); setInterval(refresh, 20000);
</script>
</body></html>
"""


@dataclass
class DashboardConfig:
    meta_path: Path | None = None
    jobs_dirs: list[Path] = field(default_factory=list)
    host: str = "127.0.0.1"
    port: int = 8300


def make_handler(config: DashboardConfig):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            if self.path.startswith("/api/status"):
                body = json.dumps(
                    collect(config.meta_path, config.jobs_dirs)
                ).encode()
                ctype = "application/json"
            elif self.path == "/" or self.path.startswith("/index"):
                body = PAGE.encode()
                ctype = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # quiet: one poll every 20s per client
            pass

    return Handler


def serve_dashboard(config: DashboardConfig) -> None:
    server = ThreadingHTTPServer(
        (config.host, config.port), make_handler(config)
    )
    logger.info("dashboard on http://%s:%d", config.host, config.port)
    print(f"vero harbor dashboard: http://{config.host}:{config.port}")
    server.serve_forever()
