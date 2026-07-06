"""`vero harbor` CLI.

Thin clients the optimizer and verifier use inside the compiled task:
  - agent (in `main`):    eval / submit / status  -> POST/GET the sidecar over VERO_EVAL_URL
  - verifier (in `main`): finalize                -> POST /finalize with the admin token,
                                                     write /logs/verifier/reward.json
`serve` (sidecar entry) and `build`/`run` (host-side compiler) are added with stage (c).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click


def _base_url() -> str:
    url = os.environ.get("VERO_EVAL_URL")
    if not url:
        raise click.ClickException("VERO_EVAL_URL is not set (the eval sidecar URL).")
    return url.rstrip("/")


def _request(method: str, path: str, *, payload: dict | None = None, headers: dict | None = None):
    import httpx

    resp = httpx.request(
        method, f"{_base_url()}{path}", json=payload, headers=headers or {}, timeout=None
    )
    if resp.status_code >= 400:
        raise click.ClickException(f"{method} {path} -> {resp.status_code}: {resp.text}")
    return resp.json()


@click.group()
def harbor() -> None:
    """Vero ⇄ Harbor: optimization-as-a-Harbor-task commands."""


@harbor.command("serve")
@click.option("--config", "config_path", required=True, help="Path to the ServeConfig JSON.")
def serve_cmd(config_path):
    """Eval-sidecar entrypoint: assemble the engine/sidecar/verifier and serve (uvicorn)."""
    from vero.harbor.serve import serve

    serve(config_path)


@harbor.command("build")
@click.option("-c", "--config", "config_path", required=True, help="Path to build.yaml.")
@click.option("-o", "--out", required=True, help="Output task directory.")
def build_cmd(config_path, out):
    """Compile a build.yaml into a runnable Harbor optimization task directory."""
    from vero.harbor.build import compile_task, load_build_config

    task_dir = compile_task(load_build_config(config_path), out)
    click.echo(f"Compiled task -> {task_dir}")


@harbor.command("run", context_settings={"ignore_unknown_options": True})
@click.option("-c", "--config", "config_path", required=True, help="Path to build.yaml.")
@click.option("-a", "--agent", required=True, help="Optimizer agent (passed to harbor run).")
@click.option("-m", "--model", default=None, help="Model for the optimizer agent.")
@click.option("-e", "--environment", "provider", default="docker", show_default=True)
@click.argument("extra", nargs=-1, type=click.UNPROCESSED)
def run_cmd(config_path, agent, model, provider, extra):
    """Build to a temp dir, then `harbor run` it (build + run convenience)."""
    import subprocess
    import tempfile

    from vero.harbor.build import compile_task, load_build_config

    task_dir = compile_task(load_build_config(config_path), Path(tempfile.mkdtemp()) / "task")
    cmd = ["uvx", "harbor", "run", "-p", str(task_dir), "-a", agent, "-e", provider]
    if model:
        cmd += ["-m", model]
    cmd += list(extra)
    click.echo(f"$ {' '.join(cmd)}")
    raise SystemExit(subprocess.call(cmd))


@harbor.command("eval")
@click.option("--dataset-id", required=True)
@click.option("--split", required=True)
@click.option("--commit", default=None, help="Defaults to the agent repo HEAD.")
@click.option("--num-samples", type=int, default=None)
@click.option("--sample-ids", default=None, help="Comma-separated sample ids.")
def eval_cmd(dataset_id, split, commit, num_samples, sample_ids):
    """Spend one evaluation of your commit on a split (agent)."""
    payload: dict = {"dataset_id": dataset_id, "split": split}
    if commit:
        payload["commit"] = commit
    if num_samples is not None:
        payload["num_samples"] = num_samples
    if sample_ids:
        payload["sample_ids"] = [int(x) for x in sample_ids.split(",")]
    click.echo(json.dumps(_request("POST", "/eval", payload=payload), indent=2))


@harbor.command("submit")
@click.option("--commit", default=None, help="Defaults to the agent repo HEAD.")
def submit_cmd(commit):
    """Nominate a commit and end the optimization run (agent; if enabled)."""
    click.echo(json.dumps(_request("POST", "/submit", payload={"commit": commit}), indent=2))


@harbor.command("status")
def status_cmd():
    """Show remaining budget, evaluable splits, and whether submit is enabled (agent)."""
    click.echo(json.dumps(_request("GET", "/status"), indent=2))


@harbor.command("finalize")
@click.option("--token-file", required=True, help="Path to the admin token (root:600).")
@click.option("--output", default="/logs/verifier/reward.json", show_default=True)
def finalize_cmd(token_file, output):
    """Verifier: select the best/submitted commit, score on the test split, write reward.json (admin)."""
    from vero.harbor.auth import read_admin_token

    token = read_admin_token(token_file)
    resp = _request("POST", "/finalize", headers={"Authorization": f"Bearer {token}"})
    # finalize returns {"rewards": {...}, "baseline": {...}}. Only the rewards are
    # the reward.json payload the outer harness consumes; the baseline outcome is
    # echoed to stdout (the trial's stdout survives teardown, the admin volume does
    # not) so a baseline skip or failure is durably recorded. Tolerate the older
    # bare-rewards shape so a mixed-version sidecar still writes a valid reward.json.
    rewards = resp["rewards"] if isinstance(resp, dict) and "rewards" in resp else resp
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rewards))
    click.echo(json.dumps(resp, indent=2))
