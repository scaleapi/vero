# Harbor circle-packing example

A **Harbor outer loop with a simple inner loop**: Harbor drives a coding agent
that edits `packing.py` to pack 26 non-overlapping circles in the unit square
(maximize the sum of radii), and each candidate is scored by a **plain
`CommandBackend`** — a fast Python scorer — **not** a nested `harbor run`.

```
 harbor run  ──►  main service (coding agent edits /work/agent/packing.py)
                        │  vero harbor eval  (self-score during the run)
                        ▼
                  eval-sidecar  ──►  CommandBackend → harness/evaluate.py
                  (trusted: scoring, budget, disclosure, final selection)
```

## Why a custom factory (and not `vero harbor build`)

`vero harbor build` compiles a task whose inner evaluation is itself a nested
`harbor run` (it hardcodes `HarborBackend` and requires a `task_source`). That
is the right tool when the inner task is another Harbor benchmark, but it is
heavier than needed here. For a **simple** inner loop you supply a custom sidecar
factory (`sidecar/circle_factory.py`) that wires a `CommandBackend` and run it
with `vero harbor serve --factory circle_factory:build` (see
`environment/docker-compose.yaml`).

## Layout

| Path | Role |
|---|---|
| `task.toml` | Harbor task definition |
| `instruction.md` | What the coding agent is told to do |
| `environment/Dockerfile` | Main service image (target + VeRO CLI) |
| `environment/main/seed.sh` | Seeds `/work/agent` as a git repo, then idles |
| `environment/agent-seed/` | Baseline `packing.py` the agent starts from |
| `environment/sidecar/Dockerfile` | Trusted sidecar image |
| `environment/sidecar/circle_factory.py` | Wires the `CommandBackend`, sidecar policies, objective, verifier |
| `environment/sidecar/harness/evaluate.py` | The scorer (`sum_radii`, `valid`, clearances) |
| `environment/agent-baseline/` | Trusted baseline the sidecar scores against |
| `solution/solve.sh` | Reference: self-score via `vero harbor eval` |
| `tests/test.sh` | Verifier: `vero harbor finalize` → `/logs/verifier/reward.json` |

The objective is `sum_radii` (maximize) subject to `valid == 1`; the baseline
scores ~0.96 and the best known result is ~2.635.

## Run it

Requires Docker and a coding agent supported by Harbor (e.g. `codex`), plus a
model endpoint. From this directory:

```bash
export OPENAI_API_KEY=...          # or your gateway key
export OPENAI_BASE_URL=...         # e.g. a LiteLLM proxy exposing /v1

harbor run -e docker -a codex -m openai/gpt-5.4 --yes
```

Harbor builds the two images, starts the sidecar, runs the agent against
`instruction.md`, then runs `tests/test.sh` to emit the final reward.

## Notes

- The images `pip install scale-vero[harbor]`. From a local checkout instead,
  copy the package into the build context and `uv pip install "/opt/vero[harbor]"`
  (see the commented lines in the Dockerfiles).
- Multi-metric rewards: the verifier emits `reward.json` (a dict), and the
  sidecar can run in a separate verifier environment for stronger isolation from
  an untrusted agent — see the Harbor task docs.
- This mirrors a pattern validated end-to-end (codex improved the baseline from
  0.96 toward the ~2.635 optimum); it needs Docker + a coding agent + model
  access to run and is not exercised by the unit test suite.
