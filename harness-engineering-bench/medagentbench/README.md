# MedAgentBench

A clinical agent over a FHIR electronic health record served inside the task
container. The agent queries the record, performs the POST action a task requires,
and finishes through the shipped CLI, which writes `/workspace/answer.json`.

Raised as a tau3 candidate in
[`../tau3-replacement-analysis.md`](../tau3-replacement-analysis.md) and kept under
consideration at the 2026-07-29 sync alongside DABstep.

## Pinned source

`stanford/medagentbench@sha256:c52d82b3462fb26417707682095e43f224a31f8f785eb7da615c1ab6adc20bf0`

Harbor-native, so nothing is vendored and it compiles on a fresh checkout.

## Split

150 of the 300 tasks, as 30 / 60 / 60, taking **15 of each of the ten task
categories** so every partition holds a balanced 3 / 6 / 6 per category.

The categories divide into retrieval (GET only) and action (issues a POST), and
published scores differ sharply between them: for the strongest reported model,
85.33% on query against 54.00% on action, and the ordering inverts for some models
(Gemini-1.5 Pro reads 52.67 query / 71.33 action). The retrieval-to-action mix
therefore sets where the seed lands, so it is fixed by construction rather than
left to a uniform draw.

150 holds the finalize wall at 8 waves, `ceil(60 x 3 / 24)`. The full 300 would cost
15 and put this in swe-bench-pro's runtime class.

Regenerate or verify, no local export needed since the stratum is in the canonical
name:

```bash
cd harness-engineering-bench/medagentbench
uvx --from 'harbor[modal]==0.20.0' python scripts/partition_medagentbench.py --check
```

## Why the seed shells the CLI instead of curling FHIR

The FHIR server listens on `localhost:8080` **inside the task container**, while
`BaseAgent.run` executes host-side, so every tool goes through
`environment.exec`, as tau3, browsecomp-plus and officeqa already do for their
in-container services.

It routes through `/usr/local/bin/medagentbench_cli.py` rather than curl for two
reasons, both load-bearing:

1. `post` records the request and its acceptance message in a history file that
   `finish` folds into `/workspace/answer.json`. The verifier grades action tasks
   from that history, so a raw curl POST executes but does not count.
2. The CLI retries connection errors with backoff for up to 120 attempts, which
   absorbs the H2 database's boot time. A direct first request would race the
   server coming up and fail the case for a reason unrelated to the harness.

## Scoring

`verify.py` calls `evaluate_submission` against `task_metadata.json`. Binary per
case, no judge model, and its only network call is a GET against the local FHIR
server, so this benchmark keeps `harness_user: harness` isolation.

## Open before this is reported

- **`baseline_reward` is unset and `score_baseline: true`.** Pin it at K=3 with
  `../scripts/rescore_candidate.py --seed` and flip the flag, or every run pays an
  extra held-out pass and the reward is not reproducible.
- **The image is a mutable tag.** All 300 tasks set
  `docker_image = "docker.io/alienkevin/medagentbench-harbor:latest"`, which
  overrides the task's own Dockerfile where the upstream digest pin lives. The
  reproducibility of any number from this benchmark depends on an image we do not
  control and cannot pin from here.
- **`allow_internet = true`** while the data is local. Worth setting false and
  re-verifying if we want a closed-corpus claim.
- **2 CPU per case**, the only benchmark in the suite that asks for it, so 24
  concurrent cases is 48 vCPU of Modal per evaluation. If capacity forces
  concurrency down, recompute `timeout_seconds` and `verifier_timeout_seconds`.
- **Entrypoint risk.** The image runs a Java H2 server. If cases fail immediately
  with "Sandbox already shut down", add
  `--ek 'keepalive=["-c","sleep infinity"]'` the way swe-atlas-qna does.
