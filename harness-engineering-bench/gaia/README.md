# GAIA

The GAIA benchmark uses the canonical Harbor task packages and verifier. It
does not reproduce the paper's pure-language subset or its original split.

The committed split is deterministic and stratified by GAIA level and whether
the task has an attached file:

- development: 33 cases (20%)
- validation: 66 cases (40%)
- test: 66 cases (40%)

All 165 cases come from the immutable dataset reference recorded in
[`partitions/manifest.json`](partitions/manifest.json). The development set is
available to the optimization agent with full result disclosure. Validation is
aggregate-only and is used to select candidates. Test is held out until Harbor
grades the completed outer task. The complete development task packages and
attachments are mounted read-only under `.evals/tasks/`; successful and failed
development evaluations place their complete Harbor trial records—including
exact failures and target-agent logs—under
`.evals/results/`. Neither validation nor test resources are mounted.

The pinned GAIA tasks declare a 600-second Harbor agent timeout. The build
config sets both `case_timeout_seconds` and `task_agent_timeout_seconds` to that
same 600, so VeRO's derived agent-timeout multiplier is `1.0` and the target
agent gets exactly the clock the dataset intends. (This was 180/600 with a `0.3`
multiplier until 2026-07-28, when timeouts were taken from each dataset's own
declaration rather than chosen by us.)

To verify or regenerate the split after downloading the pinned Harbor dataset:

```bash
uv run --python 3.12 scripts/partition_gaia.py \
  --tasks-dir /path/to/downloaded/gaia \
  --check
```

Refreshing from a different registry revision is an explicit operation: update
the dataset constant in the script and `build.yaml`, then run with
`--fetch-registry`. Review the manifest diff before committing it.
