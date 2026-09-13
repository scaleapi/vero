# tau3-bench

tau3 evaluates a customer-service agent that talks with a simulated user and a
domain environment through each task's MCP server.

## At a glance

| Item | Value |
| --- | --- |
| Domains | Airline, retail, telecom, and banking knowledge |
| Cases | 375 |
| Development / validation / test | 75 / 150 / 150 |
| Editable harness | `baseline/target/` |
| Split strategy | Domain |
| Status | Archived; not part of the reported suite |

Development exposes complete task results. Validation is aggregate-only, and
test is held out for final scoring.

## Data and split

Verify the deterministic split against an exported dataset:

~~~bash
python harness-opt-bench/scripts/partition_dataset.py tau3 \
  --tasks-dir <exported-tasks> \
  --output-dir harness-opt-bench/archive/tau3/partitions \
  --check
~~~

Use `--fetch-registry` when intentionally refreshing the pinned package.

See [the baseline guide](baseline/README.md) for the editable target and build
command.
