# SWE-Atlas-QnA

SWE-Atlas-QnA evaluates an agent that investigates a checked-out repository and
writes an evidence-backed answer to a codebase question.

## At a glance

| Item | Value |
| --- | --- |
| Cases | 124 |
| Development / validation / test | 25 / 49 / 50 |
| Editable harness | `baseline/target/` |
| Split strategy | Source repository |
| Scoring | Canonical rubric judge |
| Status | Archived; not part of the reported suite |

Development exposes complete task results and repositories. Validation is
aggregate-only, and test is held out for final scoring.

## Data and split

The split is deterministic and keeps all ten source repositories represented
across partitions. Verify it against an exported dataset:

~~~bash
python harness-opt-bench/scripts/partition_dataset.py swe-atlas-qna \
  --tasks-dir <exported-tasks> \
  --output-dir harness-opt-bench/archive/swe-atlas-qna/partitions \
  --check
~~~

Use `--fetch-registry` when intentionally refreshing the pinned package. Review
the task manifest and split before accepting new results.

See [the baseline guide](baseline/README.md) for the editable target and the two
target-model builds.
