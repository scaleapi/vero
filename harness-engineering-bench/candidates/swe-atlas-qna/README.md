# SWE-Atlas-QnA

This benchmark optimizes an agent that investigates a checked-out software
repository and writes an evidence-backed answer to a deep codebase question.
Scoring is the canonical rubric judge shipped in each Harbor task.

The pinned 124-task dataset is split 25/49/50 for development, validation, and
test. The split is deterministic and stratified by source repository so that
the ten represented codebases occur across the three partitions.

Regenerate or verify the committed split from an exported dataset:

```bash
python harness-engineering-bench/scripts/partition_dataset.py swe-atlas-qna \
  --tasks-dir /path/to/exported/dataset \
  --output-dir harness-engineering-bench/candidates/swe-atlas-qna/partitions \
  --fetch-registry

python harness-engineering-bench/scripts/partition_dataset.py swe-atlas-qna \
  --tasks-dir /path/to/exported/dataset \
  --output-dir harness-engineering-bench/candidates/swe-atlas-qna/partitions \
  --check
```

`--fetch-registry` requires the pinned Harbor package and verifies every task
name and content digest against Harbor Hub.
