# tau3-bench

This benchmark optimizes a customer-service agent that talks to the canonical
tau3 simulated user and domain environment through the task's `tau3-runtime`
MCP server. It spans airline, retail, telecom, and banking-knowledge domains.

The pinned 375-task dataset is split 75/150/150 for development, validation,
and test. The deterministic split is stratified by domain.

Regenerate or verify the committed split from an exported dataset:

```bash
python harness-engineering-bench/scripts/partition_dataset.py tau3 \
  --tasks-dir /path/to/exported/dataset \
  --output-dir harness-engineering-bench/candidates/tau3/partitions \
  --fetch-registry

python harness-engineering-bench/scripts/partition_dataset.py tau3 \
  --tasks-dir /path/to/exported/dataset \
  --output-dir harness-engineering-bench/candidates/tau3/partitions \
  --check
```
