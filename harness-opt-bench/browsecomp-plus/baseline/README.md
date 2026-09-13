# BrowseComp-Plus editable target

The seed is a research agent with three tools:

- search the pinned BM25 index;
- open a document;
- submit a response in the benchmark's required format.

The optimizer may change prompts, tool use, control flow, and dependencies in
`target/`. It cannot change the corpus, index, partitions, target model, or
verifier.

Prepare the generated tasks as described in the
[benchmark overview](../README.md), then compile from the repository root:

~~~bash
cd vero
VERO_SKIP_SECRET_CHECK=1 uv run vero harbor build \
  --config ../harness-opt-bench/browsecomp-plus/baseline/build.yaml \
  --param inner_env=<evaluation-environment> \
  --output <output-directory>
~~~

The `VERO_SKIP_SECRET_CHECK` setting is appropriate only for compile-time
validation. For a real optimization run, use a local credential file and follow
the shared [`run-benchmark` guide](../../skills/run-benchmark/SKILL.md).
