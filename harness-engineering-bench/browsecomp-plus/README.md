# BrowseComp-Plus

This benchmark turns all 830 queries from
[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) into local Harbor
tasks. It evaluates a deep-research agent against the benchmark's fixed corpus
and canonical BM25 index rather than the live web.

## Pinned sources

- Upstream repository submodule: `046949032b0328319cc9a02663a759ec601d9402`
- Query dataset `Tevatron/browsecomp-plus`:
  `144cff8e35b5eaef7e526346aa60774a9deb941f`
- BM25 index `Tevatron/browsecomp-plus-indexes`:
  `b3f37f70c33829eb09d04784a54277a31871fd63`

The submodule is the Git pointer requested for the integration. The generator
refuses to run if it is checked out at another commit. Hugging Face revisions
are full immutable commit ids as well.

## Generate Harbor tasks

Initialize the submodule and run the builder from this directory:

```bash
git submodule update --init harness-engineering-bench/browsecomp-plus/upstream

cd harness-engineering-bench/browsecomp-plus
uv run --no-project --python 3.12 --with datasets==4.0.0 -- \
  python scripts/build_tasks.py
```

The builder downloads and decrypts the pinned query dataset using the pinned
upstream implementation, then writes one complete Harbor task per query under
the ignored `tasks/` directory. It also regenerates the committed deterministic
166/332/332 development, validation, and test split. Use `--force` to replace
an existing generated tree or `--check` to verify it byte-for-byte.

The first Harbor image build downloads the pinned BM25 index (about 2.2 GB).
Every task has an identical environment, so subsequent tasks reuse that image.

## Scoring and trust boundary

Answers use BrowseComp-Plus's required Explanation / Exact Answer / Confidence
format. The verifier follows the official upstream OpenAI evaluator and its
default `gpt-4.1` judge. The upstream primary leaderboard instead runs the same
judge prompt with Qwen3-32B; this Harbor integration deliberately uses the
repository's supported API evaluator so tasks do not require a local GPU judge.

The judge runs inside the task environment and therefore receives the real
upstream credential. As with the SWE-Atlas-QnA rubric judge and tau3 task-owned
LLM services, this disables uid isolation and assumes a non-adversarial
optimizer. The editable baseline itself exposes no live-web or shell tool and
uses only the fixed local index.
