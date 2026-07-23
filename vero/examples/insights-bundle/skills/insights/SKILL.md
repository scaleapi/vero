---
name: insights
description: >-
  Corpus-level trace diagnostics for the VeRO optimizer. Mine a corpus of agent
  execution traces for failure/success patterns, validate each against the WHOLE
  corpus with quantitative evidence, and emit evidence-grounded findings. Use
  when asked to analyze why candidates fail, what distinguishes high- from
  low-scoring traces, or what patterns recur across an evaluation corpus.
---

# Insights: corpus-level trace diagnostics

This skill reproduces Scale's **Insights Generator** methodology using only the
sandbox filesystem + bash. It finds patterns across a *population* of agent
execution traces (patterns invisible in any single trace) and returns findings
backed by corpus-scale evidence.

Three rules define the method — everything else is mechanics:

1. **Compute over all traces; read few.** The corpus is too large for one
   context. Use `grep`/`jq`/`awk`/`wc` to compute *over* every trace; `cat`/`sed`
   the full text of only a sampled or cohort-relevant handful.
2. **Discovery ≠ validation.** *Scouts* skim a sample and propose many hypotheses
   (breadth). *Investigators* test one hypothesis against the *whole* corpus with
   quantitative evidence and label it confirmed / refuted / inconclusive (depth).
   **No finding ships without corpus-scale backing.**
3. **Every finding is evidence-grounded and has a persisted cohort.** description
   + prevalence % + a saved cohort of affected trace IDs + ≥8 verbatim quotes
   (each from a distinct trace) + mechanism + suggested action. A finding with no
   saved cohort is **rejected** (`validate.py`).

## The corpus layout

Under the **Harbor** eval backend (the path VeRO actually uses), each candidate's
evaluation dumps per-case trial records into a read-only `.vero` tree:

```
$CORPUS/                                # e.g. /work/agent/.vero (or a dir you're pointed at)
  evaluations/<eval_digest>/            # one dir per candidate evaluation
    evaluation.json                     # aggregate record (score, error_rate, candidate id)
    artifacts/harbor/
      stdout.log  stderr.log            # backend logs for the whole eval
      jobs/<timestamp>/<case_id>__<suffix>/     # one dir per case = a "trial"
        result.json                     # trial metadata: task_name, trial_name, agent model
        trial.log
        agent/<task>-trace.jsonl        # THE TRACE — JSONL, one event per line
        agent/answer.txt                # the agent's final answer
        verifier/reward.txt             # THE LABEL — numeric reward (e.g. 0 or 1)
        verifier/test-stdout.txt
```

- **The trace** = `agent/*.jsonl` (for GAIA, `gaia-trace.jsonl`). One JSON object
  per line. Event shapes seen so far:
  - a model turn `{"turn":N,"response_id":..,"output_text":..,"function_calls":[..]}`
    (note **`function_calls`** — plural, an array; grepping `.function_call` finds nothing);
  - a tool result `{"turn":N,"tool":"run_shell","result":{"return_code":..,"stdout":..,"stderr":..}}`;
  - non-shell tool results (`transcribe_audio`, `read_image`, `submit_answer`, …)
    whose `result` may have `"return_code":null` and empty stdout — a **silent
    tool failure** the agent often doesn't notice;
  - a recovery marker `{"recovery":"retry_after_empty_response","turn":N}`.
  The filename and inner shape are **task/agent-specific** — don't hard-code
  beyond "it's JSONL you grep line by line; `jq -c .` per line for structure".
- **The label** for cohort contrast = `verifier/reward.txt` (a bare number).
  Comparative analysis (reward 1 vs 0) is where this method earns its keep.
- **The gold + agent answer** = `verifier/test-stdout.txt`, which for exact-match
  tasks contains both `Agent answer: '...'` and `Expected answer: '...'`. This is
  the highest-value signal for wrong-answer analysis — grep both lines to see
  *what* the agent got wrong, not just that it failed.
- **The unit** = a *trial* (one case run by one candidate). The trial dir name is
  `<case_id>__<suffix>`; the same `case_id` recurs across candidate evals, so use
  the **full trial dir name as the unique `trace_id`** and keep `case_id` +
  `eval_digest` as grouping columns.
- Only **full-disclosure** evals expose per-case traces; aggregate-only evals
  (e.g. a held-out validation split) have `evaluation.json` but no trial `agent/`
  dirs. Mine the full-disclosure evals.

Build a flat index once so lookups are cheap:

Drive the index off the trace files themselves — one row per trace, so the
`jobs/<timestamp>` dir (whose name also contains `__`) never sneaks in:

```bash
# trace_id(trial)  case_id  eval  reward  trace_path      (one row per trial)
find "$CORPUS/evaluations" -path "*/agent/*.jsonl" | while read -r trace; do
  t=$(dirname "$(dirname "$trace")")            # trial dir = .../<case_id>__<suffix>
  trace_id=$(basename "$t")
  case_id=${trace_id%%__*}
  eval=$(echo "$t" | sed -E 's#.*/evaluations/([^/]{12})[^/]*/.*#\1#')
  reward=$([ -f "$t/verifier/reward.txt" ] && tr -d '[:space:]' < "$t/verifier/reward.txt")
  printf '%s\t%s\t%s\t%s\t%s\n' "$trace_id" "$case_id" "$eval" "${reward:-NA}" "$trace"
done > index.tsv
wc -l index.tsv                         # corpus size (trials)
cut -f4 index.tsv | sort | uniq -c      # reward distribution (0 vs 1 = your cohorts)
```

A trial can lack `verifier/reward.txt` (errored/uncsored case) — those rows get
`reward=NA` and simply drop out of the reward-1-vs-0 cohorts.

**Cross-trial contrast (the population-level lever).** Because the same `case_id`
is run by several candidates, grouping by `case_id` reveals patterns invisible in
any single trace — e.g. *deterministic* failures (a case that fails with the
**identical** wrong answer across every candidate → a grounding/knowledge gap,
not noise) vs *stochastic* ones (fails sometimes). This is precisely the kind of
finding this method exists to surface:

```bash
# cases that NEVER pass (deterministic failure) vs sometimes-pass (stochastic)
awk -F'\t' '$4!="NA"{tot[$2]++; if($4==1)pass[$2]++}
  END{for(c in tot) printf "%s\t%d/%d pass\n", c, pass[c]+0, tot[c]}' index.tsv \
  | sort -t/ -k1 -n
# then, for a never-pass case, check whether the wrong answer is identical across
# candidates (grep "Agent answer:" in each trial's verifier/test-stdout.txt).
```

## Tool cookbook (IG tool → bash)

| IG tool | Do this instead |
|---|---|
| `search_traces(query)` | `grep -rlF "<phrase>" "$CORPUS"` (lexical). For "did the agent NOT do X", grep for the text an agent that DID X would emit, then diff that set against the corpus. |
| `get_trace(id)` | `cat` the trace path from `index.tsv`. If huge, `chunk.sh` it first. |
| (recover the answer) | `grep -E "Agent answer:\|Expected answer:" <trial>/verifier/test-stdout.txt` — what the agent said vs the gold answer, for exact-match tasks. |
| `get_trace_chunk(id, k)` | `bash chunk.sh <trace_path> <k>` (byte-window slice; `chunk.sh <path>` with no k lists chunk count). |
| `extract(taxonomy)` → `features.jsonl` | a small loop that computes one feature per trace and appends `{"trace_id":..,"<feat>":..}` to `features.jsonl` (persist + reuse; grep for a regex signal, or a tiny LLM call per trace only when judgment is needed). |
| `compare_segments(A,B)` | `jq`/`awk` group-by over `features.jsonl` joined to `index.tsv` labels — e.g. rate of a feature in pass vs fail cohorts. |
| `save_affected_traces(ids)` | write the **defensible** cohort to `cohorts/<hypothesis-slug>.txt` (one trace_id per line). |
| `submit_finding(...)` | append a finding object to `findings.json` (schema below), then run `python validate.py findings.json index.tsv` — it **rejects** findings lacking a cohort or citations. |

Prevalence = `|cohort| / |corpus|`. If that ratio exceeds ~0.25, either say so
explicitly in `prevalence` or your cohort is actually the analyzed pool — narrow it.

## The finding contract (findings.json)

`findings.json` is `{"executive_summary": str, "findings": [<finding>], "methodology": str, "traces_analyzed": int}`.
Each `<finding>`:

```json
{
  "hypothesis_name": "Short title",
  "status": "confirmed|refuted|inconclusive",
  "summary": "Precise mechanism, not a vague restatement",
  "prevalence": "X of Y traces (Z%)",
  "cohort_file": "cohorts/<slug>.txt",
  "evidence": [
    {"trace_id": "<id>", "quote": "<verbatim excerpt>", "explanation": "2-3 sentences naming the specific mechanism in THIS trace (exception type / tool / turn index) and linking it to the prevalence claim"}
  ],
  "stats": {"metric": 0.85},
  "segments_compared": ["pass", "fail"],
  "additional_observations": ["p-values / odds ratios / notes"],
  "suggested_action": "One concrete change (exact file/prompt/config) or null"
}
```

**Evidence discipline (enforced by `validate.py`):**
- A finding MUST reference a non-empty `cohort_file` whose trace_ids exist in the corpus.
- ≥8 evidence items (aim 10-15 for high-prevalence findings), each from a
  **distinct** trace_id, each with a **verbatim** `quote` (tool-call args,
  exception messages, raw turn content, literal error strings — never paraphrase).
- `explanation` names the concrete mechanism in that trace; don't tag a quote as
  "representative of N traces".
- Include a counterexample or two (pattern absent + task succeeded), marked as such.

## Self-check before submitting a finding
Pick 2-3 random trace_ids from the cohort, `cat`/`chunk.sh` them, and confirm each
clearly exhibits the pattern. If one doesn't, narrow the cohort and re-save.
