# Benchmark configuration

Every benchmark compiles from its `*/baseline/build.yaml`, which is the source
of truth. This page records the normalized settings shared across the suite,
the per-benchmark values, and the conventions behind them, so a change to one
benchmark can be checked against the others at a glance.

## Shared settings (all benchmarks)

- **Objective**: maximize `score`; downstream task accuracy is the only input
  to selection and reward. Cost and latency are reported as evaluation metrics
  (`inference_*_tokens`, `wall_seconds`, `mean/max_case_wall_seconds`) and in
  `finalize.json`'s `reward_metrics`, but never scored.
- **Partitions**: development (full disclosure, task resources exposed),
  validation (aggregate-only, `min_aggregate_cases: 5`), test (held out;
  single target, `reward_key: reward`, `failure_value: 0`, `max_attempts: 1`).
- **Selection**: `reward_mode: submit` — the agent nominates its candidate;
  auto-best over validation and then last-candidate are fallbacks only.
  `baseline_floor: false` (a floor would gate on validation while the reward
  is on test; opt-in only). `score_baseline: false` — the seed's held-out score is
  pinned as `baseline_reward` (◆) instead of being re-measured every run, which is
  both reproducible and one fewer full evaluation per run. `rescore_top_k: 3`,
  `rescore_attempts: 1`.
- **Budgets**: 100 runs and 4 full passes of case budget on each agent
  partition (development and validation).
- **Budget disclosure**: `disclose_budget: true` (the default; no benchmark
  overrides it) — the agent sees remaining runs/cases and inference-scope
  usage through `evals status` and `plan.json`, plus per-evaluation
  `inference_*` token metrics. Setting `disclose_budget: false` is the
  budget-blind ablation: enforcement is unchanged but all budget signal is
  hidden from the agent, including a redaction of `inference_*` metrics from
  agent-facing receipts and context (latency metrics stay visible).
- **Per-trial token/latency accounting**: the trusted gateway meters each
  evaluation's usage as an input/cached/output/total split
  (`inference_input_tokens`, `inference_cached_input_tokens`,
  `inference_output_tokens`, `inference_total_tokens`, `inference_requests`) in
  `reward_metrics`; each trial's `result.json` also carries agent-self-reported
  tokens (`agent_reported_*`) and its wall window. Cost is reported per case the
  same way latency is, and because token and latency distributions are unbounded
  and heavy-tailed (a few cases carry much of the total) each carries a **mean, a
  median, and a max** — `mean/median/max_case_wall_seconds` and
  `mean/median/max_case_agent_reported_{input,cached_input,output}_tokens`.
  Accuracy, being bounded, keeps mean plus stddev. The trusted gateway figure is
  metered per evaluation rather than per case, so it contributes the sum plus a
  derived `mean_case_inference_*`; its median and max come from post-hoc
  attribution (`scripts/per_trial_tokens.py`, which reports mean/median/max over
  trials for every token and latency measure). All of these are budget signal and
  are redacted from the agent under `disclose_budget: false`; latency is not. With
  `inference_gateway.request_log_attribution: true` the gateway stamps every
  request-log record with a `thread_id`, so `scripts/per_trial_tokens.py`
  attributes gateway tokens to individual trials (trusted, versus a content-match
  fallback when off) and rolls a run — or a grid of runs — up to a flat CSV.
  Enabled on officeqa; extended across the suite as the grid rolls out. Because
  each scope's target model is fixed, **dollar cost is a linear function of the
  (input, cached, output) token triple** with a per-model rate vector — computed
  downstream, deliberately not stored anywhere in the run.
- **Target model**: `fireworks_ai/deepseek-v4-flash` by default (see the
  per-benchmark table — a benchmark may pin a different evaluated model), fixed
  per run by the gateway's evaluation scope (the `evaluation.allowed_models`
  allow-list). That confines the target to its evaluated model in the normal
  case, but it is **not a hard guarantee**: both scopes share one gateway host
  split only by URL path, and the optimizer both holds the producer token and
  authors the candidate, so an adversarial optimizer could smuggle that token
  into the candidate and reach `/scopes/producer` from the eval sandbox. Closing
  it needs per-role egress isolation; see the note in
  `vero/src/vero/gateway/inference.py`. On the three benchmarks that set
  `task_services_use_upstream`, the raw upstream credential is in the candidate
  harness's environment as well — see the isolation note below. The optimizer
  uses a separate
  producer scope bound to `${optimizer_model:-openai/gpt-5.4}`. The gateway
  matches the requested model against the allow-list as an exact string, so the
  `-m` the outer trial is launched with has to be spelled the same way as this
  default (or as whatever `--param optimizer_model=` overrides it with);
  the router resolves both `gpt-5.4` and `openai/gpt-5.4`, so the prefix is a
  convention rather than a requirement *upstream* — but the gateway's own
  allow-list check is an exact string match, so the two spellings are **not**
  interchangeable there. See the optimizer-harness note below for why that bites
  with opencode. deepseek-v4-flash was
  chosen over gpt-oss-120b and gpt-5.4-mini from a 10-trial per-benchmark probe:
  it matches or beats both on tau3 (0.875) and is ~2–3× gpt-oss on the
  grounded-reasoning benchmarks (officeqa/browsecomp 0.60) at roughly gpt-oss
  cost — far cheaper than mini. `swe-atlas-qna` is pinned to
  `fireworks_ai/gpt-oss-120b`, the one benchmark where deepseek is weaker
  (0.30 vs 0.59 mean rubric) and gpt-oss is both cheaper and stronger. `gaia` is
  pinned to `gpt-5.4-mini` for a different reason: it is the one multimodal
  benchmark, 5 of its 66 held-out tasks send image inputs, and deepseek-v4-flash
  rejects those outright (`This model does not support image inputs`), capping
  achievable reward near 0.92 and disguising the shortfall as agent failure.
- **Execution**: `harbor[modal]==0.20.0`, python 3.12, `n_attempts: 1` globally,
  but **every benchmark's `test` target overrides to `n_attempts: 3` /
  `aggregate_attempts: mean`**. This is not optional polish: each pinned
  `baseline_reward` (◆) was itself pooled over 3 rounds, so scoring a submitted
  candidate once would give it ~√3 more standard error than the floor it is
  compared against. Search and validation keep the global 1. `max_retries: 1`, 3
  infrastructure attempts at 5s, `aggregate_attempts: best`,
  `max_concurrency: 24` (see § — every timeout below is derived from this
  number), `error_rate_threshold: 0.1`, `feedback_transcripts: true` with
  `feedback_max_bytes: 16000`, `environment_name: ${inner_env:-modal}`.
  **`inner_env=docker` does not work** — the inner evaluation shells out to
  `harbor run -e docker` from inside the sidecar container, which has no docker
  CLI or socket, so every case fails with `Docker is not installed or not on
  PATH`, harbor produces 0 trial groups, and the evaluation 502s. Verified by
  `vero/examples/harness-conformance`. Inner evals are Modal-only until the
  sidecar image ships a docker client and a mounted socket.
- **Optimizer harness**: `--agent claude-code` needs nothing special — vero points
  `ANTHROPIC_BASE_URL`/`ANTHROPIC_API_KEY` at the producer scope, and the model
  string it requests matches `--model` as spelled.

  **`--agent opencode` requires two flags and the provider-native prefix.** It
  *requires* the `provider/model` form and raises `ValueError` without it, but
  registers the model under that provider as the **bare id**, so the bare id is
  what the gateway receives while `${optimizer_model}` would put the prefixed form
  in the allow-list → 403 `model_denied`. Launch a Claude model as:

      --agent opencode --model anthropic/claude-sonnet-5 \
        --param optimizer_model=claude-sonnet-5

  `--param` wins over the `--model`-derived default (`setdefault` in
  `harbor/cli.py`), so the prefixed form reaches opencode and the bare form
  reaches the allow-list. Measured on `vero/examples/harness-conformance`: reward
  1.0 over 39 steps, 39 metered `messages` calls.

  **Use the provider-native prefix, not `openai/`, for non-OpenAI models.**
  Harbor's adapter injects a `baseURL` into `~/.config/opencode/opencode.json`
  only when the provider half is `openai` (`agents/installed/opencode.py`), and
  opencode ignores `ANTHROPIC_BASE_URL`, so `anthropic/…` used to escape the
  gateway entirely. vero now supplies the missing baseURL itself via
  `--ak opencode_config=…` for any non-`openai` provider (`harbor/cli.py`), which
  the adapter deep-merges last. Two consequences:

  - `openai/claude-sonnet-5` **crashes** — do not reach for it as a workaround. It
    forces the Responses API, and litellm's Anthropic→Responses translation emits
    three id namespaces in one stream (a `resp_` id, Anthropic `msg_`/`toolu_`
    item ids, and a stray `chatcmpl-` id); opencode dies resolving a text part
    under an id it never registered. All six main-model calls returned `200`, so
    the gateway is not at fault.
  - Escaping the gateway **fails closed; it does not leak.** The optimizer only
    ever holds a scoped producer token, so a direct call to a provider's public
    endpoint returns `401 invalid x-api-key` and the run dies. The upstream
    credential never leaves the gateway container. An earlier revision of this
    note claimed such a run would hold a credential the optimizer should not see;
    that was wrong.

  **Allow a second producer model.** opencode also issues an auxiliary
  summarisation/title call using a small model of the same provider family
  (`claude-haiku-4-5` on the anthropic path, `gpt-5.4-nano` on the openai one).
  With a single allow-list entry that call `403`s. It is non-fatal but invisible
  outside the gateway request log, and the model varies by family — so allow a
  second entry rather than adding one more fixed name.

  **opencode drives the Responses API for `openai/` providers**, which is stateful
  (`previous_response_id`, no prompt resend). That matters for
  `scripts/per_trial_tokens.py`, whose content-matching fallback behaves
  differently there than on claude-code's `messages` traffic.

  **Verify every new harness or model** with `vero/examples/harness-conformance`
  (see its `SKILL.md`) before spending a benchmark on it: check the gateway
  request log for `403 model_denied`, confirm the producer request count is
  non-zero — a zero count with a working optimizer means it found another way out
  — and confirm the `endpoint` is the one you intended.
- **Telemetry**: W&B project `harness-engineering-bench` for the whole suite
  (group per benchmark, `--param wandb_run=` for the per-launch name) with trace
  uploads; inner
  sandboxes grouped under the dedicated `harness-engineering-bench` Modal app
  with a 1h idle timeout; the gateway records a per-request log.

## Per-benchmark values

| | gaia | officeqa | swe-atlas-qna | tau3 | browsecomp-plus |
|---|---|---|---|---|---|
| target model | gpt-5.4-mini ◇ | deepseek-v4-flash | gpt-oss-120b | deepseek-v4-flash | deepseek-v4-flash |
| held-out baseline (K=3) ◆ | 0.621 ±0.052 | 0.341 ±0.033 | 0.068 ±0.026 (agg 0.632) | 0.732 ±0.010 | 0.462 ±0.028 |
| split dev/val/test | 33/66/66 | 49/98/99 | 25/49/50 | 75/150/150 | 33/66/66 |
| dev budget (runs / cases) | 100 / 132 | 100 / 196 | 100 / 100 | 100 / 300 | 100 / 132 |
| val budget (runs / cases) | 100 / 264 | 100 / 392 | 100 / 196 | 100 / 600 | 100 / 264 |
| gateway max_tokens (evaluation, finalization each) ¶ | 2 B | 3 B | 2 B | 4 B | 2 B |
| max_concurrency (cases in flight) § | 24 | 24 | 24 | 24 | 24 |
| timeout_seconds (per eval) ‖ | 7200 | 28800 | 90000 | 79200 | 39600 |
| case_timeout_seconds = declared † | 600 | 1800 | 10800 | 3600 | 3600 |
| task_agent_timeout_seconds (declared) | 600 | 1800 | 10800 | 3600 | 3600 |
| declared `[verifier] timeout_sec` | 300 | 300 | 900 | 300 | 300 |
| declared `build_timeout_sec` | 300 | 600 | 600 | 600 | 7200 |
| verifier_timeout_seconds ‖ | 14400 | 54000 | 176400 | 158400 | 75600 |
| BASH_MAX_TIMEOUT_MS (tool) ¤ | 3600 s | 10800 s | 39600 s | 32400 s | 14400 s |
| harness_user | harness | harness | null ‡ | null ‡ | null ‡ |
| task_services_use_upstream | false | false | true (rubric judge) | true (user-sim + grader) | true (answer judge) |
| task-specific extras | — | `--no-force-build` (prebuilt corpus image) | `keepalive` --ek (ENTRYPOINT images) | `TAU2_*` model pins | pinned 2.2 GB BM25 index |

## Choosing an optimizer harness

Every harness reaches the gateway a different way, and getting it wrong produces
an error that reads like a credential problem rather than a routing one. vero
handles each case in `harbor/cli.py`, so a launch needs only `--agent` and
`--model` — but the model string form is not interchangeable, and a wrong one
either bypasses the gateway or fails closed against the wrong endpoint.

| `--agent` | `--model` form | how vero routes it | proven |
|---|---|---|---|
| `claude-code` | `claude-sonnet-5` | `ANTHROPIC_BASE_URL`, scope root with the trailing `/v1` stripped, because the Anthropic SDK re-appends `/v1/messages` | yes |
| `opencode` | `anthropic/claude-sonnet-5` | provider `baseURL` written into `opencode.json`; keeps the `/v1` because opencode appends only `/messages` | yes |
| `mini-swe-agent` | `anthropic/claude-sonnet-5` | litellm aliases: `OPENAI_API_BASE` keeps `/v1` (it appends `/chat/completions`), `ANTHROPIC_API_BASE` gets the fully qualified `/v1/messages` (it appends that unless already present) | yes |
| `kimi-cli` | `openai/fireworks_ai/kimi-k3` | `--ak base_url`, written into the provider block of kimi-cli's config file | yes |
| `codex` | — | reads `OPENAI_BASE_URL`/`OPENAI_API_KEY` | **not yet run** |

Two things are specific to `kimi-cli` and easy to get wrong:

- **The doubled prefix is deliberate.** kimi-cli splits the model on the first
  `/` and rejects any provider not in its own table; `fireworks_ai` is not in it.
  The `openai/` prefix selects its `openai_legacy` provider type and leaves
  `fireworks_ai/kimi-k3` as the model, which is the only form the upstream
  resolves.
- **The gateway allow-list needs the wire form, not the prefixed one**, so pass
  `--param optimizer_model=fireworks_ai/kimi-k3` alongside `--model`. `--model`
  only fills that parameter when it is otherwise unset, so an explicit `--param`
  wins. Without it the first request is a gateway 403 `model_denied`.
- kimi-cli pins a 256K context window for the k2.5/k2.6/k2.7 families by name.
  **k3 is not in that list** and falls back to a 128K default, which may be
  below its real window; set `KIMI_MODEL_MAX_CONTEXT_SIZE` once that number is
  known.

Adding a harness means finding out how it addresses a base URL before spending a
full run on it. A cheap conformance run — short budget, one benchmark — costs
about two minutes and has caught this on every harness added so far.

## Conventions

- **Timeouts are per-phase, not one shared wall.** Harbor runs the optimizer
  agent phase and the verifier (finalization) phase with independent clocks, so
  a long search does not eat into finalization's budget and vice versa. The
  **optimizer agent phase is unbounded** (vero sets no `[agent] timeout_sec`);
  the search is governed by the agent's case budget, not wall time.
- **Gateway token caps are a runaway backstop, not the spend control** (¶). The
  work is already bounded by the agent case budget and by the fixed held-out set,
  so a cap that bites first only aborts already-authorized work. Each is sized at
  4.4–6.8M tokens per case-run depending on the benchmark (derive it from each
build's own comment): 3.3–5.1× the worst measured cost of 1.33M/case-run for an
  *optimized* officeqa candidate, itself ~3× its own baseline, because more turns
  and bigger contexts are exactly what the optimizer buys. ~90% of these tokens
  are cache reads, which count at full weight against `max_tokens`.
- **`finalization` is a reserved gateway scope and every benchmark now sets its
  budget explicitly.** Left unset it inherits `evaluation`'s *limits* as a
  separate pool of the same size — the compiler mints a finalization token
  unconditionally and the gateway keys each ledger by scope name, so search spend
  cannot deplete it. The risk of omitting it is a held-out pass funded at
  search-sized numbers, not starvation. (The starvation incident below predates
  the reserved scope.) officeqa's first full
  run exhausted the shared 100M mid-finalize and reported `reward 0.0` with
  `inference_budget_exhausted`. An optimizer exhausting its *own* evaluation
  budget is a legitimate result; trusted finalization failing on budget is an
  infrastructure bug.
- **Use each dataset's declared per-case timeout; do not invent one** (†).
  `case_timeout_seconds` is not an independently enforced wall — vero's *only*
  use of it is deriving `--agent-timeout-multiplier =
  case_timeout / task_agent_timeout` for `harbor run`. So it is a *scale factor on
  the dataset's own agent clock*, and setting it equal to the declared
  `[agent] timeout_sec` makes the multiplier exactly **1.0**: the target agent
  gets precisely the clock the benchmark intends. Every benchmark now does this.
  Set both keys explicitly — omitting them silently applies harbor's 180/600
  defaults regardless of what the tasks declare.
- **No buffer above 1.0 is warranted**, because harbor runs **four independent
  clocks per trial**, each with its own multiplier (`trial/trial.py`): the agent
  phase (declared `[agent] timeout_sec`), agent **setup**
  (`_AGENT_SETUP_TIMEOUT_SEC = 360`), **environment build** (declared
  `build_timeout_sec`), and **verification** (declared `[verifier] timeout_sec`).
  Container start, venv install, image build and scoring therefore *cannot* eat
  into the agent's budget, so there is no translation loss for a 1.1 to absorb;
  it would simply be 10% more lenient than the benchmark declares. Note the
  consequence for reading measurements: `mean_case_wall_seconds` is *whole-case*
  wall including setup and verify, so it is not directly comparable to this cap.
  `_resolve_timeout_sec` is `min(base, max_sec or ∞) × multiplier`, and
  `max_timeout_sec` defaults to `None`, so nothing clamps a large declared value.
- **The verifier must never time out, so its clocks are sized to be
  unreachable, not merely generous** (‖). A verifier timeout yields no reward at
  all — the score is lost and only agent artifacts are salvaged — which makes it
  exactly as destructive as an exhausted finalization budget. Because
  `case_timeout_seconds` caps every case, there is a hard upper bound available:

      worst-case eval = ceil(trials / max_concurrency) × case_timeout_seconds

  i.e. *every* trial timing out. `timeout_seconds` is set above that, and
  `verifier_timeout_seconds` above worst-case finalize + worst-case rescore. Both
  are therefore unreachable rather than tuned. officeqa's run #3 was killed after
  this was derived: at the previous 7200 s it would have died ~46% through a
  ~4.3 h finalize.
- **These bounds are coupled to `max_concurrency` (§).** Lowering concurrency
  raises the wave count and both timeouts must be recomputed. All five are sized
  for `max_concurrency: 24` and for `n_attempts: 3` on the held-out target, so
  raising a benchmark to a 3× finalize needs no timeout change.
- **Case budgets** are 4× the partition size, i.e. four full passes.
- **Optimizer `agent_env`** (now on all five): inner evals take 15–30 min, but
  Claude Code caps a single Bash call at `BASH_MAX_TIMEOUT_MS` (default
  600000=10min), which forces the agent into `--detach` + background-poll +
  end-turn — and in headless `--print` mode, ending the turn ends the run. Set
  `BASH_MAX_TIMEOUT_MS`/`BASH_DEFAULT_TIMEOUT_MS` above a worst-case full
  validation eval so the agent can block on one in a single call.
  `ENABLE_BACKGROUND_TASKS`/`FORCE_AUTO_BACKGROUND_TASKS=0` are defence in depth
  only — **they gate *automatic* backgrounding and do not remove the Bash tool's
  `run_in_background` parameter**, which the model can still choose, and run #2
  did. Only the optimizer instruction actually forbids it.
- **`infrastructure_max_attempts: 3`** applies only to trusted finalization
  re-scores. For competitive (agent) evaluations, whole-sub-run infrastructure
  retry is disabled and a within-trial transient-infra failure is scored at the
  failure value rather than excluded — a candidate cannot inflate its mean by
  emitting a timeout/connection error. Coverage gaps (no trial produced) and
  gateway budget/auth exhaustion remain excluded/terminating for both.

◆ Held-out baseline of the seed harness on the **test** partition, mean over
K=3 independent rounds; ± is the stdev across the three round means. Pinned
into each target's `baseline_reward` with `score_baseline: false`, which avoids
re-scoring the seed every finalization. **Note the pin is currently inert at
runtime:** the verifier reads `target.baseline_reward` only inside its
`score_baseline` branch, so with that false a run's `finalize.json` carries an
empty `baseline_rewards` and the improvement delta has to be computed offline
against this table. swe-atlas's
`reward` is a binary pass/fail over a rubric and sits near the floor (0.097);
the continuous `agg_score` (0.632, sd 0.011) is the far more informative
signal — a candidate `reward_key` switch, pending the verifier emitting
`agg_score` as a selectable key. Measured with each benchmark's target model
(deepseek-v4-flash; gpt-oss-120b on swe-atlas; gpt-5.4-mini on gaia): the three
deepseek benchmarks logged zero exceptions over 945 trials, swe-atlas lost
5/150 to gpt-oss 128k context overflow, and gaia lost 4/198 (infra) after the
agent's reason/search-only-turn crash was fixed.

**Re-pin these whenever a seed agent changes.** The first set went stale because the seed agents moved 4-10 commits afterwards -- one commit touched all five -- and nothing recorded the dependency. Re-measured 2026-07-28 with `scripts/rescore_candidate.py --seed`, which reuses the original path exactly. Only tau3 moved materially (+0.121 against an sd of 0.0099, so a genuine seed improvement); the other four shifted inside their own noise.

**gaia and tau3 are too noisy for single-run comparisons.** gaia's own three rounds spanned 0.554-0.682 (sd 0.052), and tau3's optimizer scored one *unchanged* harness at 0.800 and 0.547 on development -- its user-simulator and NL-assertion grader are both LLMs, so their variance rides on every eval. Treat a gaia or tau3 delta under ~0.1 as unresolved. Their splits are not the problem: domain mix matches to the percentage point across all three partitions (airline 13%, banking 26%, retail 30%, telecom 31%), as does telecom persona difficulty.

◇ gaia is the exception to the deepseek-v4-flash default: it is multimodal and
that model is text-only. Verified against the same litellm endpoint the gateway
proxies to — gpt-5.4-mini returns 200 for every request shape the gaia agent
sends (image input, hosted `web_search`, `reasoning.effort`,
`parallel_tool_calls`), while deepseek-v4-flash returns
`400 This model does not support image inputs`. Written unprefixed on purpose:
each agent sends `model_name.removeprefix("openai/")`, so an `openai/`-prefixed
name would be allow-listed in one form and requested in another and the gateway
would deny it.

† **Taken from the dataset, not chosen by us.** Each value is that dataset's
declared `[agent] timeout_sec`, read from its `task.toml` — the hub packages under
`~/.cache/harbor/tasks/packages/` for gaia (`gaia/gaia`), swe-atlas-qna
(`scale-ai`) and tau3 (`sierra-research`), and the vendored task dirs for officeqa
and browsecomp-plus. Every dataset declares **uniformly** across its tasks (246
officeqa tasks all at 1800, 830 browsecomp tasks all at 3600), so a single value
per benchmark loses nothing.

This supersedes two earlier rounds of sizing-by-measurement. The first was a
codex probe, far too tight for the real target agents. The second used
held-out-baseline wall-time distributions (gaia p99≈608 → 900; officeqa
p99≈640/max≈1076 → 1200; swe-atlas → 1800; tau3 p99≈643/max≈1122 → 1200;
browsecomp p99≈1479/max≈1771 → 2100), which fixed the unfairness — the prior
180/300/900 caps would have killed ~9/13/26% of gaia/officeqa/browsecomp candidate
cases — but still second-guessed the benchmark. The deviations were large and
inconsistent in direction: swe-atlas ran at **0.17×** its declared clock while
gaia ran at **1.5×**. Adopting the declared value makes gaia *tighter* (900 → 600,
newly clipping ~1% of cases whose p99 sat at 608); that is the benchmark's
intent, and suggestively the gaia agent's own `MAX_TURNS` cap lands right at
~608 s, i.e. it was written against the declared 600.

¤ Bash tool cap for the optimizer, set above that benchmark's widest single
blocking eval — a full validation pass, `ceil(val_cases / 24) × case_timeout`
worst case. `BASH_DEFAULT_TIMEOUT_MS` is set equal to it so an eval invoked
without an explicit `timeout` still blocks rather than being truncated: a
truncated eval is what pushed run #2's optimizer into backgrounding and ended the
run. The agent additionally wraps its own calls (`timeout 1750`, `timeout 3000`
observed), so this is a ceiling, not the expected duration.

¶ `evaluation` and `finalization` each get this cap independently — they are
separate scopes with separate tokens and separate ledgers, so the numbers do not
share a pool. `max_requests` is 200 000 on both everywhere (a full officeqa
finalize needs ~12 000, so this is not the binding constraint either). Sizing is
per benchmark from its own case counts; see each `build.yaml` for the arithmetic.

§ **Concurrency scales with the number of LiteLLM keys.** The binding ceiling is
the per-key bucket, so more keys buy proportionally more parallel runs:

| counter | limit | scope |
|---|---|---|
| `x-ratelimit-api_key-limit-tokens` | 10 M TPM | **per key** |
| `x-ratelimit-api_key-limit-requests` | 5 000 RPM | **per key** |

One officeqa run at `max_concurrency: 24` peaks at 2.90 M metered TPM and ~182
RPM, so **roughly 3 concurrent runs per key**, TPM-bound. Two keys comfortably
cover the suite; three leave headroom.

**Do not read the `llm_provider-x-ratelimit-*` headers as a shared provider
budget.** They are the provider's limits *echoed per request*, not a depleting
meter. Proof: `remaining-tokens-prompt` reads `14062495` both on an idle probe and
while ~120 cases were running — byte-identical, and in both cases exactly
`limit − 5`, where 5 is that one probe's own prompt tokens. Meanwhile the
`x-ratelimit-api_key-*` counters moved as expected under the same load
(4999 → 4863 requests, 10 M → 9.52 M tokens). An earlier revision of this note
mistook those headers for a Fireworks account bucket and concluded the suite was
capped near 2 concurrent runs regardless of keys; that was wrong, and the
per-request arithmetic above is what disproves it.

Give each concurrent run its own `--env-file` so the per-key buckets are actually
independent and spend stays attributable: copy `secrets.env.example` to a
`*.secrets.env` name (that glob is gitignored; `secrets.2.env` is **not**). Modal
sandbox capacity is a separate ceiling. Target models still differ per benchmark
(gaia is the only non-Fireworks one), which spreads load across providers as a
side benefit, but it is not the constraint. The single `502 upstream_error` seen
so far is 1 in 4 750 requests at 24 slots.

`max_concurrency` is concurrent *trials* per evaluation — it becomes harbor's
`-n`, and with `n_attempts: 3` on the test target 24 trials is 8 cases in flight.
Raised 8 → 24 across the
board. It is the throughput lever: finalize wall time is
`ceil(trials / max_concurrency) × mean case wall`, so officeqa's ~4.1 h finalize
becomes ~1.4 h. Headroom is measured, not assumed — officeqa run #2 sustained
**16 case slots** (two detached evals × 8) at ~181 k metered TPM per slot with
`upstream_errors: 0`, so 24 slots is ~1.5× a proven configuration. Per-slot
throughput ranges ~181–361 k metered TPM depending on how context-heavy the
candidate is. When raising further, check `upstream_errors` in
`artifacts/inference/usage.json` — that measures the provider's ceiling in the
provider's own accounting, which is the only way to settle whether its quota
counts cache reads.

‖ Both eval clocks are sized to be **unreachable** given
`case_timeout_seconds` × the wave count (see Conventions), not to a multiple of
an expected duration. `timeout_seconds` bounds any single evaluation —
`evaluation/evaluator.py` wraps each in `asyncio.timeout(limits.timeout_seconds)`
— and applies to the agent's own evals as well as the finalize.
`verifier_timeout_seconds` bounds the whole verifier phase: worst-case finalize
plus worst-case `rescore_top_k=3` validation rescore, which is headroom since the
rescore does not fire in the `submit` selection path. Because phases are
independent, neither covers the search.

‡ Exception to the harness-isolation default: these tasks run LLM services
(rubric judge, user-simulator/grader, or answer judge) inside their task
containers, which
cannot reach the compose-internal gateway, so the build hands the real
upstream credential to the task environment via `task_services_use_upstream`.
That credential path is incompatible with `harness_user` isolation (the key
would sit in the isolated harness's environment), so these benchmarks run
unisolated under a non-adversarial-optimizer assumption, with a post-run
leakage audit. Restore isolation once task-service credentials are delivered
off the harness env (scoped judge key or per-role egress isolation).
