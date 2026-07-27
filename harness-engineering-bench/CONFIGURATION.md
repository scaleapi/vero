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
  is on test; opt-in only). `score_baseline: true`, `rescore_top_k: 3`,
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
  allow-list, so a candidate cannot swap it). The optimizer uses a separate
  producer scope bound to `${optimizer_model:-openai/gpt-5.4}`. The gateway
  matches the requested model against the allow-list as an exact string, so the
  `-m` the outer trial is launched with has to be spelled the same way as this
  default (or as whatever `--param optimizer_model=` overrides it with);
  the router resolves both `gpt-5.4` and `openai/gpt-5.4`, so the prefix is a
  convention rather than a requirement. deepseek-v4-flash was
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
- **Execution**: `harbor[modal]==0.20.0`, python 3.12, `n_attempts: 1`,
  `max_retries: 1`, 3 infrastructure attempts at 5s, `aggregate_attempts:
  best`, `max_concurrency: 8`, `error_rate_threshold: 0.1`,
  `feedback_transcripts: true` with `feedback_max_bytes: 16000`,
  `environment_name: ${inner_env:-modal}` (pass `--param inner_env=docker`
  for local shakedowns).
- **Telemetry**: W&B project `vero-<benchmark>` with trace uploads; inner
  sandboxes grouped under the dedicated `harness-engineering-bench` Modal app
  with a 1h idle timeout; the gateway records a per-request log.

## Per-benchmark values

| | gaia | officeqa | swe-atlas-qna | tau3 | browsecomp-plus |
|---|---|---|---|---|---|
| target model | gpt-5.4-mini ◇ | deepseek-v4-flash | gpt-oss-120b | deepseek-v4-flash | deepseek-v4-flash |
| held-out baseline (K=3) ◆ | 0.574 ±0.010 | 0.360 ±0.042 | 0.097 ±0.011 (agg 0.632) | 0.611 ±0.021 | 0.449 ±0.007 |
| split dev/val/test | 33/66/66 | 49/98/99 | 25/49/50 | 75/150/150 | 33/66/66 |
| dev budget (runs / cases) | 100 / 132 | 100 / 196 | 100 / 100 | 100 / 300 | 100 / 132 |
| val budget (runs / cases) | 100 / 264 | 100 / 392 | 100 / 196 | 100 / 600 | 100 / 264 |
| gateway max_tokens (evaluation, finalization each) ¶ | 2 B | 3 B | 2 B | 4 B | 2 B |
| max_concurrency (cases in flight) § | 24 | 24 | 24 | 24 | 24 |
| timeout_seconds (per eval) ‖ | 10800 | 21600 | 18000 | 28800 | 28800 |
| case_timeout_seconds (enforced) † | 900 | 1200 | 1800 | 1200 | 2100 |
| task_agent_timeout_seconds (declared) | 600 | 1800 | 10800 | 3600 | 3600 |
| verifier_timeout_seconds ‖ | 28800 | 43200 | 36000 | 64800 | 64800 |
| harness_user | harness | harness | null ‡ | null ‡ | null ‡ |
| task_services_use_upstream | false | false | true (rubric judge) | true (user-sim + grader) | true (answer judge) |
| task-specific extras | — | `--no-force-build` (prebuilt corpus image) | `keepalive` --ek (ENTRYPOINT images) | `TAU2_*` model pins | pinned 2.2 GB BM25 index |

## Conventions

- **Timeouts are per-phase, not one shared wall.** Harbor runs the optimizer
  agent phase and the verifier (finalization) phase with independent clocks, so
  a long search does not eat into finalization's budget and vice versa. The
  **optimizer agent phase is unbounded** (vero sets no `[agent] timeout_sec`);
  the search is governed by the agent's case budget, not wall time.
- **Gateway token caps are a runaway backstop, not the spend control** (¶). The
  work is already bounded by the agent case budget and by the fixed held-out set,
  so a cap that bites first only aborts already-authorized work. Each is sized at
  ~3M tokens per case-run: ~2.3× the worst measured cost of 1.33M/case-run for an
  *optimized* officeqa candidate, itself ~3× its own baseline, because more turns
  and bigger contexts are exactly what the optimizer buys. ~90% of these tokens
  are cache reads, which count at full weight against `max_tokens`.
- **`finalization` is a reserved gateway scope and every benchmark now sets its
  budget explicitly.** Left unset it silently inherits `evaluation`'s numbers, and
  a search-phase overspend then starves held-out scoring: officeqa's first full
  run exhausted the shared 100M mid-finalize and reported `reward 0.0` with
  `inference_budget_exhausted`. An optimizer exhausting its *own* evaluation
  budget is a legitimate result; trusted finalization failing on budget is an
  infrastructure bug.
- **`case_timeout_seconds` is the enforced per-case wall cap**;
  `task_agent_timeout_seconds` is the task-declared agent timeout used only as
  the rescale denominator (Harbor's per-case timeout = declared ×
  `case_timeout/task_agent_timeout`). `case_timeout` may exceed
  `task_agent_timeout`. Set both explicitly — omitting them silently applies the
  180/600 defaults regardless of what the tasks declare.
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
into each target's `baseline_reward` with `score_baseline: false`, so runs use
this number instead of re-scoring the seed every finalization. swe-atlas's
`reward` is a binary pass/fail over a rubric and sits near the floor (0.097);
the continuous `agg_score` (0.632, sd 0.011) is the far more informative
signal — a candidate `reward_key` switch, pending the verifier emitting
`agg_score` as a selectable key. Measured with each benchmark's target model
(deepseek-v4-flash; gpt-oss-120b on swe-atlas; gpt-5.4-mini on gaia): the three
deepseek benchmarks logged zero exceptions over 945 trials, swe-atlas lost
5/150 to gpt-oss 128k context overflow, and gaia lost 4/198 (infra) after the
agent's reason/search-only-turn crash was fixed.

◇ gaia is the exception to the deepseek-v4-flash default: it is multimodal and
that model is text-only. Verified against the same litellm endpoint the gateway
proxies to — gpt-5.4-mini returns 200 for every request shape the gaia agent
sends (image input, hosted `web_search`, `reasoning.effort`,
`parallel_tool_calls`), while deepseek-v4-flash returns
`400 This model does not support image inputs`. Written unprefixed on purpose:
each agent sends `model_name.removeprefix("openai/")`, so an `openai/`-prefixed
name would be allow-listed in one form and requested in another and the gateway
would deny it.

† Sized from the **held-out baseline** per-case wall-time distributions (the seed
target agent on each benchmark's target model), set at or above each benchmark's
observed max: gaia p99≈608/max≈609 → 900; officeqa p99≈640/max≈1076 → 1200;
swe-atlas p99≈1163/max≈1278 → 1800 (unchanged); tau3 p99≈643/max≈1122 → 1200;
browsecomp p99≈1479/max≈1771 → 2100. This replaces the earlier codex-probe
sizing, which was too tight for the real target agents — the prior caps
(180/300/900) would have killed ~9/13/26% of gaia/officeqa/browsecomp candidate
cases, scoring candidates far harsher than the leniently-measured baselines.

¶ `evaluation` and `finalization` each get this cap independently — they are
separate scopes with separate tokens and separate ledgers, so the numbers do not
share a pool. `max_requests` is 200 000 on both everywhere (a full officeqa
finalize needs ~12 000, so this is not the binding constraint either). Sizing is
per benchmark from its own case counts; see each `build.yaml` for the arithmetic.

§ `max_concurrency` is cases in flight per evaluation, raised 8 → 24 across the
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
