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
| held-out baseline (K=3) ◆ | — | 0.360 ±0.042 | 0.097 ±0.011 (agg 0.632) | 0.611 ±0.021 | 0.449 ±0.007 |
| split dev/val/test | 33/66/66 | 49/98/99 | 25/49/50 | 75/150/150 | 33/66/66 |
| dev budget (runs / cases) | 100 / 132 | 100 / 196 | 100 / 100 | 100 / 300 | 100 / 132 |
| val budget (runs / cases) | 100 / 264 | 100 / 392 | 100 / 196 | 100 / 600 | 100 / 264 |
| timeout_seconds (per eval) | 3600 | 7200 | 14400 | 14400 | 28800 |
| case_timeout_seconds (enforced) | 180 | 300 | 1800 † | 900 † | 900 † |
| task_agent_timeout_seconds (declared) | 600 | 1800 | 10800 | 3600 | 3600 |
| verifier_timeout_seconds | 7200 | 14400 | 28800 | 28800 | 57600 |
| harness_user | harness | harness | null ‡ | null ‡ | null ‡ |
| task_services_use_upstream | false | false | true (rubric judge) | true (user-sim + grader) | true (answer judge) |
| task-specific extras | — | `--no-force-build` (prebuilt corpus image) | `keepalive` --ek (ENTRYPOINT images) | `TAU2_*` model pins | pinned 2.2 GB BM25 index |

## Conventions

- **Timeouts**: `task_agent_timeout_seconds` mirrors the agent timeout the
  dataset's task packages declare; `case_timeout_seconds` is the per-case
  budget VeRO actually enforces (compiled to Harbor's agent-timeout
  multiplier). Set both explicitly — omitting them silently applies the
  180/600 defaults regardless of what the tasks declare.
- **Verifier timeout** is 2× `timeout_seconds`: finalization runs the
  candidate and the baseline test evaluations.
- **Case budgets** are 4× the partition size, i.e. four full passes.
- **`infrastructure_max_attempts: 3`** applies only to trusted finalization
  re-scores. For competitive (agent) evaluations, whole-sub-run infrastructure
  retry is disabled and a within-trial transient-infra failure is scored at the
  failure value rather than excluded — a candidate cannot inflate its mean by
  emitting a timeout/connection error. Coverage gaps (no trial produced) and
  gateway budget/auth exhaustion remain excluded/terminating for both.

◆ Held-out baseline of the seed harness on the **test** partition, mean over
K=3 independent rounds; ± is the stdev across the three round means. Pinned
into each target's `baseline_reward` with `score_baseline: false`, so runs use
this number instead of re-scoring the seed every finalization. gaia is not yet
measured (deferred). swe-atlas's `reward` is a binary pass/fail over a rubric
and sits near the floor (0.097); the continuous `agg_score` (0.632, sd 0.011)
is the far more informative signal — a candidate `reward_key` switch, pending
the verifier emitting `agg_score` as a selectable key. Measured with
deepseek-v4-flash (gpt-oss-120b on swe-atlas); the three deepseek benchmarks
logged zero exceptions over 945 trials, swe-atlas lost 5/150 to gpt-oss 128k
context overflow.

◇ gaia is the exception to the deepseek-v4-flash default: it is multimodal and
that model is text-only. Verified against the same litellm endpoint the gateway
proxies to — gpt-5.4-mini returns 200 for every request shape the gaia agent
sends (image input, hosted `web_search`, `reasoning.effort`,
`parallel_tool_calls`), while deepseek-v4-flash returns
`400 This model does not support image inputs`. Written unprefixed on purpose:
each agent sends `model_name.removeprefix("openai/")`, so an `openai/`-prefixed
name would be allow-listed in one form and requested in another and the gateway
would deny it.

† Sized from stock-agent probes (codex on the target model, 3 development
tasks each, full declared timeouts): tau3 trials took 202-211s (900s budget
≈ 4x headroom) and swe-atlas trials 379-602s (1800s ≈ 3x headroom over the
slowest). BrowseComp-Plus provisionally uses the tau3 limit. Revisit against
`wall_seconds` distributions from real runs.

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
