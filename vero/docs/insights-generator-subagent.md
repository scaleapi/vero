# Insights Generator as a VeRO optimizer sub-agent

**Status:** design / not yet implemented
**Audience:** an engineer or agent picking this up cold
**Working branch:** `native-runner-sandboxagent` (`/Users/varun.ursekar/code/vero-v05`)

---

## 1. Motivation

The VeRO optimizer improves a candidate by having a coding agent iterate on it and
scoring the result. Across a run it accumulates a large corpus of **execution
traces** (one per evaluated case, per candidate). Today nothing systematically
mines that corpus: patterns that are only visible across the *population* of runs
(e.g. "40% of failing cases silently swallow a specific exception") never surface
back into the optimizer's next proposal.

Scale's **Insights Generator (IG)** is a system built exactly for this: *systematic
corpus-level trace diagnostics for LLM agents* (paper: arXiv `2605.21347v3`,
"Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents",
Scale AI). It reads a corpus of traces + a diagnostic question and returns an
evidence-backed report of failure patterns with prevalence stats and cited trace
IDs. In their own eval it roughly doubled practitioner effectiveness over
baselines.

**The endeavour:** give the VeRO optimizer IG's capability *as a sub-agent it can
invoke* — so that between rounds it can ask "what distinguishes high- from
low-scoring candidates in this corpus?" and feed the answer into the next
generation.

## 2. The hard constraint: reproduce, do not depend

VeRO is **public**; IG is **internal/private** (`scaleapi/models`,
`enterprise/insights_generator/`). We therefore **cannot take a dependency on IG**.
We reproduce IG's *mechanisms*, not its code, using only things a public repo can
ship.

We also do **not** need IG's heavier implementation choices:
- no pandas dataframe input contract,
- no in-sandbox Python REPL,
- no vector store / schema-cache / parquet data layer,
- no Scale Sandbox.

The substrate is **the sandbox filesystem + bash**, which the Harbor path already
gives us.

## 3. What IG actually is (stripped to essentials)

Three mechanisms do the real work; everything else is implementation:

1. **Corpus-scale coverage without loading everything into context.** Compute
   *over* all traces; read the full text of only a sampled/relevant few.
2. **Discovery ≠ validation (Scout / Investigator split).** *Scouts* skim a sample
   and propose many hypotheses (breadth). *Investigators* test each one against the
   *whole* corpus with quantitative evidence and label it confirmed / refuted /
   inconclusive (depth). No finding ships without corpus-scale backing.
3. **Evidence-grounded, typed output with cohort persistence.** Every finding =
   description + prevalence % + specific trace IDs + verbatim quotes + mechanism +
   fix. IG literally rejects a finding that has no saved "affected traces" cohort.

> Note the two entry points in the IG repo are **not** equivalent. The public-ish
> `generate_insights()` in `analysis.py` is the **old single-pass** path (sample →
> format as XML → one LLM call) — closer to the baseline the paper *beats*. The
> "best-performing" system is the agentic orchestrator entered via
> `run_orchestrator(question, sandbox, config)`.

## 4. The path that matters: Harbor, and the real `claude` CLI

VeRO has two agent worlds. **Only the Harbor path is in scope.**

- **Out of scope — VeRO's own SDK agents.** `VeroAgent` (OpenAI Agents SDK) and
  `ClaudeCodeAgent` (Claude Agent SDK). These have their own `SubAgentTool`, but
  they are a *different* execution model.
- **In scope — Harbor's `ClaudeCode` agent.** This is what `harbor run -a claude`
  uses. Crucially, it is **not** the Claude Agent SDK — it installs the real
  `@anthropic-ai/claude-code` CLI into the sandbox and runs it **headless**:
  `claude --print --output-format=stream-json -- <instruction>`, as the `agent`
  user, in the environment filesystem.

Because the optimizer is the **real Claude Code CLI**, IG plugs in through Claude
Code's **native filesystem/config extension points** — subagents, skills, MCP — not
through Python objects. Harbor already *expects* subagents: it filters `subagents`
session paths and parses `isSidechain` events into its trajectory.

## 5. The design

**"The whole IG machinery as a sub-agent" = a Claude Code sub-agent bundle seeded
into the Harbor environment.** The optimizer invokes it with the `Task` tool.

### Role mapping (1:1 with IG)

| IG role | Reproduction |
|---|---|
| Orchestrator | `.claude/agents/insights-generator.md` (has `Task`) |
| Hypothesis agent (Scout) | `.claude/agents/ig-scout.md` — breadth, cheap model |
| Explore agent (Investigator) | `.claude/agents/ig-investigator.md` — depth, high effort |

Each sub-agent runs in its **own context window** — which is exactly IG's
context-economy property (its orchestrator delegates heavy trace-reading to
preserve context). We get it for free from Claude Code's `Task` model.

### Tool / data-layer mapping

| IG piece | Reproduction |
|---|---|
| `search_traces` (vector) | `grep`/`rg` over the corpus dir (lexical) |
| `extract` (feature per trace) | a script emitting `features.jsonl`, persisted + reused |
| `compare_segments` (cohort diff) | `jq`/`awk` group-by over `features.jsonl` by label |
| `get_trace` / `get_trace_chunk` | `cat` / `sed -n` a slice of a trace file |
| `save_affected_traces` | write `cohorts/<hypothesis>.txt` |
| `submit_finding` + rejection rule | a `findings.json` contract + a `validate.py` check |
| data layer (parquet/vector/schema) | **the sandbox filesystem** |
| Scale Sandbox | **the Harbor container the optimizer already runs in** |

### Delivery seams (how the bundle gets into the container)

Harbor auto-injects two of these; you seed the third yourself:

| What | How it lands in the container | Wired by |
|---|---|---|
| **Skill** (methodology + helper scripts) | `environment.skills_dir` → copied to `$CLAUDE_CONFIG_DIR/skills/` | `task.toml` |
| **MCP server** (optional real-tool surface) | `environment.mcp_servers` / `agent.mcp_servers` → `~/.claude.json` | `task.toml` / agent config |
| **Sub-agent defs** (`.claude/agents/*.md`) | **no dedicated Harbor seam** — bake into the env image/seed | your `Dockerfile` / `seed.sh` |

The last row is the one gotcha: Harbor injects skills/mcp/memory but **not** an
agents dir, so seed `.claude/agents/*.md` via the environment build (into the
agent's `cwd/.claude/agents/` or `$CLAUDE_CONFIG_DIR/agents/`).

### Proposed layout

```
environment/
  Dockerfile                      # COPY .claude/agents -> /work/.claude/agents
  main/seed.sh                    # seed /work; also lay out the trace corpus
  .claude/agents/
    insights-generator.md         # orchestrator persona (Task, Bash, Read, Grep, Glob)
    ig-scout.md                   # sample -> hypotheses
    ig-investigator.md            # validate one hypothesis over ALL traces
  skills/insights/                # -> environment.skills_dir
    SKILL.md                      # evidence-grounding rules, findings.json contract
    chunk.sh                      # slice long traces (the one primitive bash lacks)
    validate.py                   # reject findings without cohort + trace-id citations
```

```toml
# task.toml
[environment]
skills_dir = "/skills"
# mcp_servers = [ ... ]   # only if a query-server is preferred over raw bash
```

### The corpus

VeRO already dumps per-case traces into a read-only `.evals` context dir:
`.evals/results/<digest>/cases/<case_digest>/execution-trace.json` (and
`evaluation-trace.json`). That **is** the IG corpus. Label traces by the eval
scores VeRO already has (pass/fail, reward) so the investigator can contrast
cohorts — comparative analysis is where IG earns its keep.

### Where it plugs into the loop

Between rounds, the optimizer calls `Task(subagent_type="insights-generator", …)`
pointed at the current round's traces; the findings feed the next candidate
proposals. This is a native reproduction of IG's own `patcher` loop
(analyze → guide fix → re-eval), and it lines up with VeRO's evolutionary strategy
(findings become the guidance signal for the next generation of agentic offspring).

## 6. Design decisions & rationale

1. **Reproduce, don't import.** VeRO public / IG private. Non-negotiable.
2. **Target the Harbor path (real `claude` CLI), not the SDK agents.** That is the
   optimizer we ship; it makes the integration filesystem/config, not Python.
3. **IG = Claude Code sub-agent(s) via `Task`.** Faithful to IG's LLM-driven
   orchestration and reproduces its per-role context isolation for free.
4. **Filesystem + bash as the data layer / query engine.** Drops the dataframe,
   REPL, vector store, and Scale Sandbox in one move.
5. **Preserve the evidence discipline explicitly.** A `findings.json` contract plus
   a `validate.py` check reproduce `submit_finding`'s "no cohort → reject" rule;
   without it the output degrades to vibes.
6. **Start nested (orchestrator + scout + investigator).** Maps 1:1 to IG and
   exploits context isolation. A single phased sub-agent is a legitimate v0 but
   weaker on context economy for large corpora.

## 7. Known caveats

- **Chunking long traces is the one real primitive to build.** A single
  `execution-trace.json` can blow a sub-agent's context; provide `chunk.sh` (byte-
  offset or turn-boundary slicing). Everything else degrades gracefully to grep.
- **No semantic search.** `grep` is lexical. Fine for error strings, tool names,
  phrasings (most trace diagnostics). Add a small local embed helper or an MCP
  query-server only if recall disappoints.
- **Model control.** Harbor sets `CLAUDE_CODE_SUBAGENT_MODEL` when a custom base
  URL is used; set per-role `model`/`effort` in each agent `.md`.

## 8. Code pointers

### VeRO (`/Users/varun.ursekar/code/vero-v05/vero`, this repo)

| Path | Why it matters |
|---|---|
| `src/vero/runtime/context.py` (~L256–273) | Where per-case `execution-trace.json` / `evaluation-trace.json` are written into the `.evals` context — **the corpus source** |
| `src/vero/optimization/optimizer.py` | `Optimizer` (L204); `run` (L409); `_produce_candidate` (L283) — where an IG call would slot between rounds |
| `src/vero/agents/producer.py` | `AgentCandidateProducer.produce()` — how a coding agent is driven per proposal |
| `src/vero/agents/claude_code.py` | VeRO's **SDK** Claude agent — **out of scope**, but note L44 enables `Task`; do not confuse with Harbor's agent |
| `src/vero/agents/vero.py` | `VeroAgent` (OpenAI Agents SDK) — out of scope |
| `src/vero/tools/sub_agent.py` | `SubAgentTool` — the SDK-world sub-agent mechanism (not the Harbor path) |
| `src/vero/harbor/` | Harbor integration: `backend.py`, `cli.py` (`run` command), `sidecar.py`, `verifier.py` |
| `examples/harbor-circle-packing/` | Reference Harbor task: `task.toml`, `environment/Dockerfile`, `environment/main/seed.sh`, `environment/sidecar/` — the template to clone for an insights task |
| `docs/agent-setup-guide.md` | Existing agent setup guide |

### Harbor (installed from public PyPI; import path `harbor.*`)

Locate with `python -c "import harbor, os; print(os.path.dirname(harbor.__file__))"`
inside the vero venv.

| Module | Why it matters |
|---|---|
| `harbor/agents/installed/claude_code.py` | **The real optimizer.** `run()` runs `claude --print --output-format=stream-json`. `_build_register_skills_command` (skills → `$CLAUDE_CONFIG_DIR/skills/`), `_build_register_mcp_servers_command` (→ `~/.claude.json`), `_build_register_memory_command`. `_get_session_dir` / `_convert_events_to_trajectory` already handle `subagents` + `isSidechain` |
| `harbor/agents/installed/base.py` | `BaseInstalledAgent`: `__init__`, `build_cli_flags`, `exec_as_agent`, `with_prompt_template` |
| `harbor/agents/installed/` (dir) | Other agents (`codex.py`, `gemini_cli.py`, …) — same pattern if a different optimizer is used |
| `harbor/trial/trial.py` | `_resolve_effective_skills_dir` (~L1049); `mcp_servers` merge (~L738–748) — how the seams are wired from config |
| `harbor/models/task/config.py` | Task `[environment]` schema: `skills_dir` (~L450), `mcp_servers` (~L444) |
| `harbor/models/trial/config.py` | Agent config: `mcp_servers` (~L125) |
| `harbor/models/trial/paths.py` | `EnvironmentPaths`: `agent_dir`, `default_skills_dir = /harbor/skills` |

### Insights Generator (private — `scaleapi/models`, reference only, **do not import**)

Path: `enterprise/insights_generator/src/insights_generator/`

| Module | Why it matters |
|---|---|
| `agents/orchestrator.py` | `run_orchestrator(question, sandbox, config)` (~L1044) — the real agentic entry point (what we reproduce) |
| `agents/hypothesis_agent.py` | `run_hypothesis_agent` — the Scout; source for `ig-scout.md` |
| `agents/explore_agent.py` | `run_explore_agent` — the Investigator; source for `ig-investigator.md` |
| `agents/dispatch.py` | `DispatchHypothesisAgents` (~L216) / `DispatchExploreAgents` (~L455) — orchestrator dispatches roles via tool calls |
| `agents/prompts/{orchestrator,hypothesis,explore}.py` | The role prompts (23–54 KB) — **distill these into the `.md` personas** |
| `agents/prompts/_compose.py` | Jinja `AblationContext` prompt composition (we drop ablations; use `ablation_mode="full"` = orchestrator + hypothesis + explore) |
| `tools/repl_toolkit.py` | `EXPLORE_FUNCTIONS` (search_traces / extract / compare_segments / summarize / get_trace / save_affected_traces) — the tool surface we re-map to bash |
| `ingestion.py` | `ingest(...)` + the INPUT CONTRACT (`trace_id`, flattened `content`) — the trace-flattening reference |
| `analysis.py` | `generate_insights()` — the **old single-pass** path; NOT the system we want |
| `ig_config.py` | `IGConfig` — `ablation_mode`, per-role model fields |
| `vector_store.py`, `chunking.py`, `schema_cache.py` | The data layer we deliberately do not reproduce (except chunking) |

## 9. Next step (offered, not yet done)

Draft the three agent `.md` files + `SKILL.md` / `findings.json` contract +
`chunk.sh` / `validate.py`, distilled from IG's actual scout/investigator/
orchestrator prompts and grounded against a real `execution-trace.json` from the
`.evals` layout, so the bundle is runnable rather than notional.
