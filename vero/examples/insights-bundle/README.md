# insights-bundle — Insights Generator as an optimizer subagent

This is **not a Harbor task.** It is a bundle you *inject* into an existing
optimizer run (e.g. the GAIA build under `program-opt-bench/gaia/`) so the main
solver/optimizer agent can call it — mid-run, as a subagent — to mine its own
accumulated `.vero` trace corpus for patterns and feed the findings into the next
candidate proposal.

It reproduces Scale's internal **Insights Generator** *methodology* — not its
code (VeRO ships publicly; IG is internal). Everything runs on the sandbox
filesystem + bash; there is no IG dependency, no dataframe contract, no Python
REPL requirement.

## What's in it

```
.claude/agents/
  insights-generator.md   # Orchestrator: dispatches scouts + investigators, synthesizes findings
  ig-scout.md             # Discovery (breadth): skims a sample, proposes hypotheses
  ig-investigator.md      # Validation (depth): tests one hypothesis over the whole corpus, saves a cohort
skills/insights/
  SKILL.md                # Method + corpus layout + IG-tool→bash cookbook + finding contract
  chunk.sh                # Byte-windowed trace reader (get_trace_chunk over the FS)
  validate.py             # Enforces the finding contract — rejects a finding with no saved cohort
```

Each `.md` subagent gets its **own context window** — that reproduces IG's core
context-economy property natively: the corpus is mined across many agents, none of
which loads it whole. Harbor already captures subagent (`isSidechain`)
trajectories, so the whole investigation is recorded.

## How it plugs into a run

The optimizer works in `/work/agent`; its `vero harbor eval` calls write the
trace corpus to `/work/agent/.vero/evaluations/**`. Point the bundle there with
`CORPUS=/work/agent/.vero`. Two seams inject the bundle into the environment:

1. **Subagents** — Claude Code has no dedicated Harbor seam for `.claude/agents/`.
   Seed them via the environment `Dockerfile`/`seed.sh` into the project dir the
   optimizer runs in. In a circle-packing-style environment, add to the Dockerfile:

   ```dockerfile
   COPY insights-bundle/.claude/agents /opt/insights-agents
   COPY insights-bundle/skills          /opt/insights-skills
   ```
   and in `main/seed.sh`, before `exec sleep infinity`:
   ```sh
   mkdir -p /work/agent/.claude/agents /work/agent/skills
   cp -a /opt/insights-agents/.  /work/agent/.claude/agents/
   cp -a /opt/insights-skills/.  /work/agent/skills/
   ```

2. **Skill** — alternatively (or additionally) deliver the skill through Harbor's
   official seam: `task.toml [environment] skills_dir = "skills"`, which Harbor
   copies to `$CLAUDE_CONFIG_DIR/skills/`. Seeding it into `/work/agent/skills/`
   via the Dockerfile (option 1) is the simpler single-mechanism path and keeps
   the skill next to the agents.

**Run the optimizer as `claude`.** Subagents (`.claude/agents/`), skills, and the
`Task` tool are Claude Code features — inject this into a `harbor run -a claude`
optimizer, not `-a codex`. (Our GAIA shakedown used `-a codex`; to use IG you flip
the optimizer adapter to `claude`. The gateway per-scope allow-list still governs
which model the optimizer may call.)

## The loop it enables

```
optimize round N:
  optimizer proposes/edits candidate → `vero harbor eval` → traces land in .vero
  optimizer (or a between-rounds step) invokes the `insights-generator` subagent:
      orchestrator → scouts (discover) → investigators (validate) → findings.json
  optimizer reads the top findings → informs the round N+1 edit
```

This is the native reproduction of IG's patcher loop: insights don't sit in a
report, they steer the next candidate.

## Known limitations vs. real IG
- **Lexical, not semantic, search.** `grep`/`rg` replace IG's hybrid
  `search_traces`. The omission-search tactic (grep for what a correct agent
  *would* have said, then contrast cohorts — see SKILL/ig-scout) recovers much of
  the value; add a small embedding helper or MCP if you need true semantic recall.
- **Chunking is the one real primitive.** Long traces are read via `chunk.sh`
  byte windows; use `jq` on the whole file when you need JSON structure.
- **Trace format is backend-specific.** `execution-trace.json` is a JSON array of
  turns/steps whose inner shape depends on the agent adapter — grep the text;
  don't assume a fixed schema.

See `skills/insights/SKILL.md` for the full method and cookbook.
