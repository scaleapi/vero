# Agentic Patterns Catalog

A distillation of abstract patterns for AI agent systems. Organized by category with citations to source patterns.

---

## I. Reasoning & Search Patterns

### Chain-of-Thought (CoT)

Generate intermediate reasoning steps before producing final answer.

- **Zero-shot**: Add "Let's think step by step"
- **Few-shot**: Show worked examples with explicit reasoning
- **Key insight**: CoT buys "compute time" for the model to resolve dependencies before acting

### Tree-of-Thought (ToT)

Explore a search tree of intermediate "thoughts" instead of a single chain. Expand multiple possible steps, evaluate partial solutions, prune unpromising branches.

```python
queue = [root_problem]
while queue:
    thought = queue.pop()
    for step in expand(thought):
        score = evaluate(step)
        queue.push((score, step))
select_best(queue)
```

**Use when**: Tasks benefit from exploring multiple strategies—puzzles, planning, code generation. [tree-of-thought-reasoning]

### Graph-of-Thought (GoT)

Extend ToT to directed graphs where nodes represent thoughts and edges represent transformations. Supports:

- **Branching**: Generate multiple thoughts from one
- **Aggregation**: Combine insights from multiple paths
- **Refinement**: Improve thoughts based on later insights
- **Looping**: Revisit and refine iteratively

**Use when**: Complex interdependencies between reasoning steps that don't fit linear or tree structures. [graph-of-thoughts]

### Language Agent Tree Search (LATS)

Combine Monte Carlo Tree Search with LLM reflection:

1. **Selection**: Traverse tree using UCB (Upper Confidence Bound)
2. **Expansion**: Generate possible actions
3. **Simulation**: Evaluate using LLM self-reflection
4. **Backpropagation**: Update values up the tree

**Key insight**: Balances exploration of new paths with exploitation of promising ones. [language-agent-tree-search-lats]

### Self-Discover

LLM composes task-specific reasoning structures from atomic modules:

1. **Task Analysis**: Understand problem requirements
2. **Strategy Selection**: Choose relevant reasoning modules ("break into steps", "consider edge cases", etc.)
3. **Structure Composition**: Organize into coherent reasoning plan
4. **Execution**: Solve using discovered structure

**Benefit**: Up to 32% improvement over fixed CoT on challenging benchmarks. [self-discover-reasoning-structures]

### Inference-Time Scaling

Allocate additional compute during inference to improve quality:

- Generate multiple candidates, select best
- Extended reasoning chains before responding
- Iterate and refine through multiple passes
- Search solution spaces more thoroughly

**Trade-off**: Latency for quality. Smaller models + inference scaling can outperform larger models with standard inference. [inference-time-scaling]

---

## II. Feedback & Iteration Patterns

### Reflection Loop

After generating draft, model grades against metric and refines using feedback.

```python
for attempt in range(max_iters):
    draft = generate(prompt)
    score, critique = evaluate(draft, metric)
    if score >= threshold:
        return draft
    prompt = incorporate(critique, prompt)
```

**Use when**: Quality matters—writing, reasoning, code. [reflection]

### Self-Critique Evaluator Loop

Bootstrap a self-taught evaluator:

1. Generate multiple candidate outputs
2. Model judges which is better with reasoning trace
3. Fine-tune judge on its own traces
4. Use as reward model or quality gate
5. Periodically refresh with new synthetic debates

**Risk**: Evaluator-model collusion; needs adversarial tests. [self-critique-evaluator-loop]

### Rich Feedback Loops

Expose iterative, machine-readable feedback—compiler errors, test failures, linter output—after every tool call. Agent uses diagnostics to plan next step.

**Key insight**: "Give it errors, not bigger prompts." Ground truth enables self-correction. [rich-feedback-loops]

### Stop Hook Auto-Continue

Programmatically check success criteria after each agent turn. If criteria not met, automatically continue execution.

```python
on_stop_hook() {
    if run_tests().failed:
        agent.continue_with_prompt("Tests failed. Fix these issues.")
    else:
        agent.stop()
}
```

**Benefit**: "Deterministic outcomes from non-deterministic processes." [stop-hook-auto-continue-pattern]

---

## III. Multi-Agent & Orchestration Patterns

### Dual LLM Pattern

Split roles for privilege separation:

- **Privileged LLM**: Plans and calls tools but never sees raw untrusted data
- **Quarantined LLM**: Reads untrusted data but has zero tool access
- Pass data as symbolic variables; privileged side only manipulates references

**Use when**: Agents handle untrusted text and wield tools. [dual-llm-pattern]

### Oracle-Worker Multi-Model

Two-tier system with specialized roles:

- **Worker**: Fast, cost-effective agent for bulk tool use and generation
- **Oracle**: Powerful, expensive model for high-level reasoning, planning, debugging

Worker explicitly requests Oracle consultation when stuck. [oracle-and-worker-multi-model]

### Opponent Processor / Multi-Agent Debate

Spawn opposing agents with different goals to debate positions:

- **Pro vs. Con**: One argues for, another against
- **Uncorrelated context**: Independent reasoning prevents groupthink
- Let them critique each other, then synthesize

**Use when**: Decisions suffer from confirmation bias or limited perspectives. [opponent-processor-multi-agent-debate]

### Sub-Agent Spawning

Main agent spawns focused sub-agents with isolated contexts:

- **Virtual file isolation**: Subagent only sees files explicitly passed
- **Tool scoping**: Inherit all parent tools or use subset
- **Parallelization**: Run multiple subagents concurrently

**Use cases**: Context window management, concurrent work, security isolation. [sub-agent-spawning]

### LLM Map-Reduce

- **Map**: Spawn sandboxed LLMs—each ingests one untrusted chunk, emits constrained output (boolean, JSON)
- **Reduce**: Aggregate safe summaries with deterministic code or privileged LLM

**Benefit**: Malicious item can't taint others; scalable parallelism. [llm-map-reduce-pattern]

### Initializer-Maintainer Dual Agent

Two-agent architecture for long-running projects:

- **Initializer** (runs once): Creates feature list, progress tracking, environment bootstrap, initial commit
- **Maintainer** (runs each session): Reads context, selects next task, implements, verifies, commits

**Key insight**: Mirrors human shift handoffs. [initializer-maintainer-dual-agent]

### Discrete Phase Separation

Break workflows into isolated phases with clean handoffs:

1. **Research Phase**: Deep exploration, no implementation
2. **Planning Phase**: Structured roadmap, no coding
3. **Execution Phase**: Implement systematically

Pass only distilled conclusions between phases, not full conversation history. [discrete-phase-separation]

---

## IV. Control Flow Patterns

### Plan-Then-Execute

Split into two phases:

1. **Plan phase**: LLM generates fixed sequence of tool calls before seeing untrusted data
2. **Execution phase**: Controller runs exact sequence; outputs shape parameters but cannot change which tools run

**Security benefit**: Prevents prompt injection from redirecting agent. [plan-then-execute-pattern]

### Code-Then-Execute

LLM outputs sandboxed program or DSL script:

1. LLM writes code that calls tools
2. Static checker/taint engine verifies flows
3. Interpreter runs code in locked sandbox

**Benefit**: Full data-flow analysis, taint tracking, formal verifiability. [code-then-execute-pattern]

### Inversion of Control

Give agent tools + high-level goal; let it decide orchestration. Humans supply guardrails (first 10% + last 3%), agent handles middle 87%.

**Insight**: "It's a big bird, it can catch its own food." [inversion-of-control]

### Continuous Autonomous Task Loop

Continuous loop handling task selection, execution, completion:

1. Fresh context per iteration
2. Autonomous task selection via subagents
3. Automated git management
4. Intelligent rate limit handling
5. Configurable iteration limits

**Use when**: Sustained autonomous development across many tasks. [continuous-autonomous-task-loop-pattern]

### Parallel Tool Execution

Conditional execution based on tool type:

- **All read-only**: Execute concurrently
- **Any state-modifying**: Execute sequentially

**Trade-off**: Performance vs. safety. Classify tools upfront. [parallel-tool-execution]

---

## V. Context & Memory Patterns

### Context Minimization

Purge or redact untrusted segments once they've served their purpose:

```python
sql = LLM("to SQL", user_prompt)
remove(user_prompt)  # tainted tokens gone
rows = db.query(sql)
answer = LLM("summarize", rows)
```

**Benefit**: Eliminates latent injections; reduces context anxiety. [context-minimization-pattern]

### Filesystem-Based Agent State

Persist intermediate results to files for workflow resumption:

- Check if previous work exists → resume
- Checkpoint after expensive operations
- Include metadata (workflow_id, current_step, timestamps)

**Use when**: Long-running tasks, transient failures, multi-session work. [filesystem-based-agent-state]

### Episodic Memory Retrieval

Vector-backed episodic memory store:

1. After every episode, write "memory blob" (event, outcome, rationale)
2. On new tasks, embed prompt, retrieve top-k similar memories
3. Inject as hints in context
4. Apply TTL/decay scoring to prune stale memories

**Benefit**: Richer continuity, fewer repeated mistakes. [episodic-memory-retrieval-injection]

### Proactive State Externalization

Leverage model's natural tendency to write summaries/notes:

- Provide templates and schemas for agent-generated notes
- Combine with external memory (agent notes as supplementary)
- Structure to capture decision rationale, not just actions

**Risk**: Self-generated notes often incomplete; may spend tokens on documentation over progress. [proactive-agent-state-externalization]

### Progressive Tool Discovery

Present tools through filesystem-like hierarchy:

1. **Name only**: Minimal context for browsing
2. **Name + description**: Understand purpose
3. **Full definition**: Complete schema when needed

**Use when**: 20+ tools; reduces initial context consumption. [progressive-tool-discovery]

---

## VI. Learning & Adaptation Patterns

### Skill Library Evolution

Persist working code as reusable functions:

```text
Ad-hoc Code → Save Solution → Reusable Function → Documented Skill → Agent Capability
```

**Progressive disclosure**: Inject skill descriptions into prompt; provide `load_skills` tool for full content on demand.

**Lazy-loading MCP**: Bind servers to skills with selective tool loading (91% token reduction possible). [skill-library-evolution]

---

## VII. Safety & Control Patterns

### Human-in-the-Loop Approval

Insert approval gates for high-risk functions:

1. Agent requests permission before executing
2. Human receives context-rich approval request via Slack/email/SMS
3. Quick approve/reject/modify
4. Agent proceeds or adapts

**When to apply**: Production DB ops, external API calls, destructive file ops, compliance-sensitive actions. [human-in-loop-approval-framework]

### Spectrum of Control / Blended Initiative

Support multiple autonomy levels:

- **Low**: Tab-completion (human driving)
- **Medium**: Edit region/file based on instruction
- **High**: Multi-file tasks, complex refactoring
- **Very High**: Background agents for entire features/PRs

Users seamlessly switch between modes. [spectrum-of-control-blended-initiative]

### Tool Capability Compartmentalization

Split monolithic tools into reader/processor/writer micro-tools:

- Require per-call consent when composing across capability classes
- Run each class in isolated subprocess with scoped permissions
- Flag attempts to chain tools that recreate dangerous patterns

```yaml
email_reader:
  capabilities: [private_data, untrusted_input]
  permissions:
    fs: read-only:/mail
    net: none
```

[tool-capability-compartmentalization]

---

## VIII. Pattern Selection Guide

### By Problem Type

| Problem | Patterns |
| ------- | -------- |
| Complex multi-step reasoning | ToT, GoT, LATS, Self-Discover |
| Quality-sensitive generation | Reflection, Self-Critique, Evaluator-Optimizer |
| Untrusted input handling | Dual LLM, Context Minimization, Map-Reduce |
| Long-running projects | Initializer-Maintainer, Filesystem State, Phase Separation |
| Large-scale parallelization | Sub-Agent Spawning, Map-Reduce, Parallel Tool Execution |
| High-stakes decisions | Multi-Agent Debate, Human-in-the-Loop, Voting |
| Continuous automation | Stop Hook Auto-Continue, Continuous Task Loop |

### By Trade-off Priority

| Priority | Patterns |
| -------- | -------- |
| **Minimize latency** | Single-pass, Parallel Execution, Progressive Discovery |
| **Maximize quality** | Inference Scaling, Reflection, Self-Consistency |
| **Ensure safety** | Plan-Then-Execute, Dual LLM, Human-in-the-Loop |
| **Reduce cost** | Oracle-Worker, Context Minimization, Skill Reuse |
| **Handle scale** | Map-Reduce, Sub-Agents, Continuous Loop |

---

## Citations

All patterns sourced from [awesome-agentic-patterns](https://github.com/nibzard/awesome-agentic-patterns):

- [tree-of-thought-reasoning]: Based on Yao et al. (2023)
- [graph-of-thoughts]: Based on Besta et al., ETH Zurich
- [language-agent-tree-search-lats]: Based on Zhou et al.
- [self-discover-reasoning-structures]: Based on Google DeepMind, USC
- [inference-time-scaling]: Based on Google DeepMind, OpenAI
- [reflection]: Based on Shinn et al. (2023)
- [self-critique-evaluator-loop]: Based on Meta AI
- [rich-feedback-loops]: Based on Thorsten Ball, Quinn Slack
- [stop-hook-auto-continue-pattern]: Based on Boris Cherny (Anthropic)
- [dual-llm-pattern]: Based on Simon Willison, Beurer-Kellner et al.
- [oracle-and-worker-multi-model]: Based on Sourcegraph
- [opponent-processor-multi-agent-debate]: Based on Dan Shipper
- [sub-agent-spawning]: Based on Quinn Slack, Thorsten Ball
- [llm-map-reduce-pattern]: Based on Beurer-Kellner et al.
- [initializer-maintainer-dual-agent]: Based on Anthropic Engineering
- [discrete-phase-separation]: Based on Sam Stettner (Ambral)
- [plan-then-execute-pattern]: Based on Beurer-Kellner et al.
- [code-then-execute-pattern]: Based on DeepMind CaMeL
- [inversion-of-control]: Based on Quinn Slack, Thorsten Ball
- [continuous-autonomous-task-loop-pattern]: Internal Practice
- [parallel-tool-execution]: Based on Gerred Dillon
- [context-minimization-pattern]: Based on Beurer-Kellner et al.
- [filesystem-based-agent-state]: Based on Anthropic Engineering
- [episodic-memory-retrieval-injection]: Based on Cursor AI, Windsurf
- [proactive-agent-state-externalization]: Based on Cognition AI
- [progressive-tool-discovery]: Based on Anthropic Engineering
- [skill-library-evolution]: Based on Anthropic, Will Larson, Amp
- [human-in-loop-approval-framework]: Based on Dexter Horthy (HumanLayer)
- [spectrum-of-control-blended-initiative]: Based on Aman Sanger (Cursor)
- [tool-capability-compartmentalization]: Based on Simon Willison
