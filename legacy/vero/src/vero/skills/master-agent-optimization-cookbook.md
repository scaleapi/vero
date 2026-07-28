# Master Agent Optimization Cookbook

A consolidated reference for building and optimizing AI agent systems. Citations reference source artifacts.

---

## 1. Core Distinctions

**Workflows** = LLMs + tools orchestrated via predefined code paths.  
**Agents** = LLMs dynamically directing their own processes and tool usage. [anthropic-building-effective-agentic-systems]

**When to use what:**
- Single LLM call with retrieval → Most applications
- Workflows → Predictable, well-defined tasks requiring consistency
- Agents → Open-ended problems where steps can't be predicted [anthropic-building-effective-agentic-systems]

**Core principle:** Start simple. Add complexity only when simpler solutions demonstrably fail.

---

## 2. Context Engineering

Context engineering = optimizing the tokens in the LLM's context window for desired behavior. It's prompt engineering evolved for multi-turn, tool-using agents. [anthropic-effective-context-engineering]

### Why It Matters

LLMs have finite "attention budgets." As context grows:
- Performance degrades (context rot)
- Costs increase
- Latency increases

**Goal:** Find the smallest set of high-signal tokens that maximize likelihood of desired outcome. [anthropic-effective-context-engineering]

### System Prompt Best Practices

**Altitude:** Balance between too specific (brittle logic) and too vague (no actionable guidance). [anthropic-effective-context-engineering]

**Structure:** Use XML tags or Markdown headers to delineate sections:
```
<instructions>...</instructions>
<tool_guidance>...</tool_guidance>
<output_format>...</output_format>
```
[anthropic-effective-context-engineering]

**Modern model tips:**
- Newer models are more responsive to system prompts
- Dial back aggressive language ("CRITICAL: MUST" → "Use when...")
- Tell model what TO DO, not what NOT to do
- Match prompt style to desired output style [anthropic-prompting-best-practices]

### Dynamic Context

Inject runtime state explicitly:
```
Current date/time: {{ $now.toISO() }}
```
Without this, models guess dates, leading to suboptimal queries. [prompting-guide.com-context-engineering]

---

## 3. Prompting Strategies

### Chain-of-Thought (CoT)

**Use for:** Multi-step reasoning, math, logic, planning.

**Zero-shot:** Add "Let's think step by step" or "Think carefully."

**Few-shot:** Show worked examples with explicit reasoning steps.

**Note:** CoT buys "compute time" for the model to resolve dependencies before acting. [gemini-ai-agents-cookbook]

### ReAct (Reason + Act)

**Use for:** Tool-using agents requiring iterative problem-solving.

**Pattern:**
```
Thought: [reasoning about current state]
Action: tool_name(params)
Observation: [tool result]
... repeat ...
Final Answer: [result]
```
[chatgpt-ai-agents-cookbook]

### Self-Consistency

**Use for:** High-stakes decisions where confidence matters.

**Pattern:** Sample N reasoning paths (3-5 typical), extract answers, majority vote.

**Tradeoff:** 3x-5x cost for 5-15% accuracy gain. [perplexity-ai-agents-cookbook]

### CodeAct

**Use for:** Complex tool interactions, data processing, multi-step computation.

**Pattern:** Agent emits executable Python instead of JSON tool calls. More expressive, reduces round-trips.

**Benefit:** A single Python loop replaces 10 sequential tool calls. [gemini-ai-agents-cookbook]

---

## 4. Tool Design (Agent-Computer Interface)

Invest as much effort in ACI as you would in HCI. [anthropic-building-effective-agentic-systems]

### Documentation is Everything

```python
def search_database(query: str, limit: int = 5) -> str:
    """
    Search internal knowledge base for documents.
    
    Use for: Company policies, financial reports, historical data.
    Do NOT use for: General world knowledge (use web_search instead).
    
    Args:
        query: Keyword-rich semantic query.
            Bad: "revenue"
            Good: "Q3 2024 revenue breakdown for cloud division"
        limit: Max results (default 5, max 20). High limits increase latency.
    
    Returns: JSON list of document summaries with citation IDs.
    
    Example: search_database("quarterly revenue 2024", limit=3)
    """
```
[anthropic-writing-tools-for-agents, gemini-ai-agents-cookbook]

### Low-Friction Formats

**Bad:** Require diffs with line counts, JSON escaping, complex syntax.  
**Good:** Natural formats close to training data. Let model write `old_content` → `new_content` instead of unified diffs.

Give the model room to "think" before committing to output. [anthropic-building-effective-agentic-systems]

### Poka-Yoke (Error-Proofing)

Design tools so incorrect usage is difficult:
- Require absolute paths, not relative
- Use enums for constrained choices: `"format": {"enum": ["json", "csv"]}`
- Validate strictly, fail loudly [chatgpt-ai-agents-cookbook]

**Example:** Agents make mistakes with relative paths after changing directories. Fix: require absolute paths → flawless usage. [anthropic-building-effective-agentic-systems]

### Consolidate Functionality

Instead of: `list_users`, `list_events`, `create_event`  
Implement: `schedule_event` (finds availability + schedules)

Instead of: `get_customer_by_id`, `list_transactions`, `list_notes`  
Implement: `get_customer_context` (compiles all relevant info) [anthropic-writing-tools-for-agents]

### Token-Efficient Responses

- Implement pagination, filtering, truncation with sensible defaults
- Restrict tool responses to ~25K tokens by default
- Return semantic identifiers ("Jane Smith") not UUIDs
- Offer `response_format` param: "concise" vs "detailed" [anthropic-writing-tools-for-agents]

### Namespacing

With many tools, use prefixes to delineate boundaries:
- `browser_click`, `browser_navigate`, `browser_scroll`
- `shell_execute`, `shell_read_output`

Allows constraining to tool groups via response prefill. [manus-lessons-from-building-manus]

---

## 5. Workflow Patterns

### Prompt Chaining

**Use for:** Tasks decomposable into fixed sequential steps.

**Pattern:** LLM₁ → gate check → LLM₂ → gate check → LLM₃

**Examples:**
- Generate outline → validate → write sections → polish
- Generate marketing copy → translate [anthropic-building-effective-agentic-systems]

### Routing

**Use for:** Heterogeneous inputs requiring specialized handling.

**Pattern:** Classifier routes to specialist agents.

**Examples:**
- Customer queries → technical / billing / general handlers
- Easy questions → small model, hard → large model [anthropic-building-effective-agentic-systems]

### Parallelization

**Sectioning:** Split task into independent subtasks, run in parallel, aggregate.  
**Voting:** Run same task N times, majority vote.

**Examples:**
- Guardrails: content moderation ∥ response generation
- Code review: security ∥ performance ∥ style [anthropic-building-effective-agentic-systems]

### Orchestrator-Workers

**Use for:** Complex tasks where subtasks can't be predicted upfront.

**Pattern:** Orchestrator plans dynamically, delegates to specialized workers, synthesizes results.

**Key difference from parallelization:** Subtasks determined at runtime, not predefined. [anthropic-building-effective-agentic-systems]

### Evaluator-Optimizer

**Use for:** Tasks with clear evaluation criteria where iteration adds value.

**Pattern:** Generate → Evaluate → Feedback → Improve → Loop

**Signs of good fit:**
1. Human feedback demonstrably improves output
2. LLM can provide such feedback [anthropic-building-effective-agentic-systems]

---

## 6. Long-Horizon Task Management

### Compaction

Summarize context nearing limit, reinitialize with summary + recent state.

**Approach:** Preserve architectural decisions, unresolved bugs, implementation details. Discard redundant tool outputs. Continue with compressed context + most recently accessed files.

**Lightest touch:** Clear tool call results deep in history—why would agent need raw results again? [anthropic-effective-context-engineering]

### Structured Note-Taking

Agent writes persistent notes outside context window, retrieves later.

**Pattern:** Maintain `todo.md` or `NOTES.md`, update as task progresses.

**Game-playing example:** Agent maintains objective tallies across thousands of steps, maps of explored regions, combat strategy notes. Enables multi-hour coherence across context resets. [anthropic-effective-context-engineering]

### Sub-Agent Architectures

Delegate focused tasks to sub-agents with clean context windows.

**Pattern:** Main agent coordinates high-level plan. Sub-agents explore extensively (10K+ tokens), return condensed summaries (1-2K tokens).

**Benefit:** Separation of concerns—search context isolated within sub-agents, lead agent focuses on synthesis. [anthropic-effective-context-engineering]

### Multi-Window Task Management

For tasks spanning multiple context windows:
1. First window: Set up framework (write tests, create setup scripts)
2. Subsequent windows: Iterate on todo-list
3. Have model write tests in structured format (e.g., `tests.json`)
4. Create setup scripts (`init.sh`) to gracefully restart [anthropic-prompting-best-practices]

---

## 7. Production Patterns

### KV-Cache Optimization

**The metric:** KV-cache hit rate directly affects latency and cost.

**Cached vs uncached:** Can be up to 10x cost difference depending on provider.

**Rules:**
1. Keep prompt prefix stable (no timestamps at start!)
2. Make context append-only (no modifications to previous turns)
3. Ensure deterministic serialization (JSON key ordering)
4. Mark cache breakpoints explicitly if needed [manus-lessons-from-building-manus]

### Mask, Don't Remove

Don't dynamically add/remove tools mid-iteration:
- Tool definitions at front of context → changes invalidate KV-cache for all subsequent content
- Missing tool definitions confuse model when previous turns reference them

**Solution:** Use context-aware state machine to mask token logits during decoding, not remove tool definitions.

**Response prefill modes:**
- Auto: `<|im_start|>assistant`
- Required: `<|im_start|>assistant<tool_call>`
- Specified subset: `<|im_start|>assistant<tool_call>{"name": "browser_` [manus-lessons-from-building-manus]

### File System as Context

Treat file system as external memory: unlimited, persistent, agent-operable.

**Pattern:** Agent writes to / reads from files on demand. Compression becomes restorable—content can be dropped if path preserved.

**Example:** Coding agents can discover state from filesystem rather than relying solely on compaction. [manus-lessons-from-building-manus]

### Attention Manipulation via Recitation

**Problem:** Long loops cause goal drift.

**Solution:** Agent rewrites todo list, reciting objectives at context end. Pushes global plan into recent attention span.

**Manus:** Creates `todo.md`, updates step-by-step, checking off items. Not cute behavior—deliberate attention manipulation. [manus-lessons-from-building-manus]

### Keep Errors In Context

**Don't:** Hide errors, clean traces, retry silently.  
**Do:** Leave failed actions and stack traces in context.

When model sees failure + observation, it updates beliefs and shifts away from repeating mistake. Error recovery is a key indicator of true agentic behavior. [manus-lessons-from-building-manus]

### Avoid Few-Shot Ruts

**Problem:** Repetitive action-observation pairs cause model to mimic patterns blindly.

**Example:** Reviewing 20 resumes → agent falls into rhythm, overgeneralizes.

**Solution:** Introduce structured variation—different serialization templates, alternate phrasing, minor formatting noise. Break the pattern. [manus-lessons-from-building-manus]

### Infinite Loop Prevention

**Pattern:** Hash recent tool calls. If hash repeats, inject warning:
```python
if history_hashes[-1] == history_hashes[-2]:
    inject("WARNING: Repeating same action. Try different approach.")
```
[gemini-ai-agents-cookbook]

---

## 8. Testing & Evaluation

### Evaluation-Driven Tool Development

1. Build prototype, test manually
2. Generate diverse evaluation tasks grounded in real-world use
3. Run programmatic evaluation with simple agentic loops
4. Analyze results—what agents omit is often more important than what they include
5. Iterate based on findings [anthropic-writing-tools-for-agents]

**Strong eval tasks:** Multi-step, realistic data, not sandbox simplifications.
```
"Customer ID 9182 reported triple charge. Find log entries, 
determine if others affected."
```

**Weak eval tasks:** Single-step, pre-specified parameters.
```
"Search payment logs for purchase_complete and customer_id=9182."
```
[anthropic-writing-tools-for-agents]

### Metrics to Track

- Top-level accuracy
- Tool call counts (redundant calls → adjust pagination)
- Tool errors (invalid params → clearer descriptions)
- Total runtime
- Token consumption [anthropic-writing-tools-for-agents]

### LLM-as-Judge

For subjective qualities, use separate LLM to evaluate:
```python
judge_prompt = f"""Rate this output on:
- Clarity (1-5)
- Completeness (1-5)
- Tone (1-5)

Output: {output}

Respond in JSON with scores and justification."""
```
[perplexity-ai-agents-cookbook]

---

## 9. Quick Reference Tables

### Workflow Pattern Selection

| Scenario | Pattern |
|----------|---------|
| Fixed sequential steps | Prompt Chaining |
| Different input types need different handling | Routing |
| Independent subtasks | Parallelization (Sectioning) |
| Need high confidence | Parallelization (Voting) |
| Subtasks unpredictable upfront | Orchestrator-Workers |
| Quality improves with iteration | Evaluator-Optimizer |
| Open-ended with tool use | ReAct Agent Loop |

### Prompting Strategy Selection

| Task Type | Strategy |
|-----------|----------|
| Simple QA | Few-shot |
| Multi-step reasoning | CoT + Self-Consistency |
| Tool-using iteration | ReAct |
| Complex computation | CodeAct |
| High-stakes decision | Self-Consistency (3-5 samples) |
| Multiple perspectives needed | Role-based multi-expert |

### Tool Design Checklist

- [ ] Description follows "Tool to X. Use when Y." format
- [ ] Includes example usage in docstring
- [ ] Documents edge cases and error handling
- [ ] Uses enums for constrained choices
- [ ] Requires absolute paths if applicable
- [ ] Returns semantic identifiers, not UUIDs
- [ ] Has token-efficient response format
- [ ] Namespaced with consistent prefix

---

## 10. Anti-Patterns

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| Timestamp at prompt start | Kills KV-cache | Move to end or inject conditionally |
| Dynamic tool add/remove | Invalidates cache, confuses model | Mask via logits, not removal |
| Hiding errors | Removes learning signal | Keep failures in context |
| Uniform action format | Few-shot mimicry, drift | Introduce structured variation |
| Full file rewrites | Token expensive, truncation risk | Use diffs or search-replace |
| Vague tool descriptions | Wrong tool selection | "Tool to X. Use when Y." + examples |
| Relative paths | Errors after directory change | Require absolute paths |
| No iteration limit | Infinite loops, runaway costs | Set max_iterations, detect loops |

---

## Citations

- [anthropic-building-effective-agentic-systems]: `anthropic-building-effective-agentic-systems.md`
- [anthropic-effective-context-engineering]: `anthropic-effective-context-engineering-for-AI-agents.md`
- [anthropic-prompting-best-practices]: `anthropic-prompting-best-practices.md`
- [anthropic-writing-tools-for-agents]: `anthropic-writing-tools-for-agents.md`
- [chatgpt-ai-agents-cookbook]: `chatgpt-ai-agents-cookbook.md`
- [claude-ai-agents-cookbook]: `claude-ai-agents-cookbook.md`
- [gemini-ai-agents-cookbook]: `gemini-ai-agents-cookbook.md`
- [manus-lessons-from-building-manus]: `manus-lessons-from-building-manus.md`
- [perplexity-ai-agents-cookbook]: `perplexity-ai-agents-cookbook.md`
- [prompting-guide.com-context-engineering]: `prompting-guide.com-context-engineering-guide.md`
