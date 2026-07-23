# AI Agent Design Cookbook

**A practical cookbook for building state-of-the-art AI agents**

This cookbook provides a collection of practical, composable strategies ("recipes") for designing AI agents. It focuses on what actually works in practice, drawing from recent research, production systems, and open-source patterns. The goal is to enable AI engineers to mix and match techniques to build robust, agentic systems.

---

## Primary References

- [Awesome Agentic Patterns](https://github.com/nibzard/awesome-agentic-patterns)
- [ADAS Paper](https://arxiv.org/pdf/2408.08435)
- [AFlow Paper](https://arxiv.org/pdf/2410.10762)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Cloudflare Agent Patterns](https://github.com/cloudflare/agents/tree/main/guides/anthropic-patterns/src/flows)

---

## Part I: Prompting Strategies

### 1.1 Chain-of-Thought (CoT)

**When to use:**
- Reasoning-heavy tasks
- Math, logic, planning
- Multi-step decision making

**Recipe:**
Encourage the model to generate intermediate reasoning before producing a final answer.

**Example instruction:**
> "Think step by step before producing the final answer."

**Variants:**
- **Explicit CoT:** "Explain your reasoning step by step."
- **Implicit CoT:** "Think carefully and make sure the answer is correct."

**Notes:**
- Improves performance on complex reasoning tasks
- Some models (e.g. Anthropic) prefer implicit CoT
- In production, reasoning can be hidden while still being used internally

---

### 1.2 Self-Consistency (Sampling + Voting)

**When to use:**
- High-stakes reasoning
- Math and logic problems
- When latency allows multiple samples

**Recipe:**
Sample multiple reasoning paths and aggregate the final answers.

**Practical steps:**
1. Run the same prompt N times with temperature > 0
2. Collect final answers
3. Choose the most common answer (or score them)

**Benefits:**
- Reduces stochastic reasoning errors
- Often outperforms single greedy decoding

**Reference:**
[Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)

---

### 1.3 ReAct (Reason + Act)

**When to use:**
- Tool-using agents
- Retrieval, browsing, APIs
- Iterative problem solving

**Recipe:**
Interleave reasoning steps with actions.

**Conceptual loop:**
```
Thought → Action → Observation → Thought → Action …
```

**Example:**
```
Thought: I need external information.
Action: Search("X documentation")
Observation: Search results
Thought: Based on this result…
```

**Benefits:**
- Strong grounding
- Transparent behavior
- Enables dynamic tool usage

**Reference:**
[ReAct Paper](https://arxiv.org/abs/2210.03629)

---

### 1.4 CodeAct

**When to use:**
- Complex tool interactions
- Data processing
- Multi-step computation

**Recipe:**
Allow the agent to emit executable Python code instead of rigid tool calls.

**Example pattern:**
1. Agent outputs Python code
2. Runtime executes code
3. Stdout/errors are returned to the agent

**Benefits:**
- More expressive than fixed tool schemas
- Simplifies integration with many APIs
- Strong results in coding and browsing agents

**Reference:**
[CodeAct Paper](https://arxiv.org/pdf/2408.08435)

---

### 1.5 Self-Critique / Actor–Critic

**When to use:**
- High-quality generation
- Writing, reasoning, planning
- Tasks with clear evaluation criteria

**Recipe:**
Use one model (or pass) to generate output and another to critique or improve it.

**Common patterns:**
- "What is wrong with this answer?"
- "Improve the previous output"
- Separate actor and critic roles

**Benefits:**
- Iterative improvement
- Reduces hallucinations
- Mimics human editing workflows

---

### 1.6 Tree-of-Thought / Search-Based Prompting

**When to use:**
- Very hard reasoning tasks
- Planning and exploration
- Problems with many possible paths

**Recipe:**
Explore multiple reasoning branches instead of a single linear chain.

**Common approaches:**
- Tree of Thought (ToT)
- Monte Carlo Tree Search (MCTS)
- Language Agent Tree Search (LATS)

**Benefits:**
- Systematic exploration
- Better global solutions

**Reference:**
[LATS Paper](https://arxiv.org/abs/2310.04406)

---

## Part II: Tool Design

### 2.1 Design Tools for Models, Not Humans

**Principles:**
- Simple, familiar interfaces
- Clear argument names
- Minimal formatting requirements

**Guidelines:**
- Prefer JSON-like or function-call schemas
- Avoid complex syntax (diffs, regex-heavy formats)
- Provide examples in tool descriptions

---

### 2.2 Poka-Yoke (Error-Proofing)

**Recipe:**
Design tools so incorrect usage is difficult or impossible.

**Examples:**
- Require absolute paths instead of relative ones
- Validate arguments strictly
- Fail loudly and clearly

**Benefits:**
- Reduces agent confusion
- Improves reliability

---

### 2.3 General-Purpose Tools Worth Having

**Commonly useful tools:**
- Web search
- HTTP request tool
- Python code executor
- File system reader/writer
- Embedding + vector search
- Memory / retrieval tool

**Reference implementations:**
[Cloudflare Agent Patterns](https://github.com/cloudflare/agents)

---

### 2.4 Code-Based Tools (Python)

**Recipe:**
Expose tools as Python functions.

**Example pattern:**
1. Register Python functions
2. Let the agent call them
3. Return structured outputs

**Benefits:**
- Easy to debug
- Composable
- Works well with CodeAct agents

---

### 2.5 Safety and Isolation

**Best practices:**
- Sandbox code execution
- Restrict network access where possible
- Log all tool usage
- Never expose secrets directly

**Pattern:**
"Plan-then-execute" to prevent prompt injection via tool output

---

## Part III: Workflow Design (Inference Strategies)

### 3.1 Prompt Chaining

**When to use:**
- Structured tasks
- Writing, summarization, analysis

**Recipe:**
Break a task into sequential prompts.

**Example:**
1. Generate outline
2. Expand sections
3. Edit and polish

**Tradeoff:**
- Higher latency
- Better control and accuracy

---

### 3.2 Routing

**When to use:**
- Heterogeneous inputs
- Different task types

**Recipe:**
Use a classifier (or LLM) to route inputs to specialized flows.

**Examples:**
- Simple queries → small model
- Complex queries → large model
- Code → coding agent

---

### 3.3 Parallelization (Divide and Conquer)

**When to use:**
- Independent subtasks
- Redundancy for correctness

**Patterns:**
- Sectioning (split the task)
- Voting (multiple agents solve same task)

**Benefits:**
- Faster execution
- Higher robustness

---

### 3.4 Orchestrator–Worker

**When to use:**
- Open-ended tasks
- Research, large codebases

**Recipe:**
1. One orchestrator agent plans and assigns tasks
2. Worker agents execute subtasks
3. Results are aggregated

**Reference:**
[Anthropic Research Agent Patterns](https://www.anthropic.com/engineering/building-effective-agents)

---

### 3.5 Evaluator–Optimizer

**When to use:**
- Quality-sensitive generation
- Iterative refinement

**Recipe:**
1. Generate candidate output
2. Evaluate it
3. Improve based on feedback
4. Repeat until satisfied

---

### 3.6 Plan-Then-Execute

**When to use:**
- Safety-critical workflows
- Deterministic tool usage

**Recipe:**
1. Generate full plan of actions
2. Freeze the plan
3. Execute actions without re-planning

**Benefits:**
- Prevents tool-output prompt injection
- Improves predictability

---

### 3.7 Self-Consistency at the Agent Level

**When to use:**
- High-stakes decisions

**Recipe:**
Run the entire agent multiple times and aggregate results.

**Example:**
- 3 independent agent runs
- Majority vote or merge findings

---

## Final Note

There is no single "best" agent architecture.

**Strong agents are built by:**
- Combining simple prompting strategies
- Designing model-friendly tools
- Choosing the right workflow pattern for the task

*Treat this cookbook as a toolbox — not a prescription.*
