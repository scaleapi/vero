# The AI Agents Cookbook: Practical Recipes for Building Effective AI Agents

**A practical guide to designing, prompting, and optimizing autonomous AI agents that work in production.**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Part I: Prompting Strategies](#part-i-prompting-strategies)
3. [Part II: Tool Design](#part-ii-tool-design)
4. [Part III: Workflow Design](#part-iii-workflow-design)
5. [Part IV: Testing & Evaluation](#part-iv-testing--evaluation)
6. [Advanced Recipes](#advanced-recipes)
7. [Common Pitfalls & Solutions](#common-pitfalls--solutions)

---

## Introduction

Building effective AI agents requires more than just connecting an LLM to some tools. Success lies in understanding **three interconnected domains**:

1. **Prompting**: How to structure instructions so LLMs reason effectively
2. **Tool Design**: How to expose capabilities in ways agents can reliably use
3. **Workflow Design**: How to orchestrate multiple LLM calls and tools into coherent systems

This cookbook provides battle-tested recipes for each domain—not theory, but practical strategies that work across reasoning tasks, code generation, and knowledge-intensive problems.

### When to Use This Cookbook

- You're building an agent and need concrete patterns, not frameworks
- You want to understand what prompting strategies work for different task types
- You're optimizing tool definitions and struggling with agent errors
- You're designing multi-step workflows and need proven orchestration patterns
- You want to test agents rigorously before deployment

### Key Principle: Start Simple, Add Complexity Only When Needed

The most successful production agents in the wild follow this principle:

1. **Baseline**: Direct LLM call with in-context examples (few-shot prompting)
2. **Layer 2**: Add tools and augmented retrieval
3. **Layer 3**: Introduce workflow decomposition (prompt chaining, routing)
4. **Layer 4**: Enable autonomous agents (tool use + planning loops)

Most applications stop at Layer 2-3. Only move to autonomous agents when simpler patterns fail.

---

# Part I: Prompting Strategies

## Recipe 1.1: Zero-Shot Chain of Thought (Simple Reasoning)

**Use Case**: Tasks requiring multi-step reasoning (math, logic, analysis). No domain-specific examples available.

**Why It Works**: By explicitly asking the model to reason before answering, you focus its attention on intermediate steps rather than jumping to conclusions.

### Template

```
{task_description}

Let's think step-by-step:
1. First, I'll {identify_key_elements}
2. Then, I'll {analyze_relationships}
3. Finally, I'll {derive_conclusion}
```

### Example: Mathematical Problem

```
Problem: A store sells apples at $2 each and oranges at $3 each. 
Sarah buys 5 apples and 4 oranges. How much does she spend?

Let's think step-by-step:
1. First, I'll calculate the cost of apples
2. Then, I'll calculate the cost of oranges
3. Finally, I'll add them together to get the total cost
```

### Python Implementation

```python
def solve_with_cot(problem: str, model_client) -> str:
    prompt = f"""{problem}

Let's work through this step-by-step:
1. First, identify the key information
2. Then, determine what calculation is needed
3. Finally, compute the answer and verify it makes sense
"""
    response = model_client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

### Variations & Tuning

| Trigger Phrase | Best For | Trade-off |
|---|---|---|
| "Let's think step-by-step" | General reasoning | Fast, works 80% of the time |
| "Let's work this out in a step-by-step way to be sure we have the right answer" | High-stakes decisions | Slightly longer, more careful |
| "First, let's think about this logically" | Logical reasoning | Works well for classification |
| "Consider the principles involved" | Deep understanding needed | Requires more tokens |

**Pro Tip**: Phrase matters. "Let's think step-by-step" is surprisingly effective because it's how CoT was introduced in research. Longer phrases help for genuinely hard problems but add latency.

**When NOT to Use**: For simple lookups (factual questions with single answers), CoT adds latency without benefit. Use direct prompting instead.

---

## Recipe 1.2: Few-Shot Prompting (In-Context Learning)

**Use Case**: When you have 2-10 quality examples that demonstrate the task pattern. Essential for formatting-sensitive tasks.

**Why It Works**: Models learn from patterns in the examples, not just instructions. This is "learning by example" without fine-tuning.

### Template Structure

```
You are a {role}.

Example 1:
Input: {input_1}
Output: {output_1}

Example 2:
Input: {input_2}
Output: {output_2}

Example 3:
Input: {input_3}
Output: {output_3}

Now, perform the task for:
Input: {new_input}
Output:
```

### Example: Customer Support Classification

```
You are a customer support triage system. Classify customer messages into 
one of: General Question, Refund Request, Technical Issue, or Billing Problem.

Example 1:
Message: "How do I reset my password?"
Category: General Question

Example 2:
Message: "I was charged twice for my order on Jan 15. Please refund the duplicate charge."
Category: Billing Problem

Example 3:
Message: "The app keeps crashing when I try to upload a photo."
Category: Technical Issue

Example 4:
Message: "Can I get a refund for my purchase?"
Category: Refund Request

Now classify this message:
Message: "Your product doesn't work with my phone model."
Category:
```

### Python Implementation

```python
def classify_with_few_shot(message: str, examples: list, model_client) -> str:
    """
    examples: List of {"message": str, "category": str} dicts
    """
    few_shot_text = "You are a customer support triage system.\n\n"
    
    for i, example in enumerate(examples, 1):
        few_shot_text += f"Example {i}:\nMessage: \"{example['message']}\"\n"
        few_shot_text += f"Category: {example['category']}\n\n"
    
    prompt = f"""{few_shot_text}Now classify this message:
Message: "{message}"
Category:"""
    
    response = model_client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()
```

### Best Practices for Few-Shot Success

1. **Diversity Matters More Than Quantity**: 3 diverse examples beat 10 similar ones
   - Include edge cases, not just canonical examples
   - Show variety in input length, complexity, and style

2. **Ordering Effects Are Real**: Put easiest examples first
   - Model learns better when difficulty increases
   - Hard examples at the start confuse the learning signal

3. **Format Consistency Is Critical**: All examples must follow exact same format
   ```python
   # Good: Consistent markup
   Example: <input>How do I reset?</input> → <output>General</output>
   
   # Bad: Inconsistent formatting
   Input: "How do I reset?" → Category is "General"
   ```

4. **Optimal Count**: 2-5 examples for most tasks
   - 1-2: Insufficient for learning patterns
   - 3-5: Sweet spot for accuracy without overfitting
   - 10+: Diminishing returns, wastes tokens

5. **Mirror Real-World Distribution**: If 80% of questions are general, show that ratio
   - Imbalanced examples teach wrong distribution
   - Use stratified sampling when selecting examples

### Advanced: Dynamic Few-Shot Selection

When you have many possible examples, dynamically select the most relevant:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def select_best_examples(new_input: str, candidate_examples: list, 
                         embeddings_fn, num_examples: int = 3) -> list:
    """
    Selects examples most similar to the new input.
    embeddings_fn: Function that returns embedding vector
    """
    new_embedding = embeddings_fn(new_input).reshape(1, -1)
    
    similarities = []
    for example in candidate_examples:
        example_embedding = embeddings_fn(example["input"]).reshape(1, -1)
        sim = cosine_similarity(new_embedding, example_embedding)[0][0]
        similarities.append((sim, example))
    
    # Sort by similarity and return top K
    similarities.sort(reverse=True)
    return [ex for _, ex in similarities[:num_examples]]
```

---

## Recipe 1.3: ReACT—Reasoning + Acting with Tool Interleaving

**Use Case**: Multi-step tasks requiring external information or actions. Ideal when reasoning and tool use need to interleave.

**Why It Works**: By alternating between thinking and acting, the model can gather information mid-reasoning, adjust plans based on observations, and recover from errors.

### The ReACT Loop

```
Thought: I need to find X. Let me search for it.
Action: search("X")
Observation: [search results]
Thought: Now I understand. Let me use this to solve the problem.
Action: calculate(data_from_above)
Observation: [calculation result]
Thought: I have the answer.
Final Answer: ...
```

### Example: Research Agent

```python
def react_research_agent(question: str, tools: dict, model_client):
    """
    Implements ReACT pattern for answering research questions.
    """
    system_prompt = """You are a research assistant. You have access to:
- search(query): Search for information
- fetch_url(url): Get full text from a webpage
- extract_facts(text): Extract key facts from text

Follow this format for each step:
Thought: <your reasoning about what to do next>
Action: <tool_name>(<arguments>)
Observation: <result of the action>

Repeat until you have enough information to answer. Then provide:
Final Answer: <your answer with sources>"""

    messages = [
        {"role": "user", "content": question}
    ]
    
    max_iterations = 10
    for iteration in range(max_iterations):
        response = model_client.messages.create(
            model="claude-3-5-sonnet",
            system=system_prompt,
            max_tokens=2048,
            messages=messages
        )
        
        response_text = response.content[0].text
        
        # Check if we're done
        if "Final Answer:" in response_text:
            return response_text.split("Final Answer:")[-1].strip()
        
        # Parse Thought-Action-Observation cycle
        if "Action:" in response_text:
            # Extract action
            action_part = response_text.split("Action:")[-1].split("\n")[0].strip()
            
            # Execute tool
            observation = execute_action(action_part, tools)
            
            # Add to conversation
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            # Model didn't format correctly, return what we have
            return response_text
    
    return "Could not find answer within iteration limit"

def execute_action(action_str: str, tools: dict) -> str:
    """Parse and execute tool calls."""
    import re
    
    # Parse "tool_name(args)" format
    match = re.match(r'(\w+)\((.*)\)', action_str)
    if not match:
        return "Error: Invalid action format"
    
    tool_name, args_str = match.groups()
    
    if tool_name not in tools:
        return f"Error: Tool '{tool_name}' not found"
    
    try:
        # Simple argument parsing (enhance for complex args)
        args = [arg.strip().strip('"\'') for arg in args_str.split(',')]
        result = tools[tool_name](*args)
        return str(result)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"
```

### Best Practices for ReACT

1. **Tool Descriptions Are Critical**: Agents decide what to use based on descriptions
   ```python
   # Good: Clear action and context
   tools = {
       "search": {
           "description": "Search for recent information. Use when you need current facts, statistics, or news.",
           "parameters": {"query": "search query string"}
       }
   }
   
   # Bad: Vague description
   "search": {"description": "Search the internet"}
   ```

2. **Observe Format Matters**: Observations should be concise and structured
   ```
   # Good: Structured observation
   Observation: [Source: Wikipedia] Photosynthesis is a process where...
   
   # Bad: Overwhelming data
   Observation: [10,000 words of raw HTML]
   ```

3. **Set Iteration Limits**: Prevent infinite loops
   ```python
   max_iterations = 10  # For web search
   max_iterations = 5   # For API calls (faster feedback)
   max_iterations = 20  # For complex reasoning tasks
   ```

4. **Add Explicit Stopping Criteria**: Don't rely on format alone
   ```python
   if "Final Answer:" in response or iteration >= max_iterations:
       return extract_final_answer(response)
   ```

---

## Recipe 1.4: Self-Consistency Sampling (Ensemble Reasoning)

**Use Case**: Complex reasoning tasks where you can't predict the right answer. Math, logic, open-ended analysis.

**Why It Works**: Different reasoning paths can lead to the same correct answer. By sampling multiple paths and voting, you get robust results with minimal additional cost.

### The Pattern

```
1. Generate multiple diverse reasoning paths (3-10)
2. Extract the final answer from each path
3. Use majority voting to select the most common answer
4. Return both the answer and confidence (based on vote distribution)
```

### Python Implementation

```python
def self_consistency_reasoning(problem: str, model_client, 
                               num_samples: int = 5, temperature: float = 0.7) -> dict:
    """
    Samples multiple reasoning paths and uses majority voting.
    """
    from collections import Counter
    
    prompt = f"""{problem}

Think through this step-by-step and provide your final answer.
Final Answer: [answer only]"""
    
    answers = []
    reasoning_paths = []
    
    for sample_idx in range(num_samples):
        response = model_client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            temperature=temperature,  # Stochastic decoding
            messages=[{"role": "user", "content": prompt}]
        )
        
        full_text = response.content[0].text
        reasoning_paths.append(full_text)
        
        # Extract final answer
        if "Final Answer:" in full_text:
            answer = full_text.split("Final Answer:")[-1].strip()
            answers.append(answer)
    
    # Majority voting
    answer_counts = Counter(answers)
    most_common_answer, vote_count = answer_counts.most_common(1)[0]
    confidence = vote_count / num_samples
    
    return {
        "answer": most_common_answer,
        "confidence": confidence,
        "vote_distribution": dict(answer_counts),
        "all_reasoning_paths": reasoning_paths,
        "num_samples": num_samples
    }
```

### Cost-Benefit Analysis

| Samples | Performance Gain | Cost | Latency | When to Use |
|---------|-----------------|------|---------|------------|
| 1 | Baseline | 1x | 1x | Always start here |
| 3 | +5-10% | 3x | ~3x | Good balance for accuracy |
| 5 | +8-15% | 5x | ~5x | Standard for important tasks |
| 10 | +12-20% | 10x | ~10x | Only for critical decisions |

**Pro Tip**: Don't exceed 5-10 samples. Diminishing returns kick in hard. For 95% of cases, 3 samples provide excellent accuracy gain at 3x cost.

### Advanced: Weighted Voting

Not all reasoning paths are equally reliable. Use confidence scores:

```python
def weighted_self_consistency(problem: str, model_client, num_samples: int = 5) -> str:
    """
    Uses model confidence to weight votes instead of simple majority.
    """
    import re
    
    paths_with_confidence = []
    
    for _ in range(num_samples):
        response = model_client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            temperature=0.7,
            messages=[{"role": "user", "content": problem}]
        )
        
        text = response.content[0].text
        
        # Extract answer and confidence markers
        answer = extract_answer(text)
        confidence = 0.9 if "certain" in text.lower() else 0.6
        
        paths_with_confidence.append({
            "answer": answer,
            "confidence": confidence,
            "text": text
        })
    
    # Weighted voting
    answer_scores = {}
    for path in paths_with_confidence:
        answer = path["answer"]
        weight = path["confidence"]
        answer_scores[answer] = answer_scores.get(answer, 0) + weight
    
    best_answer = max(answer_scores.items(), key=lambda x: x[1])[0]
    return best_answer
```

---

## Recipe 1.5: Tree of Thought (Complex Problem Solving)

**Use Case**: Highly complex problems requiring exploration of multiple solution strategies. Puzzles, planning, creative tasks.

**Why It Works**: Human experts don't follow a single reasoning path. They explore branches, evaluate options, backtrack, and recombine ideas. ToT mimics this process.

### The Pattern

```
Step 1: Decompose problem into "thoughts" (partial solutions)
Step 2: Generate multiple possible next thoughts at each node
Step 3: Evaluate which thoughts are promising (prune dead ends)
Step 4: Search the tree (BFS for breadth, DFS for depth)
Step 5: Return the best solution found
```

### Python Implementation (Simplified)

```python
def tree_of_thought_solver(problem: str, model_client, max_depth: int = 3, 
                           branching_factor: int = 3) -> dict:
    """
    Implements Tree of Thought prompting.
    """
    import queue
    
    class TreeNode:
        def __init__(self, thought: str, depth: int, parent=None):
            self.thought = thought
            self.depth = depth
            self.parent = parent
            self.children = []
            self.score = 0.0
    
    # Start with root thought
    root_prompt = f"""{problem}

What are the key sub-problems or steps needed to solve this?
List them as "Thought 1: ...", "Thought 2: ...", etc."""
    
    response = model_client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=1024,
        messages=[{"role": "user", "content": root_prompt}]
    )
    
    root_thought = response.content[0].text
    root = TreeNode(root_thought, depth=0)
    
    # BFS to explore the tree
    q = queue.Queue()
    q.put(root)
    all_nodes = [root]
    
    while not q.empty() and len(all_nodes) < 20:  # Prevent explosion
        node = q.get()
        
        if node.depth >= max_depth:
            continue
        
        # Generate child thoughts
        expand_prompt = f"""Current thinking:
{node.thought}

Generate {branching_factor} different ways to develop this thinking further.
Format as "Option 1: ...", "Option 2: ...", etc."""
        
        response = model_client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": expand_prompt}]
        )
        
        options_text = response.content[0].text
        
        # Evaluate options
        eval_prompt = f"""Given these options:
{options_text}

Which ones are most promising? Score each as 0 (dead end) to 10 (very promising).
Format as "Option 1: [score]", etc."""
        
        response = model_client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=500,
            messages=[{"role": "user", "content": eval_prompt}]
        )
        
        scores_text = response.content[0].text
        
        # Parse scores and create children
        for i in range(1, branching_factor + 1):
            child_thought = f"Expanding on option {i}"  # Simplified
            child = TreeNode(child_thought, depth=node.depth + 1, parent=node)
            child.score = extract_score(scores_text, i)  # Parse score
            
            node.children.append(child)
            all_nodes.append(child)
            
            if child.score >= 5:  # Only explore promising nodes
                q.put(child)
    
    # Find best leaf
    best_node = max(all_nodes, key=lambda n: n.score)
    
    return {
        "best_solution": best_node.thought,
        "score": best_node.score,
        "depth": best_node.depth,
        "total_nodes_explored": len(all_nodes)
    }

def extract_score(text: str, option_num: int) -> float:
    """Extract score for a given option from evaluation response."""
    import re
    pattern = rf"Option {option_num}.*?(\d+)"
    match = re.search(pattern, text)
    return float(match.group(1)) if match else 0.0
```

### When to Use ToT vs Other Approaches

| Problem Type | Best Strategy | Why |
|---|---|---|
| Math/Logic puzzles | Tree of Thought | Need to explore multiple paths |
| Factual QA | Zero-shot or few-shot | Single answer lookup |
| Creative writing | ToT or Self-Consistency | Multiple valid solutions |
| Code generation | ReACT + test feedback | Iterative refinement needed |
| Simple classification | Few-shot CoT | Clear decision rules |

---

## Recipe 1.6: Role-Based Prompting (Multi-Perspective Reasoning)

**Use Case**: Complex analysis requiring multiple viewpoints. Better decisions through diverse expertise.

**Why It Works**: Different "personas" bring different reasoning styles. A critic catches flaws that a proponent misses.

### The Pattern

```
1. Assign distinct expert roles
2. Have each expert analyze the problem independently
3. Let experts critique each other's reasoning
4. Synthesize into final recommendation
```

### Python Implementation

```python
def multi_expert_analysis(problem: str, model_client) -> dict:
    """
    Get analysis from multiple expert roles.
    """
    experts = {
        "analyst": "You are a data analyst. Focus on facts, evidence, and quantitative analysis.",
        "critic": "You are a critical thinker. Identify weaknesses, assumptions, and edge cases.",
        "visionary": "You are a visionary strategist. Think about long-term implications and novel approaches.",
    }
    
    # Round 1: Independent analysis
    analyses = {}
    for expert_name, role in experts.items():
        prompt = f"""{role}

Problem: {problem}

Provide your perspective and key points:"""
        
        response = model_client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            system=role
        )
        
        analyses[expert_name] = response.content[0].text
    
    # Round 2: Cross-examination
    cross_exam = {}
    for expert_name, role in experts.items():
        other_views = "\n".join([
            f"{e}: {a}" for e, a in analyses.items() if e != expert_name
        ])
        
        prompt = f"""{role}

Other experts have made these arguments:
{other_views}

What are the strengths and weaknesses of their perspectives?"""
        
        response = model_client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            system=role
        )
        
        cross_exam[expert_name] = response.content[0].text
    
    # Round 3: Synthesis
    synthesis_prompt = f"""Three experts have analyzed this problem:

Analyst perspective:
{analyses['analyst']}

Critic perspective:
{analyses['critic']}

Visionary perspective:
{analyses['visionary']}

Synthesize their insights into a balanced recommendation that acknowledges trade-offs."""
    
    response = model_client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=1024,
        messages=[{"role": "user", "content": synthesis_prompt}]
    )
    
    return {
        "individual_analyses": analyses,
        "cross_examination": cross_exam,
        "synthesis": response.content[0].text
    }
```

### Expert Combinations for Different Domains

| Domain | Expert Roles | Why These Work |
|--------|---|---|
| Business decision | Analyst, Critic, Visionary | Covers data, risks, and strategy |
| Technical design | Architect, Pragmatist, Skeptic | Structure, implementation, edge cases |
| Content review | Writer, Editor, Audience advocate | Quality, clarity, relevance |
| Code review | Designer, Pragmatist, Security expert | Architecture, maintainability, safety |

---

# Part II: Tool Design

## Recipe 2.1: Designing Clear, Reliable Tools

**Core Principle**: A tool's success depends 80% on its description and parameter design, 20% on its implementation.

### The Tool Template

```python
def create_tool_definition(name: str, description: str, parameters: dict) -> dict:
    """
    Standard tool definition format compatible with Claude's tool_use feature.
    """
    return {
        "name": name,
        "description": description,  # This is critical—make it crystal clear
        "input_schema": {
            "type": "object",
            "properties": parameters,
            "required": list(parameters.keys())  # Keep minimal
        }
    }

# Example: Good tool definition
search_tool = create_tool_definition(
    name="search_web",
    description="Search for current information on the web. Use when you need facts, news, or recent data that might not be in your training data.",
    parameters={
        "query": {
            "type": "string",
            "description": "The search query. Be specific—'Python async/await' not 'Python'."
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return. 3-5 usually sufficient.",
            "default": 5
        }
    }
)

# Example: Bad tool definition (too vague)
bad_tool = {
    "name": "search",
    "description": "Search for things",  # Too vague!
    "input_schema": {
        "type": "object",
        "properties": {
            "q": {"type": "string"}  # Unclear what goes here
        }
    }
}
```

### Best Practices for Tool Descriptions

**Rule of Three: Template, Not Template**

Every description should follow: "Tool to **[action]**. Use when **[situation]**."

```python
# Good descriptions
tools = [
    {
        "name": "search_documentation",
        "description": "Tool to search technical documentation. Use when you need to understand an API, library, or framework.",
    },
    {
        "name": "execute_code",
        "description": "Tool to run Python code safely in a sandbox. Use when you need to verify calculations, test logic, or generate data.",
    },
    {
        "name": "fetch_webpage",
        "description": "Tool to get full text from a URL. Use when search results aren't sufficient and you need complete context.",
    }
]

# Bad descriptions (too vague)
bad_tools = [
    {"name": "search", "description": "Search for information"},  # No action/situation clarity
    {"name": "run_code", "description": "Execute Python"},  # Missing when/why
]
```

### Parameter Design for Agent Reliability

**Principle: Make It Hard to Use Wrong**

```python
# Bad parameter design (agents make mistakes with this)
bad_params = {
    "file_path": {
        "type": "string",
        "description": "Path to file"  # Too vague—relative or absolute?
    },
    "format": {
        "type": "string",
        "description": "Format for output"  # Too open-ended
    }
}

# Good parameter design (agents use correctly)
good_params = {
    "file_path": {
        "type": "string",
        "description": "Absolute file path (e.g., /home/user/docs/file.txt). Use forward slashes even on Windows.",
        "examples": ["/data/report.csv", "/tmp/output.json"]
    },
    "format": {
        "type": "string",
        "enum": ["json", "csv", "plain_text"],
        "description": "Output format. Only these three are supported."
    }
}
```

### Implementation: Making Tools Composable

Design tools as atomic operations that can be chained:

```python
# Bad: One big tool that does everything
def analyze_data(data, cleaning_strategy, analysis_type, visualization, export_format):
    """Does too much—hard for agent to use correctly"""
    pass

# Good: Atomic tools that compose
def load_data(source: str) -> dict:
    """Load data from source. Returns raw data."""
    pass

def clean_data(data: dict, strategy: str) -> dict:
    """Clean data using specified strategy. Returns cleaned data."""
    pass

def analyze_data(data: dict, analysis_type: str) -> dict:
    """Perform statistical analysis. Returns results."""
    pass

def visualize_results(results: dict, chart_type: str) -> str:
    """Create visualization. Returns image path."""
    pass

# Agent can now compose: load → clean → analyze → visualize
```

---

## Recipe 2.2: Function Calling with JSON Schema

**Use Case**: Whenever you need the LLM to invoke tools. This is the standard interface for modern agentic systems.

### The Pattern

```python
from anthropic import Anthropic

client = Anthropic()

# Define tools with JSON schema
tools = [
    {
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or coordinates"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder for a future time",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "What to be reminded about"
                },
                "time": {
                    "type": "string",
                    "description": "When to remind (e.g., '2 hours', 'tomorrow at 9am')"
                }
            },
            "required": ["message", "time"]
        }
    }
]

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return result as string."""
    if tool_name == "get_weather":
        location = tool_input.get("location")
        unit = tool_input.get("unit", "celsius")
        # In real implementation, call weather API
        return f"Weather in {location}: 22°{unit[0].upper()}, Sunny"
    
    elif tool_name == "set_reminder":
        message = tool_input.get("message")
        time = tool_input.get("time")
        return f"Reminder set: '{message}' at {time}"
    
    return "Tool not found"

def agent_loop(user_message: str, max_iterations: int = 10) -> str:
    """Run agent with tool use."""
    messages = [{"role": "user", "content": user_message}]
    
    for _ in range(max_iterations):
        # Get model response (may include tool use)
        response = client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        # Check if we're done
        if response.stop_reason == "end_turn":
            # Extract final text response
            for block in response.content:
                if hasattr(block, 'text'):
                    return block.text
            return "No response generated"
        
        # Process tool uses
        if response.stop_reason == "tool_use":
            # Add assistant's response to messages
            messages.append({"role": "assistant", "content": response.content})
            
            # Execute tools and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_result
                    })
            
            # Add tool results back to conversation
            messages.append({"role": "user", "content": tool_results})
        else:
            # Unexpected stop reason
            break
    
    return "Max iterations reached"

# Usage
result = agent_loop("What's the weather in Paris? Then remind me to pack tomorrow at 9am.")
print(result)
```

### Common Tool Use Patterns

#### Pattern 1: Sequential Tool Use

```python
# Agent needs to use multiple tools in sequence
# Conversation flow:
# User: "How many Python packages has Alice published on PyPI?"
# Agent: search("Alice Python packages PyPI") → search result
# Agent: fetch_profile("alice_profile_url") → detailed info
# Agent: Count and return answer
```

#### Pattern 2: Parallel Tool Use

```python
# Agent can use multiple tools at once
# Conversation flow:
# User: "Compare prices for laptop X across Amazon, BestBuy, and Newegg"
# Agent: [get_price(Amazon), get_price(BestBuy), get_price(Newegg)] → all at once
# Agent: Return comparison
```

#### Pattern 3: Conditional Tool Use

```python
# Agent uses different tools based on previous results
# Conversation flow:
# User: "Find me the best rated Italian restaurant"
# Agent: search("Italian restaurants near me") → list of restaurants
# Agent: For each restaurant, get_reviews(restaurant_id) if rating > 4.5
# Agent: Return filtered list
```

---

## Recipe 2.3: Tool Composition & Helper Tools

**Use Case**: When you have many possible tools, help agents navigate the search space with helper/meta-tools.

### The Problem

When agents have 20+ tools, they suffer from decision paralysis:
- They forget which tools exist
- They misuse tools or use wrong ones
- Tool descriptions get mixed up in context

### Solution: Router Tool + Capability Tools

```python
def create_router_tool() -> dict:
    """
    A meta-tool that helps the agent understand available capabilities.
    """
    return {
        "name": "list_available_tools",
        "description": "Tool to discover what capabilities are available. Use this when you're unsure what to do next.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["search", "data", "messaging", "calculation", "all"],
                    "description": "Category of tools to list"
                }
            }
        }
    }

def execute_router(category: str) -> str:
    """Return relevant tools based on category."""
    tools_by_category = {
        "search": [
            "- search_web: Find current information online",
            "- search_docs: Search internal documentation",
            "- search_code: Search code repositories"
        ],
        "data": [
            "- load_csv: Load data from CSV file",
            "- query_database: Query SQL database",
            "- transform_data: Apply transformations to data"
        ],
        "messaging": [
            "- send_email: Send an email",
            "- send_slack: Post message to Slack",
            "- send_sms: Send text message"
        ],
        "calculation": [
            "- execute_code: Run Python code safely",
            "- calculate_stats: Calculate statistical measures"
        ],
    }
    
    if category == "all":
        output = "Available tools by category:\n"
        for cat, tools in tools_by_category.items():
            output += f"\n{cat.upper()}:\n"
            output += "\n".join(tools)
        return output
    
    return "\n".join(tools_by_category.get(category, ["No tools in this category"]))

# Use router when agent is confused
tools = [
    create_router_tool(),
    # ... all other specific tools
]
```

### Reducing Tool Context

When you have many tools, limit what's visible at once:

```python
class DynamicToolSelector:
    """Selectively expose tools based on task."""
    
    def __init__(self, all_tools: dict):
        self.all_tools = all_tools
    
    def select_for_task(self, task_description: str) -> list:
        """Return only relevant tools for this task."""
        
        task_tool_map = {
            "weather": ["get_weather", "get_location"],
            "data analysis": ["load_csv", "query_database", "calculate_stats"],
            "web research": ["search_web", "fetch_page", "extract_facts"],
            "communication": ["send_email", "send_slack", "list_contacts"],
        }
        
        # Find most relevant category
        best_match = None
        for category, tools in task_tool_map.items():
            if category.lower() in task_description.lower():
                best_match = tools
                break
        
        # Return tools + router as fallback
        relevant_tools = [self.all_tools[t] for t in (best_match or [])]
        relevant_tools.append(create_router_tool())
        
        return relevant_tools

# Usage
selector = DynamicToolSelector(all_tools)
tools_for_agent = selector.select_for_task("Find me the weather in Paris and forecast for tomorrow")
```

---

# Part III: Workflow Design

## Recipe 3.1: Prompt Chaining (Sequential Task Decomposition)

**Use Case**: Well-defined tasks that naturally break into steps. The steps are independent and can be validated.

**Why It Works**: Each step is simpler than the full task. You can add validation gates and error handling between steps.

### The Pattern

```
Step 1: Extract/Understand → Gate check
Step 2: Transform/Analyze → Gate check
Step 3: Generate/Synthesize → Final output
```

### Example: Document Analysis Pipeline

```python
async def analyze_document_chain(document_text: str, client) -> dict:
    """
    Multi-step document analysis with validation gates.
    """
    
    # Step 1: Extract key information
    extraction_prompt = f"""Analyze this document and extract:
- Main topic
- Key facts (up to 5)
- Target audience
- Document type

Document:
{document_text}

Respond in JSON format."""
    
    extraction_response = await client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=1024,
        messages=[{"role": "user", "content": extraction_prompt}]
    )
    
    extracted_info = parse_json(extraction_response.content[0].text)
    
    # Gate 1: Validate extraction
    if not all(k in extracted_info for k in ["main_topic", "key_facts"]):
        return {"error": "Extraction failed validation"}
    
    # Step 2: Generate summary
    summary_prompt = f"""Based on this information:
Topic: {extracted_info['main_topic']}
Facts: {', '.join(extracted_info['key_facts'])}
Audience: {extracted_info.get('target_audience', 'General')}

Write a 2-3 sentence summary."""
    
    summary_response = await client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=256,
        messages=[{"role": "user", "content": summary_prompt}]
    )
    
    summary = summary_response.content[0].text
    
    # Gate 2: Validate summary isn't too long
    if len(summary.split()) > 100:
        return {"error": "Summary exceeded word limit"}
    
    # Step 3: Create recommendations
    recommendations_prompt = f"""For a {extracted_info['target_audience']} audience, 
what are the top 3 actions based on this summary:

{summary}

Format as JSON with "recommendations": [list]"""
    
    recommendations_response = await client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=512,
        messages=[{"role": "user", "content": recommendations_prompt}]
    )
    
    recommendations = parse_json(recommendations_response.content[0].text)
    
    return {
        "extracted_info": extracted_info,
        "summary": summary,
        "recommendations": recommendations.get("recommendations", [])
    }

def parse_json(text: str) -> dict:
    """Extract JSON from model response."""
    import json
    import re
    
    # Find JSON block
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    return {}
```

### When Prompt Chaining Works Best

| Scenario | Suitable? | Why |
|----------|-----------|-----|
| Extracting → Summarizing → Recommending | ✅ Yes | Clear sequential steps |
| Analyzing → Identifying issues → Proposing fixes | ✅ Yes | Each step builds on previous |
| Classifying → Looking up → Formatting | ✅ Yes | Steps are independent |
| Open-ended brainstorming | ❌ No | No clear sequence |
| Iterative refinement | ❌ No | Needs loops, not chains |

### Gates and Error Handling

```python
def validated_chain(steps: list, client) -> dict:
    """
    Generic prompt chaining with validation gates.
    """
    results = {}
    
    for step in steps:
        # Execute step
        response = client.messages.create(
            model=step.get("model", "claude-3-5-sonnet"),
            max_tokens=step.get("max_tokens", 1024),
            messages=[{"role": "user", "content": step["prompt"]}]
        )
        
        result = response.content[0].text
        
        # Apply validation gate if provided
        if "validation_fn" in step:
            is_valid = step["validation_fn"](result)
            if not is_valid:
                return {
                    "error": f"Step '{step['name']}' failed validation",
                    "step": step["name"],
                    "output": result
                }
        
        results[step["name"]] = result
        
        # Pass output to next step if specified
        if "next_prompt_template" in step:
            next_prompt = step["next_prompt_template"].format(
                previous_output=result
            )
            step["prompt"] = next_prompt
    
    return results
```

---

## Recipe 3.2: Routing (Task Specialization)

**Use Case**: Different task types need different handling. Routing classifies input and dispatches to specialized handlers.

**Why It Works**: Specialized prompts beat general prompts. A classifier first, then specialized solver second, beats one general solver.

### The Pattern

```
1. Classify the input
2. Route to specialized sub-agent
3. Return result
```

### Example: Customer Support Router

```python
async def customer_support_router(customer_message: str, client) -> dict:
    """
    Routes customer inquiries to specialized handlers.
    """
    
    # Step 1: Classify the inquiry
    classification_prompt = f"""Classify this customer message into ONE category:
- General Question
- Billing/Payment Issue
- Technical Problem
- Refund Request
- Account Management
- Product Feedback

Customer: "{customer_message}"

Respond with ONLY the category name."""
    
    classification = await client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=50,
        messages=[{"role": "user", "content": classification_prompt}]
    )
    
    category = classification.content[0].text.strip()
    
    # Step 2: Route to specialized handler
    handlers = {
        "General Question": handle_general_question,
        "Billing/Payment Issue": handle_billing,
        "Technical Problem": handle_technical,
        "Refund Request": handle_refund,
        "Account Management": handle_account,
        "Product Feedback": handle_feedback,
    }
    
    handler = handlers.get(category, handle_general_question)
    response = await handler(customer_message, client)
    
    return {
        "category": category,
        "response": response
    }

async def handle_billing(message: str, client) -> str:
    """Specialized handler for billing issues."""
    prompt = f"""You are a billing specialist. 
Customer query: {message}

Provide a helpful response about billing, payments, and charges.
If they mention a specific charge, ask for the date and amount."""
    
    response = await client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

async def handle_technical(message: str, client) -> str:
    """Specialized handler for technical problems."""
    prompt = f"""You are a technical support specialist.
Customer query: {message}

Provide troubleshooting steps. Be specific about:
1. What to check first
2. Common causes
3. Step-by-step solutions"""
    
    response = await client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# Similar handlers for other categories...
```

### Dynamic Routing with Confidence

```python
def routing_with_fallback(message: str, client) -> dict:
    """
    Routes with confidence scores. If confidence is low, escalate.
    """
    
    routing_prompt = f"""Classify this message and provide confidence.
Message: "{message}"

Respond in JSON:
{{"category": "...", "confidence": 0.0-1.0, "requires_escalation": boolean}}"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=100,
        messages=[{"role": "user", "content": routing_prompt}]
    )
    
    import json
    result = json.loads(response.content[0].text)
    
    if result["confidence"] < 0.7 or result["requires_escalation"]:
        return {
            "response": "I'm not confident in my understanding. Let me transfer you to a specialist.",
            "escalate": True,
            "likely_category": result["category"]
        }
    
    return {
        "response": "Handling with handler: " + result["category"],
        "category": result["category"],
        "escalate": False
    }
```

---

## Recipe 3.3: Parallelization (Sectioning and Voting)

**Use Case**: Large tasks that can split into independent subtasks. Or when you want high confidence through ensemble voting.

**Why It Works**: Two reasons: (1) Speed—parallel tasks complete faster, (2) Accuracy—multiple perspectives find better answers.

### Pattern 1: Sectioning (Large Input Split)

```python
import asyncio

async def analyze_long_document_parallel(document: str, client) -> dict:
    """
    Split long document into sections, analyze in parallel, synthesize results.
    """
    
    # Step 1: Split into sections
    split_prompt = f"""Split this document into 3-4 major sections.
For each section, provide a section header and the text in that section.

Document (first 5000 chars):
{document[:5000]}...

Respond in JSON: {{"sections": [{{"header": "...", "content": "..."}}]}}"""
    
    split_response = await client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=1024,
        messages=[{"role": "user", "content": split_prompt}]
    )
    
    import json
    sections = json.loads(split_response.content[0].text)["sections"]
    
    # Step 2: Analyze each section in parallel
    async def analyze_section(section: dict) -> dict:
        analysis_prompt = f"""Analyze this section and provide:
- Key points (up to 3)
- Sentiment
- Importance (1-5 scale)

Section: {section['header']}
{section['content'][:1000]}...

Respond in JSON."""
        
        response = await client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=512,
            messages=[{"role": "user", "content": analysis_prompt}]
        )
        
        try:
            analysis = json.loads(response.content[0].text)
        except:
            analysis = {"error": "Parse failed", "raw": response.content[0].text}
        
        analysis["section_header"] = section['header']
        return analysis
    
    # Run all analyses in parallel
    analyses = await asyncio.gather(*[
        analyze_section(section) for section in sections
    ])
    
    # Step 3: Synthesize findings
    synthesis_prompt = f"""These are the key findings from analyzing a document by sections:

{json.dumps(analyses, indent=2)}

Provide:
1. Overall summary (2-3 sentences)
2. Most important points (top 3)
3. Recommendations (if any)"""
    
    synthesis_response = await client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=512,
        messages=[{"role": "user", "content": synthesis_prompt}]
    )
    
    return {
        "section_analyses": analyses,
        "synthesis": synthesis_response.content[0].text
    }
```

### Pattern 2: Voting (Ensemble for Confidence)

```python
import asyncio
from collections import Counter

async def high_confidence_answer(question: str, client, 
                                 num_votes: int = 5) -> dict:
    """
    Get answer from multiple agents, use majority voting.
    """
    
    async def get_answer(prompt: str) -> str:
        response = await client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=256,
            temperature=0.7,  # Diversity
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    # Get multiple answers in parallel
    answers = await asyncio.gather(*[
        get_answer(question) for _ in range(num_votes)
    ])
    
    # Voting
    answer_counts = Counter(answers)
    most_common, vote_count = answer_counts.most_common(1)[0]
    confidence = vote_count / num_votes
    
    return {
        "answer": most_common,
        "confidence": confidence,
        "num_votes": num_votes,
        "agreement_ratio": vote_count / num_votes,
        "all_answers": answers
    }
```

### When to Use Parallelization

| Use Case | Type | Benefit |
|----------|------|---------|
| Split 10K-word document | Sectioning | Accuracy (more detail per agent) |
| Analyze 100 customer reviews | Sectioning | Speed (batch process reviews) |
| Answer complex question with voting | Voting | Confidence through ensemble |
| Review code from multiple angles | Voting | Completeness (catch edge cases) |

---

## Recipe 3.4: Evaluator-Optimizer Loop (Iterative Refinement)

**Use Case**: Tasks where quality improves with feedback. Writing, design, complex problem solving.

**Why It Works**: One agent generates, another critiques, first agent improves. This mirrors human writing/design processes.

### The Pattern

```
Generate → Evaluate → Is good? → Yes: Return
                    → No: Improve → Loop
```

### Python Implementation

```python
async def iterative_refinement(initial_prompt: str, client, max_iterations: int = 3) -> dict:
    """
    Generate content, evaluate, refine iteratively.
    """
    
    current_output = None
    feedback_history = []
    
    for iteration in range(max_iterations):
        # Step 1: Generate (or improve)
        if iteration == 0:
            generation_prompt = initial_prompt
        else:
            # Use feedback to improve
            generation_prompt = f"""{initial_prompt}

Previous version feedback:
{feedback_history[-1]}

Generate an improved version addressing the feedback."""
        
        generation_response = await client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": generation_prompt}]
        )
        
        current_output = generation_response.content[0].text
        
        # Step 2: Evaluate
        evaluation_prompt = f"""Evaluate this text on:
1. Clarity (is it clear?)
2. Completeness (does it cover everything?)
3. Tone (is it appropriate?)
4. Quality (is it well-written?)

Text to evaluate:
{current_output}

Respond in JSON:
{{"clarity": 1-5, "completeness": 1-5, "tone": 1-5, "quality": 1-5, "feedback": "..."}}"""
        
        evaluation_response = await client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=512,
            messages=[{"role": "user", "content": evaluation_prompt}]
        )
        
        import json
        evaluation = json.loads(evaluation_response.content[0].text)
        feedback_history.append(evaluation)
        
        # Check if satisfied
        average_score = sum([
            evaluation["clarity"],
            evaluation["completeness"],
            evaluation["tone"],
            evaluation["quality"]
        ]) / 4
        
        if average_score >= 4.0 or iteration == max_iterations - 1:
            return {
                "final_output": current_output,
                "final_evaluation": evaluation,
                "iterations": iteration + 1,
                "feedback_history": feedback_history,
                "satisfied": average_score >= 4.0
            }
    
    return {
        "final_output": current_output,
        "final_evaluation": feedback_history[-1] if feedback_history else None,
        "iterations": max_iterations,
        "feedback_history": feedback_history,
        "satisfied": False
    }

# Usage
result = await iterative_refinement(
    "Write a product description for an AI agent framework",
    client,
    max_iterations=3
)
print(f"Final output:\n{result['final_output']}")
print(f"Satisfied: {result['satisfied']}")
```

### Specialized Evaluators

Different tasks benefit from specialized evaluators:

```python
class SpecializedEvaluators:
    """Different evaluator prompts for different tasks."""
    
    @staticmethod
    def code_evaluator_prompt(code: str) -> str:
        return f"""Evaluate this Python code on:
1. Correctness (does it work?)
2. Readability (is it clear?)
3. Efficiency (is it fast enough?)
4. Safety (any security issues?)

Code:
{code}

Provide JSON with scores 1-5 and specific feedback."""
    
    @staticmethod
    def writing_evaluator_prompt(text: str) -> str:
        return f"""Evaluate this writing on:
1. Clarity (is meaning clear?)
2. Engagement (is it interesting?)
3. Accuracy (is it factually correct?)
4. Grammar (correct English?)

Text:
{text}

Provide JSON with scores and feedback."""
    
    @staticmethod
    def design_evaluator_prompt(description: str) -> str:
        return f"""Evaluate this design proposal on:
1. Feasibility (can it be built?)
2. Usability (is it easy to use?)
3. Aesthetics (is it attractive?)
4. Scalability (will it grow?)

Design:
{description}

Provide JSON with scores and feedback."""
```

---

# Part IV: Testing & Evaluation

## Recipe 4.1: Building a Test Suite for Agents

**Core Principle**: Test agents like you'd test software—with deterministic tests, edge cases, and load testing.

### Test Categories

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class AgentTest:
    name: str
    description: str
    input: str
    expected_contains: list  # Strings that should be in output
    expected_not_contains: list  # Strings that should NOT be in output
    evaluator: Callable[[str], bool] = None  # Custom validation function
    max_tokens: int = 1024
    timeout_seconds: float = 30.0

# Example test cases
test_suite = [
    # Category 1: Functional correctness
    AgentTest(
        name="math_addition",
        description="Can the agent do basic math?",
        input="What is 15 + 27?",
        expected_contains=["42"],
        expected_not_contains=[]
    ),
    
    # Category 2: Edge cases
    AgentTest(
        name="empty_input",
        description="Does it handle empty input gracefully?",
        input="",
        expected_contains=["clarify", "help", "error"],  # Helpful response
        expected_not_contains=["Traceback", "Error:", "null"]  # No crashes
    ),
    
    # Category 3: Instruction following
    AgentTest(
        name="format_instruction",
        description="Does it follow format instructions?",
        input="List 3 benefits of Python. Format as numbered list.",
        expected_contains=["1.", "2.", "3."],  # Must follow format
        expected_not_contains=["•", "-", ""],  # Not using other formats
    ),
    
    # Category 4: Tool use correctness
    AgentTest(
        name="tool_invocation",
        description="Does it correctly invoke the search tool?",
        input="Search for 'Python 3.12 release date' and tell me when it was released.",
        expected_contains=["search", "2023"],  # Tool was used
        expected_not_contains=["I don't have access"],  # Not refusing
        evaluator=lambda output: "2023" in output and len(output) > 50
    ),
]

async def run_test_suite(agent_fn: Callable, tests: list) -> dict:
    """
    Run all tests and return results.
    """
    results = {
        "total": len(tests),
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "details": []
    }
    
    for test in tests:
        try:
            # Run agent
            output = await asyncio.wait_for(
                agent_fn(test.input),
                timeout=test.timeout_seconds
            )
            
            # Check expected content
            passed = True
            failure_reason = None
            
            for expected in test.expected_contains:
                if expected.lower() not in output.lower():
                    passed = False
                    failure_reason = f"Missing expected: '{expected}'"
                    break
            
            if passed:
                for not_expected in test.expected_not_contains:
                    if not_expected.lower() in output.lower():
                        passed = False
                        failure_reason = f"Unexpectedly found: '{not_expected}'"
                        break
            
            # Run custom evaluator if provided
            if passed and test.evaluator:
                try:
                    passed = test.evaluator(output)
                    if not passed:
                        failure_reason = "Custom evaluator returned False"
                except Exception as e:
                    failure_reason = f"Evaluator error: {str(e)}"
                    passed = False
            
            # Record result
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
            
            results["details"].append({
                "name": test.name,
                "passed": passed,
                "reason": failure_reason,
                "output_length": len(output)
            })
        
        except asyncio.TimeoutError:
            results["errors"] += 1
            results["details"].append({
                "name": test.name,
                "passed": False,
                "reason": f"Timeout after {test.timeout_seconds}s",
                "output_length": 0
            })
        
        except Exception as e:
            results["errors"] += 1
            results["details"].append({
                "name": test.name,
                "passed": False,
                "reason": f"Exception: {str(e)}",
                "output_length": 0
            })
    
    return results

# Usage
async def my_agent(query: str) -> str:
    # Your agent implementation
    pass

results = await run_test_suite(my_agent, test_suite)
print(f"Passed: {results['passed']}/{results['total']}")
for detail in results['details']:
    status = "✓" if detail['passed'] else "✗"
    print(f"{status} {detail['name']}: {detail['reason']}")
```

### LLM-as-Judge Evaluation

For subjective qualities, use another LLM as evaluator:

```python
async def llm_judge_evaluation(output: str, client, criteria: list) -> dict:
    """
    Use an LLM to evaluate output on subjective criteria.
    """
    criteria_description = "\n".join([
        f"- {c['name']}: {c['description']}" for c in criteria
    ])
    
    judge_prompt = f"""You are an expert evaluator. 
Rate this output on the following criteria (1-5 scale):

{criteria_description}

Output to evaluate:
{output}

Respond in JSON:
{{
  "scores": {{"criterion_name": score, ...}},
  "justification": "...",
  "overall_quality": 1-5
}}"""
    
    response = await client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=512,
        messages=[{"role": "user", "content": judge_prompt}]
    )
    
    import json
    evaluation = json.loads(response.content[0].text)
    
    average_score = sum(evaluation["scores"].values()) / len(evaluation["scores"])
    evaluation["average_score"] = average_score
    
    return evaluation
```

---

## Recipe 4.2: Benchmarking Against Baselines

**Use Case**: Comparing your agent against baseline approaches.

```python
async def benchmark_agent_variants(task: str, variants: dict, client, num_runs: int = 5):
    """
    Compare multiple agent implementations on the same task.
    """
    import statistics
    
    results = {}
    
    for variant_name, variant_fn in variants.items():
        scores = []
        latencies = []
        
        for run in range(num_runs):
            import time
            start = time.time()
            
            try:
                output = await asyncio.wait_for(variant_fn(task), timeout=30)
                latency = time.time() - start
                
                # Evaluate output
                eval_result = await llm_judge_evaluation(
                    output, client,
                    [
                        {"name": "correctness", "description": "Is the answer correct?"},
                        {"name": "clarity", "description": "Is it clear?"},
                        {"name": "completeness", "description": "Does it cover the topic?"}
                    ]
                )
                
                scores.append(eval_result["average_score"])
                latencies.append(latency)
            
            except Exception as e:
                scores.append(0)  # Failed run
                latencies.append(30)  # Timeout
        
        results[variant_name] = {
            "mean_score": statistics.mean(scores),
            "stdev_score": statistics.stdev(scores) if len(scores) > 1 else 0,
            "mean_latency": statistics.mean(latencies),
            "success_rate": sum(1 for s in scores if s > 0) / len(scores)
        }
    
    # Print comparison
    print("\nBenchmark Results:")
    print("-" * 60)
    for variant, metrics in results.items():
        print(f"\n{variant}:")
        print(f"  Quality (avg): {metrics['mean_score']:.2f} ± {metrics['stdev_score']:.2f}")
        print(f"  Latency (avg): {metrics['mean_latency']:.2f}s")
        print(f"  Success rate: {metrics['success_rate']:.1%}")
    
    return results
```

---

# Advanced Recipes

## Recipe 5.1: Multi-Agent Orchestration (Divide & Conquer)

**Use Case**: Complex tasks requiring multiple specialized agents. Each agent handles part of the problem, results are synthesized.

### Architecture

```
User Query
    ↓
Main Orchestrator Agent (Planner)
    ↓
Task Decomposition
    ↓
[Specialist 1] [Specialist 2] [Specialist 3] (Parallel)
    ↓
Synthesizer Agent
    ↓
Final Answer
```

### Implementation

```python
import asyncio

class MultiAgentOrchestrator:
    def __init__(self, client):
        self.client = client
    
    async def decompose_task(self, task: str) -> list:
        """Break task into subtasks."""
        decomposition_prompt = f"""Break this task into 2-4 independent subtasks:

Task: {task}

Respond in JSON:
{{"subtasks": [
    {{"id": 1, "description": "...", "required_expertise": "..."}},
    ...
]}}"""
        
        response = await self.client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=512,
            messages=[{"role": "user", "content": decomposition_prompt}]
        )
        
        import json
        result = json.loads(response.content[0].text)
        return result["subtasks"]
    
    async def solve_subtask(self, subtask: dict, expertise: str) -> str:
        """Specialist agent for a subtask."""
        prompt = f"""You are an expert in {expertise}.

Solve this subtask:
{subtask['description']}

Provide a detailed, focused answer on just this subtask."""
        
        response = await self.client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    async def synthesize_results(self, task: str, subtask_results: list) -> str:
        """Combine subtask results into coherent answer."""
        synthesis_prompt = f"""Original task: {task}

Subtask results:
{chr(10).join([f'{i+1}. {r}' for i, r in enumerate(subtask_results)])}

Synthesize these results into a comprehensive, coherent answer to the original task."""
        
        response = await self.client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        
        return response.content[0].text
    
    async def solve(self, task: str) -> str:
        """Orchestrate the entire multi-agent process."""
        # Decompose
        subtasks = await self.decompose_task(task)
        
        # Solve in parallel
        solutions = await asyncio.gather(*[
            self.solve_subtask(subtask, subtask["required_expertise"])
            for subtask in subtasks
        ])
        
        # Synthesize
        final_answer = await self.synthesize_results(task, solutions)
        
        return final_answer

# Usage
orchestrator = MultiAgentOrchestrator(client)
result = await orchestrator.solve("Design a mobile app for fitness tracking")
```

---

## Recipe 5.2: Self-Reflection & Error Correction

**Use Case**: Complex tasks where agents can learn from mistakes. Coding, debugging, content generation.

```python
async def self_reflecting_agent(task: str, client, max_reflections: int = 3) -> dict:
    """
    Agent that critiques its own output and improves.
    """
    
    current_attempt = None
    reflection_history = []
    
    for attempt in range(max_reflections):
        # Generate attempt
        if attempt == 0:
            generation_prompt = task
        else:
            # Use previous reflection to guide improvement
            generation_prompt = f"""{task}

Previous attempt feedback:
{reflection_history[-1]['reflection']}

Generate an improved solution addressing the feedback."""
        
        attempt_response = await client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": generation_prompt}]
        )
        
        current_attempt = attempt_response.content[0].text
        
        # Self-reflect
        reflection_prompt = f"""You just generated this solution:

{current_attempt}

Critically evaluate it:
1. What are the weaknesses?
2. What could be improved?
3. Are there edge cases you missed?
4. Is there a better approach?

Provide JSON: {{"weaknesses": [...], "improvements": [...], "needs_retry": boolean}}"""
        
        reflection_response = await client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=512,
            messages=[{"role": "user", "content": reflection_prompt}]
        )
        
        import json
        reflection = json.loads(reflection_response.content[0].text)
        reflection_history.append({
            "attempt": attempt + 1,
            "solution": current_attempt,
            "reflection": reflection
        })
        
        # Check if satisfied
        if not reflection.get("needs_retry", False) or attempt == max_reflections - 1:
            break
    
    return {
        "final_solution": current_attempt,
        "reflection_history": reflection_history,
        "total_attempts": len(reflection_history)
    }
```

---

# Common Pitfalls & Solutions

## Pitfall 1: Tool Descriptions Are Too Vague

**Problem**: Agent doesn't understand when to use a tool.

```python
# Bad
{"name": "search", "description": "Search"}

# Good
{
    "name": "search_docs",
    "description": "Tool to search internal documentation. Use when you need to understand technical concepts, APIs, or system architecture."
}
```

**Solution**: Follow the "Tool to X. Use when Y." template always.

---

## Pitfall 2: Agents Make Random Tool Call Errors

**Problem**: Agent calls tools with wrong parameters or calls non-existent tools.

**Solutions**:

1. **Limit tool exposure**: Only show relevant tools for the task
2. **Use enums**: Force choices into valid set
   ```python
   "format": {"type": "string", "enum": ["json", "csv"]}  # Can't be invalid
   ```
3. **Add examples to descriptions**
   ```python
   "description": "Search docs. Example: search_docs(query='async/await in Python')"
   ```

---

## Pitfall 3: Context Window Overload

**Problem**: Too many tools and examples cause context bloat and errors.

**Solutions**:

1. **Dynamic tool selection**: Only include relevant tools
   ```python
   selected_tools = [t for t in all_tools if t["relevance"] in ["high", "medium"]]
   ```
2. **Compress examples**: Use 3-5 diverse examples, not 20
3. **Tool routing**: Add a router tool to help navigate

---

## Pitfall 4: Evaluation Metrics Are Useless

**Problem**: Testing if agent "ran without error" isn't real testing.

**Solutions**:

1. **Test for correctness**: Check output for expected content
2. **Use LLM judges**: Evaluate subjective qualities
3. **Test edge cases**: Empty input, very long input, contradictory requests
4. **Measure latency**: Track performance over time

---

## Pitfall 5: Infinite Loops in Agents

**Problem**: Agent keeps calling tools without making progress.

**Solutions**:

1. **Set max iterations**
   ```python
   max_iterations = 10
   ```
2. **Track state changes**: If state hasn't improved, exit
   ```python
   if current_state == previous_state:
       break
   ```
3. **Explicit stopping criteria**
   ```python
   if "Final Answer:" in response:
       return extract_answer(response)
   ```

---

## Conclusion

This cookbook provides battle-tested recipes for the three domains of agent building:

**Prompting**: Start with zero-shot CoT for reasoning, add few-shot examples, move to ReACT or self-consistency when needed, use specialized patterns (ToT, role-based) for complex problems.

**Tool Design**: Keep descriptions simple and clear ("Tool to X. Use when Y."), use strong typing to prevent errors, compose tools from atomic operations, test extensively.

**Workflow Design**: Match pattern to task type—prompt chaining for sequential steps, routing for classification, parallelization for independent work, evaluator-optimizer loops for iterative refinement.

**Testing**: Build deterministic test suites, use LLM judges for subjective evaluation, benchmark against baselines, test edge cases and error scenarios.

The key to production-grade agents isn't complexity—it's **simplicity, clarity, and rigorous testing**. Start simple, measure continuously, add complexity only when simpler patterns fail.

---

## Quick Reference: Decision Matrix

| Task Type | Prompting Strategy | Workflow Pattern | Tool Approach |
|-----------|-------------------|------------------|---------------|
| **Simple QA** | Few-shot CoT | None (direct call) | Search + Retrieval |
| **Multi-step reasoning** | CoT + Self-Consistency | Prompt Chaining | Specialized by step |
| **Complex analysis** | Tree of Thought | Orchestrator-Workers | Divide & conquer |
| **Multiple categories** | Routing classifier | Routing | Specialized per category |
| **Real-time interaction** | ReACT | Tool loop | Environment interaction |
| **Content quality** | Role-based + Evaluator | Evaluator-Optimizer | Refinement tools |
| **High confidence needed** | Self-Consistency | Parallelization (voting) | Ensemble tools |

---

**Version**: 1.0  
**Last Updated**: January 2026  
**Status**: Ready for production use

*This cookbook synthesizes research from AFLOW (arXiv:2410.10762), ADAS (arXiv:2408.08435), Anthropic's agent research, and production practices from dozens of teams building AI agents at scale.*
