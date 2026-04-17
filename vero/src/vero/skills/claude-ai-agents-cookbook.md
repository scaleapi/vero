# AI Agent Design Cookbook

A practical guide for building effective AI agents with proven strategies for prompting, tool design, and workflow orchestration.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Prompting Strategies](#prompting-strategies)
3. [Tool Design Patterns](#tool-design-patterns)
4. [Workflow Architectures](#workflow-architectures)
5. [Implementation Examples](#implementation-examples)

---

## Introduction

### What Are AI Agents?

**Agents** are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks. They differ from **workflows**, which use predefined code paths to orchestrate LLMs and tools.

### When to Use Agents

- **Use simple prompts** when single LLM calls suffice
- **Use workflows** for predictable, well-defined tasks requiring consistency
- **Use agents** when flexibility and model-driven decision-making are needed at scale

### Core Principles

1. **Maintain simplicity** in design
2. **Prioritize transparency** by showing planning steps
3. **Craft interfaces carefully** through documentation and testing

---

## Prompting Strategies

### 1. Zero-Shot Prompting

**When to use:** Tasks within the model's training distribution where no examples are needed.

**Recipe:**
```python
prompt = """
Task: {task_description}

Requirements:
- {requirement_1}
- {requirement_2}

Output format: {desired_format}
"""
```

**Best practices:**
- Be explicit about expectations
- Define output format clearly
- Include constraints and boundaries
- Use direct, unambiguous language

---

### 2. Few-Shot Prompting

**When to use:** When you need specific formatting, tone, or approach that's easier to demonstrate than describe.

**Recipe:**
```python
prompt = """
You are a {role}. Here are examples of how to handle similar tasks:

Example 1:
Input: {example_input_1}
Output: {example_output_1}

Example 2:
Input: {example_input_2}
Output: {example_output_2}

Now handle this:
Input: {actual_input}
Output:
"""
```

**Best practices:**
- Use 2-5 diverse examples covering edge cases
- Keep examples structurally identical
- Select representative samples, not outliers
- Order examples from simple to complex

---

### 3. Chain-of-Thought (CoT) Prompting

**When to use:** Complex reasoning tasks requiring multi-step logic, math, or analysis.

**Recipe - Few-Shot CoT:**
```python
prompt = """
Solve these problems step by step:

Q: If a store has 15 apples and sells 6, then receives 8 more, how many does it have?
A: Let me think through this:
1. Starting amount: 15 apples
2. After selling 6: 15 - 6 = 9 apples
3. After receiving 8: 9 + 8 = 17 apples
Answer: 17 apples

Q: {your_question}
A: Let me think through this:
"""
```

**Recipe - Zero-Shot CoT:**
```python
prompt = """
{question}

Let's approach this step by step:
"""
# or simply add: "Think step by step."
```

**Best practices:**
- Works best with models >50B parameters
- Explicitly request reasoning steps
- Use for math, logic puzzles, decision-making
- Combine with few-shot for complex domains

---

### 4. ReAct (Reasoning + Acting)

**When to use:** Tasks requiring both thinking and tool use in an iterative loop.

**Recipe:**
```python
system_prompt = """
You solve problems by alternating between Thought, Action, and Observation.

Format:
Thought: [Your reasoning about what to do next]
Action: [Tool name and parameters]
Observation: [Result from the tool]
... (repeat as needed)
Thought: I now have enough information
Answer: [Final answer]

Available tools:
- Search[query]: Search the web
- Calculate[expression]: Evaluate math
- GetWeather[location]: Get weather data
"""
```

**Best practices:**
- Make reasoning explicit before each action
- Use tool results to inform next steps
- Ideal for research, data gathering, multi-step problems
- More flexible than pre-planned workflows

---

### 5. Self-Consistency Prompting

**When to use:** High-stakes decisions where you want multiple reasoning paths.

**Recipe:**
```python
def self_consistency(prompt, n=5):
    """Generate multiple solutions and pick most common answer"""
    responses = []
    for i in range(n):
        response = llm.generate(prompt + "\nLet's solve this step by step.")
        responses.append(extract_answer(response))
    
    # Return most frequent answer
    return most_common(responses)
```

**Best practices:**
- Generate 3-7 reasoning paths
- Use majority voting for final answer
- Effective for math, logic, critical decisions
- Increases latency but improves accuracy

---

### 6. Role-Based Prompting

**When to use:** When you need domain expertise or specific communication style.

**Recipe:**
```python
prompt = """
You are a {specific_role} with expertise in {domain}.

Your characteristics:
- {trait_1}
- {trait_2}
- {trait_3}

User query: {query}

Respond as this expert would, using appropriate terminology and perspective.
"""
```

**Examples:**
- "You are a senior software architect reviewing code for security vulnerabilities"
- "You are a patient teacher explaining complex topics to beginners"
- "You are a critical analyst identifying flaws in arguments"

---

### 7. Meta Prompting

**When to use:** Creating reusable prompt templates that work across similar tasks.

**Recipe:**
```python
meta_prompt = """
For any coding problem, follow this structure:
1. Understand the requirements
2. Break down into steps
3. Write the solution
4. Test with edge cases

Now apply this to: {specific_task}
"""
```

**Best practices:**
- Define logical structure, not specific content
- Use for token efficiency
- Good for repetitive tasks with varying inputs

---

### 8. Structured Output Specification

**When to use:** When you need reliable JSON, XML, or formatted data.

**Recipe:**
```python
prompt = """
Extract information and return ONLY valid JSON with this exact structure:

{
  "name": "string",
  "age": number,
  "skills": ["string"],
  "active": boolean
}

Text to analyze: {input_text}

JSON output:
"""
```

**Best practices:**
- Provide exact schema with types
- Use "ONLY return" to prevent extra text
- Validate output programmatically
- Consider using structured output APIs when available

---

## Tool Design Patterns

### Core Principles

> "Invest as much effort in agent-computer interfaces (ACI) as you would in human-computer interfaces (HCI)"

### 1. Clear Tool Documentation

**Recipe:**

```python
def search_database(
    query: str,
    filters: dict = None,
    max_results: int = 10
) -> list:
    """
    Search the product database with natural language queries.
    
    Args:
        query: Natural language search query (e.g., "red shoes under $50")
        filters: Optional filters like {"category": "footwear", "in_stock": True}
        max_results: Maximum number of results to return (default: 10, max: 100)
    
    Returns:
        List of matching products with name, price, and availability
    
    Examples:
        search_database("wireless headphones")
        search_database("laptop", filters={"price_max": 1000})
    
    Edge cases:
        - Empty query returns error
        - No matches returns empty list
        - Invalid filters are ignored with warning
    """
```

**Best practices:**
- Treat tool descriptions as documentation for a junior developer
- Include examples of correct usage
- Document edge cases and error handling
- Use clear, descriptive parameter names
- Specify types and constraints

---

### 2. Low-Friction Tool Formats

**Bad - High cognitive overhead:**
```python
def edit_file(file_path: str, diff: str):
    """Apply a unified diff to a file"""
    # Requires model to count lines, format headers correctly
```

**Good - Natural format:**
```python
def edit_file(file_path: str, old_content: str, new_content: str):
    """
    Replace old_content with new_content in file.
    
    The model just writes what it wants to replace and what to replace it with.
    No line counting or special formatting required.
    """
```

**Best practices:**
- Minimize formatting overhead (escaping, line counting)
- Keep formats close to natural text
- Give the model room to "think" before committing
- Avoid requiring exact counts or complex structures

---

### 3. Tool Examples Library

#### File Operations
```python
def read_file(path: str, start_line: int = None, end_line: int = None) -> str:
    """Read file contents, optionally specifying line range for large files"""

def write_file(path: str, content: str, mode: str = 'w') -> bool:
    """Write content to file. Mode: 'w' (overwrite) or 'a' (append)"""

def list_directory(path: str, pattern: str = None) -> list:
    """List files in directory, optionally filtered by glob pattern"""
```

#### Web & API
```python
def web_search(query: str, num_results: int = 5) -> list:
    """Search the web and return top results with titles, URLs, snippets"""

def fetch_url(url: str, timeout: int = 10) -> str:
    """Fetch and return the text content of a web page"""

def api_call(endpoint: str, method: str = 'GET', data: dict = None) -> dict:
    """Make HTTP request to API endpoint"""
```

#### Data Processing
```python
def query_database(sql: str, params: list = None) -> list:
    """Execute SQL query and return results"""

def process_csv(file_path: str, operation: str, **kwargs) -> dict:
    """Perform operations on CSV: 'filter', 'aggregate', 'transform'"""

def calculate(expression: str) -> float:
    """Safely evaluate mathematical expressions"""
```

#### Code Execution
```python
def execute_python(code: str, timeout: int = 30) -> dict:
    """
    Execute Python code in isolated environment.
    Returns: {"output": str, "error": str, "execution_time": float}
    """

def run_tests(test_file: str) -> dict:
    """Run test file and return results with pass/fail status"""
```

---

### 4. Poka-Yoke (Error-Proofing) Tools

**Problem:** Model makes mistakes with relative paths after changing directories.

**Solution:** Require absolute paths
```python
# Before
def edit_code(file_path: str, changes: str):
    """file_path can be relative - error prone!"""

# After  
def edit_code(absolute_path: str, changes: str):
    """
    Edit a code file. Path MUST be absolute.
    Use get_absolute_path(relative) if needed.
    
    Wrong: edit_code("src/main.py", ...)
    Right: edit_code("/home/user/project/src/main.py", ...)
    """
```

**Best practices:**
- Design tools to prevent common mistakes
- Use parameter names that clarify requirements
- Provide helper tools for error-prone operations
- Make the "right way" the easy way

---

### 5. Dual-Use Tool Design

**Concept:** Tools usable by both agents AND humans via code/CLI.

```python
class FileEditor:
    """Tool that works for both agents and developers"""
    
    def edit_file_agent(self, path: str, old: str, new: str) -> dict:
        """Agent-friendly: natural language interface"""
        return self._edit(path, old, new)
    
    def edit_file_cli(self, path: str, pattern: str, replacement: str) -> dict:
        """Developer-friendly: regex support"""
        return self._edit(path, pattern, replacement, use_regex=True)
    
    def _edit(self, path: str, old: str, new: str, use_regex: bool = False):
        """Shared implementation"""
```

---

### 6. Progressive Tool Discovery

**Pattern:** Start with basic tools, add complexity as needed.

```python
# Level 1: Basic tools
initial_tools = [
    "read_file", 
    "write_file", 
    "search_web"
]

# Level 2: Add based on task type
if task_type == "coding":
    tools.extend(["execute_code", "run_tests", "lint_code"])
elif task_type == "research":
    tools.extend(["search_papers", "summarize_pdf", "extract_citations"])

# Level 3: Add based on agent's actions
if agent_attempted("database_query"):
    tools.append("query_database")
```

**Best practices:**
- Start minimal to reduce decision paralysis
- Add tools contextually as needed
- Monitor which tools are actually used

---

## Workflow Architectures

### 1. Prompt Chaining

**When to use:** Task can be decomposed into clear sequential steps.

**Architecture:**
```
User Input → LLM1 (Outline) → Gate Check → LLM2 (Draft) → Gate Check → LLM3 (Polish) → Output
```

**Implementation:**
```python
def prompt_chain(user_input: str) -> str:
    # Step 1: Create outline
    outline = llm_call(
        f"Create an outline for: {user_input}",
        model="fast-model"
    )
    
    # Gate: Validate outline has required sections
    if not validate_outline(outline):
        return "Error: Invalid outline structure"
    
    # Step 2: Write draft
    draft = llm_call(
        f"Write a draft based on this outline:\n{outline}",
        model="quality-model"
    )
    
    # Step 3: Polish
    final = llm_call(
        f"Polish this draft:\n{draft}\n\nMake it more concise and clear.",
        model="quality-model"
    )
    
    return final
```

**Best practices:**
- Add programmatic gates between steps
- Use faster/cheaper models for simple steps
- Each step should make the task simpler
- Clear handoff between stages

**Examples:**
- Marketing copy → Translation
- Requirements → Design → Implementation
- Research → Outline → Writing

---

### 2. Routing

**When to use:** Different input types need different specialized handling.

**Architecture:**
```
Input → Classifier → Route to Specialist → Output
           ├→ Specialist A (Technical)
           ├→ Specialist B (General)
           └→ Specialist C (Urgent)
```

**Implementation:**
```python
def route_query(user_query: str) -> str:
    # Classify the query type
    classification = llm_call(
        f"""Classify this query as: technical_support, billing, general_question
        
        Query: {user_query}
        
        Return only the category.""",
        model="fast-model"
    )
    
    # Route to appropriate specialist
    if classification == "technical_support":
        return technical_agent(user_query)
    elif classification == "billing":
        return billing_agent(user_query)
    else:
        return general_agent(user_query)

def technical_agent(query: str) -> str:
    return llm_call(
        f"You are a senior engineer. {query}",
        tools=["check_logs", "run_diagnostic"],
        model="quality-model"
    )
```

**Best practices:**
- Use smaller model for classification
- Have clear, mutually exclusive categories
- Fallback to general handler for edge cases
- Consider cost/performance per route

**Examples:**
- Customer support triage
- Easy questions → cheap model, hard → expensive model
- Different languages → language-specific models

---

### 3. Parallelization

**When to use:** Independent subtasks can run simultaneously.

**Patterns:**

#### A. Sectioning (Divide and Conquer)
```python
async def parallel_analysis(document: str) -> dict:
    """Analyze different aspects in parallel"""
    
    tasks = [
        llm_call_async("Check for security issues", document),
        llm_call_async("Review code style", document),
        llm_call_async("Assess performance", document),
        llm_call_async("Check accessibility", document)
    ]
    
    results = await asyncio.gather(*tasks)
    
    return {
        "security": results[0],
        "style": results[1],
        "performance": results[2],
        "accessibility": results[3]
    }
```

#### B. Voting (Multiple Attempts)
```python
def voting_consensus(question: str, n: int = 5) -> str:
    """Generate multiple answers and vote"""
    
    answers = []
    for i in range(n):
        answer = llm_call(f"{question}\n\nProvide your answer:", temperature=0.7)
        answers.append(extract_answer(answer))
    
    # Majority voting
    from collections import Counter
    vote_counts = Counter(answers)
    consensus = vote_counts.most_common(1)[0][0]
    
    return consensus
```

**Best practices:**
- Ensure tasks are truly independent
- Use async/parallel execution
- Aggregate results programmatically
- Consider cost vs. speed tradeoff

**Examples:**
- Guardrails: content moderation + response generation
- Multi-aspect evaluation (security + style + performance)
- Consensus building for critical decisions

---

### 4. Orchestrator-Workers

**When to use:** Complex tasks where subtasks aren't predictable upfront.

**Architecture:**
```
User Task → Orchestrator → Worker 1 (File A)
                        → Worker 2 (File B)  → Orchestrator → Synthesis → Output
                        → Worker 3 (Tests)
```

**Implementation:**
```python
def orchestrator_workflow(task: str) -> str:
    # Orchestrator plans the work
    plan = llm_call(
        f"""Break down this coding task into specific file changes needed:
        
        Task: {task}
        
        List each file that needs modification and what changes are needed.""",
        model="smart-model"
    )
    
    # Parse plan into subtasks
    subtasks = parse_plan(plan)
    
    # Delegate to workers
    results = []
    for subtask in subtasks:
        worker_result = llm_call(
            f"""You are a specialist in {subtask['file_type']}.
            
            Make this change: {subtask['description']}
            File: {subtask['file']}
            
            Provide the updated code.""",
            model="quality-model"
        )
        results.append(worker_result)
    
    # Orchestrator synthesizes
    final = llm_call(
        f"""Review these changes and create a summary:
        
        {results}
        
        Ensure consistency and completeness.""",
        model="smart-model"
    )
    
    return final
```

**Best practices:**
- Orchestrator handles planning and synthesis
- Workers are specialists with focused prompts
- Dynamic task decomposition
- Can use different models for different workers

**Examples:**
- Complex code changes across multiple files
- Research with unpredictable information needs
- Content creation with multiple components

---

### 5. Evaluator-Optimizer

**When to use:** Quality improves through iterative feedback.

**Architecture:**
```
Input → Generator → Evaluator → [Good? → Output]
           ↑                         |
           └─────── Feedback ────────┘
```

**Implementation:**
```python
def evaluator_optimizer(task: str, max_iterations: int = 3) -> str:
    content = None
    
    for iteration in range(max_iterations):
        # Generate or revise
        if content is None:
            content = llm_call(f"Create: {task}", model="generator")
        else:
            content = llm_call(
                f"Improve based on feedback:\n\nContent:{content}\n\nFeedback: {feedback}",
                model="generator"
            )
        
        # Evaluate
        evaluation = llm_call(
            f"""Evaluate this content and provide specific feedback:
            
            {content}
            
            Rate quality (1-10) and suggest improvements.""",
            model="evaluator"
        )
        
        score = extract_score(evaluation)
        if score >= 8:
            return content  # Good enough
        
        feedback = extract_feedback(evaluation)
    
    return content  # Return best attempt
```

**Best practices:**
- Clear evaluation criteria
- Specific, actionable feedback
- Limit iterations to avoid diminishing returns
- Can use same or different models

**Examples:**
- Translation with quality review
- Creative writing with editorial feedback
- Complex research requiring multiple search iterations

---

### 6. ReAct Agent Loop

**When to use:** Open-ended problems requiring adaptive tool use.

**Architecture:**
```
Task → [Think → Act → Observe] → [Think → Act → Observe] → ... → Answer
```

**Implementation:**
```python
def react_agent(task: str, max_iterations: int = 10) -> str:
    conversation_history = []
    
    for iteration in range(max_iterations):
        # Think + Act
        response = llm_call(
            f"""Task: {task}
            
            History: {conversation_history}
            
            Think about what to do next, then either:
            - Use a tool: Action: ToolName[parameters]
            - Provide final answer: Answer: [your answer]
            
            Format:
            Thought: [your reasoning]
            Action: [tool call] OR Answer: [final answer]
            """,
            tools=available_tools
        )
        
        # Parse response
        thought = extract_thought(response)
        action = extract_action(response)
        
        conversation_history.append(f"Thought: {thought}")
        
        # Check if done
        if action.startswith("Answer:"):
            return action.replace("Answer:", "").strip()
        
        # Execute action
        observation = execute_tool(action)
        conversation_history.append(f"Action: {action}")
        conversation_history.append(f"Observation: {observation}")
    
    return "Max iterations reached without answer"
```

**Best practices:**
- Clear tool documentation
- Explicit thought/action/observation format
- Timeout after max iterations
- Log full trace for debugging
- Use tool results to inform next steps

---

### 7. CodeAct Pattern

**When to use:** Python code is the best way to solve the problem.

**Architecture:**
```
Task → Generate Code → Execute → Observe Results → [Success? → Done | Iterate]
```

**Implementation:**
```python
def codeact_agent(task: str, max_iterations: int = 5) -> dict:
    for iteration in range(max_iterations):
        # Generate code
        code = llm_call(
            f"""Task: {task}
            
            Write Python code to solve this. The code will be executed.
            You can import libraries and use previous results if iterating.
            
            Python code:
            """,
            model="code-model"
        )
        
        # Execute safely
        result = execute_python_safely(code)
        
        if result["error"]:
            # Iterate with error feedback
            task = f"{task}\n\nPrevious attempt failed:\n{result['error']}\n\nFix and try again."
        else:
            return {
                "success": True,
                "output": result["output"],
                "code": code
            }
    
    return {"success": False, "error": "Max iterations reached"}

def execute_python_safely(code: str, timeout: int = 30) -> dict:
    """Execute in sandboxed environment"""
    try:
        # Use Docker or similar for isolation
        output = subprocess.run(
            ["docker", "run", "--rm", "python:3.9", "python", "-c", code],
            capture_output=True,
            timeout=timeout,
            text=True
        )
        return {"output": output.stdout, "error": output.stderr if output.returncode != 0 else None}
    except Exception as e:
        return {"output": None, "error": str(e)}
```

**Best practices:**
- Sandbox code execution (Docker, VMs)
- Set timeouts and resource limits
- Feed errors back for iteration
- Good for data analysis, calculations, automation

---

### 8. Multi-Agent Collaboration

**When to use:** Complex tasks benefit from specialized roles.

**Architecture:**
```
Task → Coordinator → [Researcher + Coder + Reviewer] → Coordinator → Output
```

**Implementation:**
```python
class MultiAgentSystem:
    def __init__(self):
        self.researcher = Agent("researcher", "Find relevant information")
        self.coder = Agent("coder", "Write code solutions")
        self.reviewer = Agent("reviewer", "Review and critique")
    
    def solve(self, task: str) -> str:
        # Phase 1: Research
        research = self.researcher.run(
            f"Research this task: {task}"
        )
        
        # Phase 2: Implementation
        code = self.coder.run(
            f"Based on research, implement: {task}\n\nResearch: {research}"
        )
        
        # Phase 3: Review
        review = self.reviewer.run(
            f"Review this code:\n{code}\n\nDoes it solve: {task}?"
        )
        
        # Phase 4: Refinement (if needed)
        if "issues found" in review.lower():
            code = self.coder.run(
                f"Fix based on review:\n{review}\n\nOriginal code:\n{code}"
            )
        
        return code

class Agent:
    def __init__(self, role: str, instruction: str):
        self.role = role
        self.instruction = instruction
    
    def run(self, task: str) -> str:
        return llm_call(
            f"You are a {self.role}. {self.instruction}\n\nTask: {task}",
            model=f"{self.role}-optimized-model"
        )
```

**Best practices:**
- Each agent has clear specialty and role
- Agents communicate through structured handoffs
- Can use different prompts/models per agent
- Coordinator orchestrates the workflow

**Examples:**
- Software development (PM + Engineer + QA)
- Content creation (Writer + Editor + Fact-checker)
- Research (Searcher + Analyzer + Synthesizer)

---

## Implementation Examples

### Example 1: Customer Support Agent

```python
from typing import List, Dict
import json

class CustomerSupportAgent:
    """ReAct-style support agent with routing"""
    
    def __init__(self):
        self.tools = {
            "search_orders": self.search_orders,
            "check_inventory": self.check_inventory,
            "create_ticket": self.create_ticket,
            "process_refund": self.process_refund
        }
    
    def run(self, customer_query: str) -> str:
        # Step 1: Route by category
        category = self.classify_query(customer_query)
        
        if category == "simple_question":
            return self.simple_response(customer_query)
        
        # Step 2: ReAct loop for complex queries
        return self.react_loop(customer_query, max_steps=5)
    
    def classify_query(self, query: str) -> str:
        prompt = f"""Classify this customer query:
        - simple_question: Can be answered directly (hours, policies, etc.)
        - order_issue: About specific orders, refunds, tracking
        - product_question: About products, availability, features
        
        Query: {query}
        
        Category:"""
        
        return llm_call(prompt, model="gpt-4o-mini").strip()
    
    def react_loop(self, query: str, max_steps: int) -> str:
        history = []
        
        for step in range(max_steps):
            # Think and act
            prompt = f"""Customer query: {query}

History:
{chr(10).join(history)}

Available tools:
- search_orders[customer_id]: Find customer orders
- check_inventory[product_id]: Check product availability
- create_ticket[description]: Escalate to support
- process_refund[order_id]: Issue refund

Think about what to do next:
Thought: [your reasoning]
Action: [tool_name[params]] OR Answer: [final response]
"""
            
            response = llm_call(prompt, model="gpt-4o")
            thought = self.extract_thought(response)
            action = self.extract_action(response)
            
            history.append(f"Thought: {thought}")
            
            # Check if we have final answer
            if action.startswith("Answer:"):
                return action.replace("Answer:", "").strip()
            
            # Execute tool
            observation = self.execute_tool(action)
            history.append(f"Action: {action}")
            history.append(f"Observation: {observation}")
        
        return "I need to escalate this to a human agent."
    
    def search_orders(self, customer_id: str) -> str:
        # Mock implementation
        return json.dumps([
            {"order_id": "12345", "status": "shipped", "date": "2024-01-15"}
        ])
    
    def execute_tool(self, action: str) -> str:
        # Parse "tool_name[params]" format
        tool_name = action.split("[")[0]
        params = action.split("[")[1].rstrip("]")
        
        if tool_name in self.tools:
            return self.tools[tool_name](params)
        return "Tool not found"
    
    # ... other tool implementations
```

---

### Example 2: Code Review Agent (Multi-Agent)

```python
class CodeReviewSystem:
    """Multi-agent system for thorough code review"""
    
    def review(self, code: str, language: str) -> Dict:
        # Agent 1: Security review
        security = self.security_agent(code, language)
        
        # Agent 2: Performance review  
        performance = self.performance_agent(code, language)
        
        # Agent 3: Style review
        style = self.style_agent(code, language)
        
        # Synthesizer: Combine findings
        summary = self.synthesize(security, performance, style)
        
        return {
            "security": security,
            "performance": performance,
            "style": style,
            "summary": summary,
            "approved": self.should_approve(security, performance, style)
        }
    
    def security_agent(self, code: str, language: str) -> Dict:
        prompt = f"""You are a security expert reviewing {language} code.

Code:
```{language}
{code}
```

Check for:
- SQL injection vulnerabilities
- XSS vulnerabilities  
- Authentication/authorization issues
- Secrets in code
- Input validation

Return JSON:
{{
  "issues": [
    {{"severity": "high|medium|low", "line": number, "description": "...", "fix": "..."}}
  ]
}}
"""
        
        response = llm_call(prompt, model="gpt-4o")
        return json.loads(response)
    
    def performance_agent(self, code: str, language: str) -> Dict:
        prompt = f"""You are a performance expert reviewing {language} code.

Code:
```{language}
{code}
```

Check for:
- Inefficient algorithms (O(n²) that could be O(n))
- Unnecessary loops or operations
- Memory leaks
- Database query optimization

Return JSON:
{{
  "issues": [
    {{"severity": "high|medium|low", "line": number, "description": "...", "improvement": "..."}}
  ]
}}
"""
        
        response = llm_call(prompt, model="gpt-4o")
        return json.loads(response)
    
    def synthesize