# GSM-8k Agent

An example of an agent package that can be optimized using `vero`.

## Structure

```markdown
gsm8k-agent/
├── src/gsm8k_agent/
│   ├── __init__.py
│   ├── agent.py              # The agent implementation
│   └── vero_tasks/           # Vero task definitions
│       ├── __init__.py
│       └── main.py           # Main evaluation task
├── pyproject.toml
└── uv.lock
```

## Evaluation Setup

This example uses the **vero_tasks** pattern. The evaluation logic is defined in `src/gsm8k_agent/vero_tasks/main.py`.

### Running Optimization

```python
from vero.policy import Policy
from vero.agents.vero import VeroAgent

policy = Policy(
    project_path="/path/to/gsm8k-agent",
    dataset="/path/to/gsm8k-dataset",
    agent=VeroAgent(model="anthropic/claude-sonnet-4-5-20250929"),
    task="main",
    train_budget=10,
    max_turns=200,
)

best = await policy.run()
```

## Dataset

This agent is evaluated on the [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k) - a dataset of grade school math word problems.
