# Pharma Summarizer

An AI agent that generates concise, technically accurate summaries of pharmaceutical and chemical documentation sections.

## Overview

The Pharma Summarizer agent takes sections of pharmaceutical/chemical text and produces summaries that:
- Are approximately 100 words in length (±20% tolerance)
- Preserve all critical chemical names and technical terminology
- Maintain high semantic similarity to the original content
- Demonstrate technical accuracy and completeness

## Agent Architecture

The agent uses the OpenAI Agents framework with GPT-5 as the underlying model. It's implemented as a simple summarization assistant without specialized tools, relying on the model's capabilities for technical text understanding.

### Key Components

- **Agent**: `SummarizationAgent` - Configured to produce accurate summaries from input text sections
- **Model**: GPT-5
- **Tools**: None (relies on model capabilities)

## Evaluation Metrics

The agent is evaluated using a composite scoring system with four distinct metrics:

### 1. Word Length Deviation (0-1 score)
- **Target**: 100 words
- **Tolerance**: ±20% (80-120 words)
- **Scoring**: Full score within tolerance band, linear decay outside

### 2. Chemical Name Preservation (0-1 score)
- **Requirement**: All chemical names from a master list must be preserved
- **Master List**: Includes 90+ chemical compounds (e.g., [1-¹³C]Pyruvic acid, DMSO, TRIS, etc.)
- **Scoring**: Proportion of present chemicals correctly retained in summary

### 3. Cosine Similarity (0 or 1)
- **Method**: OpenAI text-embedding-3-large embeddings
- **Threshold**: ≥0.85 similarity → score 1, otherwise 0
- **Purpose**: Ensures semantic alignment with original content

### 4. LLM as Judge (0 or 1)
- **Judge Model**: GPT-5
- **Threshold**: ≥0.85 quality score → score 1, otherwise 0
- **Criteria**: Technical accuracy, factual preservation, minimal information loss

### Composite Score

**Pass Threshold**: Sum of all 4 metrics = 4.0 (perfect score on all metrics)

This strict threshold ensures summaries are:
- Appropriately concise
- Technically precise
- Semantically faithful
- Factually complete

## Installation

From the `pharma_summarizer` directory:

```bash
# Install dependencies
uv sync

# Install development dependencies (includes testing tools)
uv sync --group dev
```

## Usage

### As a Library

```python
from pharma_summarizer.agent import run_agent

# Run the agent on a text section
result = await run_agent("Your pharmaceutical text section here...")
print(result.final_output)
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run evaluation test with Vero parameters
uv run pytest tests/test_evaluation.py::test_evaluation
```

## Project Structure

```
pharma_summarizer/
├── src/pharma_summarizer/
│   ├── __init__.py
│   └── agent.py           # Main agent implementation
├── tests/
│   ├── __init__.py
│   ├── conftest.py        # Test configuration
│   ├── metrics.py         # Evaluation metrics implementation
│   └── test_evaluation.py # Vero evaluation test suite
├── pyproject.toml
└── README.md
```

## Dependencies

- `openai` (≥2.7.2): OpenAI API client
- `openai-agents` (≥0.5.0): OpenAI Agents framework
- `vero` (dev): Vero evaluation framework

## Technical Notes

### Chemical Name Master List

The evaluator maintains a comprehensive list of chemical compounds commonly found in pharmaceutical documentation, including:
- Isotope-labeled compounds (¹³C, ¹²C variations)
- Common solvents (DMSO, THF, ACN, etc.)
- Buffers and reagents (TRIS, EDTA, etc.)
- Complex chemical structures (trityl radicals, etc.)

### Evaluation Philosophy

The strict 4/4 scoring threshold reflects the critical nature of pharmaceutical documentation where:
- Technical precision is non-negotiable
- Chemical nomenclature must be exact
- Information loss can have serious implications
- Brevity must not compromise accuracy

## Development

To add new metrics or modify evaluation logic, see `tests/metrics.py`. Each metric function should:
- Accept the summary and original section as inputs
- Return a float score between 0 and 1
- Be added to the composite score in `evaluate_summary()`