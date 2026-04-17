# Trace Analysis

This module provides tools for analyzing vero optimization session traces.

## Overview

The trace analyzer loads session data (traces, commits, experiments) and uses an LLM to summarize what the agent did in each optimization phase.

## Quick Start

```python
from pathlib import Path
from vero.traces.analysis import (
    TraceAnalysisPayload,
    TraceAnalyzer,
    plot_session_scores,
)

# Setup
project_path = Path.home() / "your-project"
session_id = "your-session-id"

# Create analyzer
analyzer = TraceAnalyzer(model="gpt-4.1")

# Analyze all phases (returns DataFrame)
df = await analyzer.analyze_session(
    session_id=session_id,
    project_path=project_path,
    max_concurrency=5,
)

# Visualize results
fig = plot_session_scores(
    df,
    show_annotations=True,
    show_best_so_far=True,
)
```

## Components

### TraceAnalysisPayload

Loads and structures session data from disk.

```python
payload = await TraceAnalysisPayload.from_session_id(
    session_id=session_id,
    project_path=project_path,
)

# Summary of the session
payload.summary()

# Access phases
for i, phase in enumerate(payload.phases):
    print(f"Phase {i}: {phase.final_commit.commit[:8]}")
    print(f"  Experiments: {len(phase.experiments)}")
    print(f"  Trace segments: {len(phase.trace_segments)}")
    print(f"  Trace items: {phase.num_trace_items}")

# Get detailed phase info (with optional diffs)
from vero.workspace.git import GitWorkspace

workspace = await GitWorkspace.create(str(project_path))
phase_info = await payload.get_phase_info(phase_index=0, workspace=workspace)
```

### TraceAnalyzer

LLM-based analyzer that summarizes each optimization phase.

```python
analyzer = TraceAnalyzer(
    model="gpt-4.1",  # OpenAI model to use
    max_trace_items=50,  # Limit trace items sent to LLM
)

# Analyze a single phase
phase_info = await payload.get_phase_info(0, workspace=workspace)
result = await analyzer.analyze_phase(phase_info)
print(result["analysis"].short_summary)
print(result["analysis"].tags)

# Analyze all phases (returns DataFrame)
df = await analyzer.analyze_session(
    session_id=session_id,
    project_path=project_path,
    max_concurrency=5,
    show_progress=True,
)
```

### DataFrame Output

The `analyze_session` method returns a pandas DataFrame with columns:

| Column | Description |
|--------|-------------|
| `phase_index` | 0-based phase index |
| `commit` | Final commit hash for the phase |
| `short_summary` | LLM-generated 5-word summary |
| `description` | Detailed description of changes |
| `agent_reasoning` | LLM's interpretation of agent reasoning |
| `tags` | List of change tags (e.g., `prompt_modified`, `tool_added`) |
| `subtag` | Additional tag info if tag is `other` |
| `train_mean_score` | Mean score on train split |
| `train_num_samples` | Number of train samples |
| `validation_mean_score` | Mean score on validation split |
| `validation_num_samples` | Number of validation samples |
| `test_mean_score` | Mean score on test split |
| `test_num_samples` | Number of test samples |

### Visualization

```python
from vero.traces.analysis import plot_session_scores

fig = plot_session_scores(
    df,
    title="Optimization Session Progress",
    show_annotations=True,  # Show short_summary labels
    show_best_so_far=True,  # Show best score line (validation, fallback to train)
    annotation_score_column="train_mean_score",  # Column for annotation y-position
)
```

## Change Tags

The analyzer categorizes changes using these tags:

- `prompt_added`, `prompt_modified`, `prompt_deleted`
- `tool_added`, `tool_modified`, `tool_deleted`
- `workflow_added`, `workflow_modified`, `workflow_deleted`
- `config_modified`
- `bug_fix`
- `refactor`
- `other` (with optional `subtag` for details)

## Customization

You can customize the LLM prompt and output model:

```python
from pydantic import BaseModel, Field

class CustomAnalysis(BaseModel):
    summary: str = Field(description="One-line summary")
    score_prediction: float = Field(description="Predicted score improvement")

custom_prompt = """
Analyze this optimization phase and predict score improvement.
{phase_info}
"""

analyzer = TraceAnalyzer(
    model="gpt-4.1",
    output_model=CustomAnalysis,
    prompt_template=custom_prompt,
)
```

## Listing Sessions

```python
import os
from vero.core.sessions import get_vero_home_dir

sessions_dir = get_vero_home_dir() / "sessions"
sessions = os.listdir(sessions_dir)
print(f"Found {len(sessions)} sessions")
```
