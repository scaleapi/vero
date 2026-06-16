# GAIA optimization example (Harbor Mode B)

This example shows the **vero ⇄ Harbor** integration optimizing a coding agent on a
real benchmark. An optimizer (e.g. Claude Code) edits a GAIA agent's prompt; each
candidate is scored by a **nested `harbor run`** of the agent on real
[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) tasks (on Modal). The
reward is accuracy on a hidden split.

This is "Mode B": vero does **no** inference itself — evaluation is delegated to a
nested Harbor run, and the reward comes from Harbor's verifier. (Contrast "Mode A",
e.g. [`../gsm8k-agent`](../gsm8k-agent), where vero runs inference and scoring directly.)

## What's here

```
gaia-optimization/
├── build.yaml                         # the optimization task definition (vero harbor build -c)
├── pyproject.toml                     # deps: harbor[modal]
└── src/gaia_agent/
    ├── agent.py                       # GaiaAgent(Terminus2): the editable agent
    └── prompts/                       # the OPTIMIZATION SURFACE — the optimizer edits these
        ├── terminus-json-plain.txt
        └── terminus-xml-plain.txt
```

`GaiaAgent` subclasses Harbor's `Terminus2` and overrides only its prompt-template
path so the prompt is read from this package's editable `prompts/` directory. The
optimizer improves `prompts/terminus-json-plain.txt`; the terminal loop, tmux
session, and response parsing are reused from `Terminus2` unchanged.

## Prerequisites

- The `harbor` CLI (`uvx --from 'harbor[modal]' harbor ...`) and Docker (outer trial).
- A [Modal](https://modal.com) account for the inner GAIA runs:
  `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in your shell env.
- An OpenAI-compatible LLM endpoint for the **inner** GAIA agent:
  `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL` to point at a gateway). The model is
  set in `build.yaml` (`harbor.model`, default `openai/gpt-4o-mini`).
- Creds for the **outer** optimizer agent, per that agent (e.g. `ANTHROPIC_API_KEY`
  for `-a claude-code`). Harbor forwards these from your shell into the optimizer's
  container; they are **not** shared with the eval sidecar.

Secrets are resolved from your shell at run time and injected into the eval sidecar
**only** (see `build.yaml`'s `secrets:` — those are variable *names*, not values).

## Run it

```bash
# install vero with the harbor extra
uv pip install 'scale-vero[harbor]'

# build the task, then run it with an optimizer of your choice
vero harbor build -c build.yaml -o /tmp/gaia-task
uvx harbor run -p /tmp/gaia-task -a claude-code -m claude-haiku-4-5 -e docker

# ...or build + run in one step:
vero harbor run -c build.yaml -a claude-code -m claude-haiku-4-5 -e docker
```

The optimizer reads the task instruction, edits `src/gaia_agent/prompts/...`, commits,
and calls `vero harbor eval --split train` to measure candidates within its budget.
At the end, the best train commit is scored on the hidden `validation` split and the
accuracy is written to Harbor's `reward.json`.

## Notes

- **GAIA is hard.** A terminal agent solves only some tasks; expect low scores and
  weak optimization signal on a 5-task subset. Increase the subset, pick easier tasks,
  or use a stronger model for a more meaningful run.
- **Cost/time.** Each GAIA task is a full agent rollout on a Modal sandbox (minutes +
  LLM tokens). The default budget keeps a run to a handful of nested evals.
- Pick your own task ids by enumerating the benchmark:
  `python -c "import asyncio; from harbor.models.job.config import DatasetConfig as D; print(asyncio.run(D(name='gaia/gaia').get_task_configs()))"`

## Attribution

`src/gaia_agent/prompts/*.txt` are copied from Harbor's `terminus_2` agent
(© Harbor authors, Apache-2.0) so the prompt stays compatible with the parser
`GaiaAgent` inherits. They are included here as the editable optimization surface.
