# Harbor integration — tutorial

This walks through compiling a vero optimization run into a Harbor task and running it
with an optimizer agent. Read the [architecture](./architecture.md) first for the
concepts (modes, the trust boundary, the optimization loop).

## Install

```bash
uv pip install 'scale-vero[harbor]'          # adds the `vero harbor` CLI
# the Harbor CLI itself is invoked via uvx; for Modal-backed inner runs use the extra:
uvx --from 'harbor[modal]' harbor --help
```

## 1. Write a `build.yaml`

A build config describes the optimization task: the repo to optimize, how candidates
are scored, the split tiers, the budget, and the reward.

### Mode A — vero runs inference + scoring

```yaml
name: myorg/gsm8k-opt
agent_repo: /path/to/gsm8k-agent     # the repo the optimizer edits
mode: A
task: gsm8k                          # vero task name
task_module: gsm8k_agent.vero_tasks  # module that registers it
dataset: /path/to/gsm8k-dataset      # a saved DatasetDict (inputs + labels)

splits:
  - { split: validation, access: non_viewable }   # optimizer sees aggregate score only
  - { split: test,       access: no_access }       # hidden; scored at the end
budgets:
  - { split: validation, total_run_budget: 5 }
reward_mode: auto_best                # best validation commit auto-selected
selection_split: validation
targets:
  - { split: test, reward_key: reward }
read_only_paths:
  - src/gsm8k_agent/vero_tasks        # the scorer — optimizer may not edit it
secrets: [OPENAI_API_KEY, OPENAI_BASE_URL]   # injected into the eval sidecar only
```

### Mode B — a nested `harbor run` scores (e.g. on Modal)

```yaml
name: myorg/gaia-opt
agent_repo: /path/to/gaia-agent
mode: B
harbor:
  agent_import_path: "gaia_agent:GaiaAgent"   # the agent inside agent_repo
  task_source: gaia/gaia                       # Harbor registry benchmark (or a local dir)
  environment: modal
  model: openai/gpt-4o-mini                    # the inner agent's model
partition:                                     # {split: [harbor task names]} — one task = one sample
  train:      [gaia/<id1>, gaia/<id2>, ...]
  validation: [gaia/<id6>, gaia/<id7>, ...]
splits:
  - { split: train,      access: non_viewable }
  - { split: validation, access: no_access }
budgets:
  - { split: train, total_run_budget: 3 }
reward_mode: auto_best
selection_split: train
targets:
  - { split: validation, reward_key: accuracy }
secrets: [OPENAI_API_KEY, OPENAI_BASE_URL, MODAL_TOKEN_ID, MODAL_TOKEN_SECRET]
```

`secrets` are variable **names**: their values are read from your shell at run time and
injected into the eval sidecar only — never into the optimizer's container. The full
field list is in `vero/harbor/build/config.py` (`BuildConfig`).

## 2. Build the task

```bash
vero harbor build -c build.yaml -o /tmp/opt-task
```

This emits a Harbor task directory: `environment/` (a Docker Compose env = the optimizer
workbench `main` + the `eval-sidecar`, plus volumes), `instruction.md` (the protocol the
optimizer reads), and `tests/test.sh` (the verifier). The dataset/scorer/baseline repo
and the sidecar's `ServeConfig` are baked in.

## 3. Run it with an optimizer

Any Harbor agent can be the optimizer. Provide its creds in your shell (Harbor forwards
them into `main`); e.g. for `claude-code` set `ANTHROPIC_API_KEY` (+ `ANTHROPIC_BASE_URL`
if routing through a gateway).

```bash
# build + run in one step:
vero harbor run -c build.yaml -a claude-code -m claude-haiku-4-5 -e docker

# or run a pre-built task dir:
uvx harbor run -p /tmp/opt-task -a claude-code -m claude-haiku-4-5 -e docker

# the `oracle` agent runs solution/solve.sh (a scripted optimizer) — handy for a smoke test:
uvx harbor run -p /tmp/opt-task -a oracle -e docker
```

The reward lands in the job's `verifier/reward.json` (e.g. `{"reward": 0.42}`), and Harbor
reports it as the trial reward.

## What the optimizer does (the agent-side protocol)

Inside `main`, the optimizer follows `instruction.md`. The `vero harbor` CLI talks to the
eval sidecar over `VERO_EVAL_URL` (set automatically):

```bash
vero harbor status                                   # remaining budget, evaluable splits
# edit the repo, commit, then measure the current HEAD:
vero harbor eval --dataset-id <id> --split validation
vero harbor submit                                   # (if reward_mode: submit) nominate the final commit
```

- `eval` returns an aggregate score + remaining budget; for `no_access` splits it is
  rejected, and labels are never returned.
- With `reward_mode: auto_best`, the best commit on `selection_split` is chosen
  automatically; with `submit`, the agent nominates one.
- The verifier scores the chosen commit on the hidden `targets` split at the end.

## Inspecting a run

```bash
uvx harbor view <jobs-dir>          # browse trials
cat <jobs-dir>/*/*/verifier/reward.json
```

## Examples

- [`examples/gsm8k-agent`](../../examples/gsm8k-agent) (Mode A agent, vero scores gsm8k). It ships the agent + vero task but not a `build.yaml` yet; use the Mode A `build.yaml` snippet above to drive it.
- [`examples/gaia-optimization`](../../examples/gaia-optimization) — Mode B (terminus on
  GAIA via nested Harbor on Modal), with an editable-prompt optimization surface.
