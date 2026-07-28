# VeRO setup guide (for coding agents)

This guide helps a coding agent (Claude, Cursor, Copilot, …) set up and run a
VeRO optimization for a user. VeRO optimizes a **versioned program** against an
**evaluation**: a producer (usually a coding agent) proposes candidate edits, a
trusted evaluator scores them, and the best candidate is selected.

You do two things: (1) author an **evaluation harness**, and (2) run the
**optimizer**. There is no `.veroaccess`, `Policy`, or resource/namespace setup —
those were removed. Containment now comes from the sandbox the producer runs in,
and disclosure is controlled per evaluation set.

## Before you start

Understand the user's program (what it does, how it's invoked, what "better"
means) and read a matching example:

- `examples/c-matmul/` — a language-neutral **command harness** + `vero.toml`.
- `examples/circle-packing/` — a Python program scored by a command harness.
- `../vero-tasks/examples/matmul-{kernel,eval}/` — a **Python task** using the
  `scale-vero-tasks` protocol.
- `examples/harbor-circle-packing/` — a **Harbor** outer loop (contained agent)
  with a simple command inner loop.

## 1. Author the evaluation

Pick whichever fits the user's program.

### Option A — command harness (any language)

Write a script that reads a request JSON and writes a report JSON. VeRO's
`CommandBackend` invokes it with placeholders it substitutes at run time:

```bash
python evaluate.py --workspace {workspace} --request {request} \
                   --report {report} --artifacts {artifacts}
```

- `{workspace}` — the candidate checkout to score.
- `{request}` — JSON: `{"schema_version": 1, "request": {"seed": <int|null>, ...}}`.
- `{report}` — write JSON: `{"schema_version": 1, "status": "success",
  "metrics": {"<name>": <float>, ...}}` (a **dict of metrics** — one becomes the
  objective; others can be constraints or diagnostics).

See `examples/circle-packing/harness/` for a complete scorer.

### Option B — Python task (`scale-vero-tasks`)

For Python benchmarks, use the task protocol (`../vero-tasks`):

```python
from vero_tasks import TaskContext, TaskOutput, TaskResult, create_task

task = create_task("main", required_env_vars=["OPENAI_API_KEY"])

@task.inference()
async def infer(case: dict, context: TaskContext) -> TaskOutput:
    from my_agent import run_agent
    return TaskOutput(output=await run_agent(case["question"]))

@task.evaluation()
async def evaluate(case: dict, output: TaskOutput, context: TaskContext) -> TaskResult:
    return TaskResult.from_task_output(
        output, score=float(output.output == case["answer"])
    )
```

The `scale-vero-tasks` runner adapts this to the same command-harness contract.
See `../vero-tasks/README.md` and `../vero-tasks/examples/matmul-eval/`.

**Rules:** functions are `async`; let exceptions propagate (VeRO records them as
errored cases); keep heavy imports inside functions; declare API keys via
`required_env_vars`; the score is a float (define whether higher or lower is
better via the objective's `direction`).

## 2. Run the optimizer

The target must be a git repo whose working tree is clean.

### Config-driven

```bash
vero init ./run                 # scaffolds run/vero.toml (+ target/, harness/)
# edit run/vero.toml
vero check    --config run/vero.toml    # validate paths, git, eval refs
vero evaluate --config run/vero.toml    # score the baseline only
vero run      --config run/vero.toml    # optimize
```

`vero.toml` sections: `[target]` (root, ref), `[backend]` (kind = `command`,
`harness_root`, `command`), one or more `[[evaluations]]` (name, partition,
`agent_can_evaluate`, `disclosure`, optional `agent_budget`), `[protocol]`
(`selection_evaluation`, `final_evaluation`, `max_proposals`), `[objective]`
(metric, direction), and `[optimizer]` (`kind = "vero"`, optional `model`,
`instruction`).

### One-shot (flags)

```bash
vero optimize ./target \
  --harness-root ./harness \
  --evaluate "python3 {harness}/evaluate.py --workspace {workspace} --request {request} --report {report} --artifacts {artifacts}" \
  --agent vero \
  --instruction "Improve the program without changing its intended behavior." \
  --metric score --direction maximize \
  --evaluation-set default --max-proposals 4 --max-turns 60
```

## 3. The optimizer agent (`--agent vero`)

`VeroAgent` runs on the OpenAI Agents SDK. Its harness runs on the host; its
shell/file effects run inside a sandbox (bound to the candidate checkout) via
function tools (`shell`, `read_file`, `write_file`), plus an `evaluate` tool for
mid-run self-scoring. There are no custom bash/grep/git tools to configure and no
in-process ACLs — the sandbox is the boundary.

- **Model:** any provider via LiteLLM. Set `VERO_OPTIMIZER_MODEL`
  (e.g. `openai/gpt-5.4`, `anthropic/claude-sonnet-4-5-...`) or `[optimizer] model`.
- **Containment:** the sandbox client is the seam — local for fast/trusted runs;
  for a contained/untrusted agent, run it through **Harbor** (see
  `examples/harbor-circle-packing/`).
- **Strategy:** the default proposes one candidate per round; for
  population/evolutionary search use `EvolutionaryStrategy`
  (`vero.optimization.EvolutionaryStrategy`).

## Don't

- Don't let the harness/task code be edited during optimization — it scores the
  candidate and must stay fixed (keep it in `--harness-root`, outside the target).
- Don't hardcode absolute paths in harness/task code; use the `{...}` placeholders
  and request parameters.
- Don't suppress exceptions in inference/evaluation — VeRO records them.
- Don't put secrets in code; declare them as env vars.
