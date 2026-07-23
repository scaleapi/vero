# Harness engineering benchmarks

`harness-engineering-bench` contains end-to-end benchmarks for automatically
improving the harness of an agent or the code of a component used to build
agents. Each leaf directory pairs one editable target program with one immutable
Harbor dataset and compiles them into an outer Harbor optimization task.

The benchmark definitions intentionally keep three boundaries visible:

- `target/` is the program the optimization agent may edit.
- `partitions/` pins the cases and the development/validation/test split.
- `build.yaml` is trusted configuration: model, evaluator, access policy,
  budgets, and final scoring.

In each benchmark, the complete development tasks and attachments are mounted
read-only for the optimization agent. Development evaluations expose per-case
results and complete Harbor trial records, including exact failures and
target-agent logs. Validation remains aggregate-only, and
test is reachable only by the trusted final verifier.

The paper-era benchmark stack remains available on the `paper/v1` branch and
the `paper-v1` tag. New benchmarks should use this Harbor-native layout.

## Benchmarks

Promoted benchmarks live at the top level. Task sets still under review live in
`candidates/`; we work through the list in the paper's `benchmark-scoping.md` and
promote a task set to the top level once it is ready.

### Promoted

| Benchmark | Editable target | Dataset | Split |
| --- | --- | --- | --- |
| [GAIA baseline](gaia/baseline/) | Tool-using Responses API agent | Harbor `gaia/gaia` | 20% / 40% / 40% |

### Candidates

| Benchmark | Editable target | Dataset | Split |
| --- | --- | --- | --- |
| [SWE-Atlas-QnA baseline](candidates/swe-atlas-qna/baseline/) | Codebase investigation agent | Harbor `scale-ai/swe-atlas-qna` | 20% / 40% / 40% |
| [tau3 baseline](candidates/tau3/baseline/) | MCP customer-service agent | Harbor `sierra-research/tau3-bench` | 20% / 40% / 40% |
| [OfficeQA baseline](candidates/officeqa/baseline/) | Grounded document-QA agent | Treasury Bulletin corpus | 20% / 40% / 40% |
| [ALE-Bench (ahc011)](candidates/ale-bench/) | C++ solver program (component, not an agent) | ALE-Bench AHC problem | public / private seeds |
