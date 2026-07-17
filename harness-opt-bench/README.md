# Harness optimization benchmarks

`harness-opt-bench` contains end-to-end benchmarks for improving programs with
VeRO. Each leaf directory pairs one editable target program with one immutable
Harbor dataset and compiles them into an outer Harbor optimization task.

The benchmark definitions intentionally keep three boundaries visible:

- `target/` is the program the optimization agent may edit.
- `partitions/` pins the cases and the development/validation/test split.
- `build.yaml` is trusted configuration: model, evaluator, access policy,
  budgets, and final scoring.

The paper-era benchmark stack remains available on the `paper/v1` branch and
the `paper-v1` tag. New benchmarks should use this Harbor-native layout.

## Benchmarks

| Benchmark | Editable target | Dataset | Split |
| --- | --- | --- | --- |
| [GAIA baseline](gaia/baseline/) | Tool-using Responses API agent | Harbor `gaia/gaia` | 20% / 40% / 40% |

