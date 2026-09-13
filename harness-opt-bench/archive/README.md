# Archived benchmarks

These benchmarks are retained for reproducibility and future development but are
not part of the reported HarnessOpt-Bench suite.

| Benchmark | Why it was archived |
| --- | --- |
| [SWE-Atlas-QnA](swe-atlas-qna/) | Binary rewards did not separate optimizer results reliably |
| [tau3](tau3/) | Simulator and grader variability obscured small harness improvements |
| [SWE-bench-Pro](swe-bench-pro/) | The configuration was not normalized to the shared evaluation protocol |

The configurations remain useful as working examples, but results should not be
compared directly with the reported suite without first reviewing their pinned
baselines, budgets, attempts, and timeouts.

Each benchmark directory explains its task, split, editable target, and current
limitations.
