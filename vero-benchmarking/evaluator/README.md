# VeRO benchmark evaluator

This minimal project is the trusted process boundary for Python benchmark
evaluation. VeRO overlays an editable target program at runtime and invokes the
versioned `scale-vero-tasks` runner from this environment. Keeping the harness
separate prevents coding-agent and analysis dependencies from entering every
evaluation environment.
