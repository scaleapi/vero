# Circle packing program optimization

This is a non-trivial VeRO benchmark adapted from
[ShinkaEvolve's circle-packing example](https://github.com/SakanaAI/ShinkaEvolve/tree/main/examples/circle_packing).
The editable program places 26 circles in a unit square and maximizes the sum
of their radii. A trusted external harness checks the exact geometry and emits
the score, detailed validation measurements, a JSON layout, and an SVG rendering.

The baseline is deliberately simple. The search space includes better initial
layouts, numerical optimization of radii and centers, stochastic global search,
local refinement, restarts, and hybrid algorithms. Unlike ShinkaEvolve's marked
code block, VeRO gives the coding agent an ordinary versioned repository and
allows it to change the complete implementation.

## Run it

The candidate must first be initialized as its own Git repository:

```bash
cd examples/circle-packing/target
git init -b main
git add .
git -c user.name=vero -c user.email=vero@localhost commit -m baseline
cd ..

vero check --config vero.toml
vero evaluate --config vero.toml
vero run --config vero.toml
```

`vero evaluate` is credential-free and records the baseline score. `vero run`
uses the built-in VeRO coding agent with the configured LiteLLM model identifier
and permits up to 30 agent-requested development evaluations in one proposal.
Change `optimizer.model` to any model available through your provider. The best
nominated candidate is then re-evaluated through the hidden final evaluation.
Candidate versions remain in the session candidate repository, while evaluation
artifacts are available in `.vero/session/evaluations/` and to the agent through
its read-only `.evals` context.

Evaluation launches through the candidate's locked uv environment. The seed
program has no third-party dependencies; an optimizer can add reproducible
scientific-computing dependencies with `uv add`. The evaluator applies the
configured protocol seed to Python and NumPy RNGs and fixes `PYTHONHASHSEED` so
stochastic candidates are reproducible when they use those standard sources.
The same value is available to candidates as `VERO_EVALUATION_SEED`.

## Attribution

The starting algorithm is adapted from ShinkaEvolve, Copyright 2025 Sakana AI,
under the Apache License 2.0. The original used NumPy and restricted evolution
to a marked block; this version uses the standard library and is structured as
a complete VeRO target repository. See [UPSTREAM_LICENSE](UPSTREAM_LICENSE) for
the upstream license.
