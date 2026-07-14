# Generic C program optimization

This example is the Phase A product proof: the editable target is a C repository with
no Python package, VeRO import, dataset, task module, or agent framework. A trusted
harness outside that repository compiles it, checks numerical correctness, and reports
latency. The objective minimizes latency subject to correctness.

Initialize the example target as its own versioned program, then evaluate or optimize
it through the same `vero.toml` path used by other generic programs:

```bash
cd examples/c-matmul/target
git init -b main
git add .
git -c user.name=vero -c user.email=vero@localhost commit -m baseline
cd ..

vero evaluate --config vero.toml
vero run --config vero.toml
```

The checked-in optimizer is deterministic so CI can prove that VeRO creates, evaluates,
and selects an improved candidate without an external model. In normal use, replace
`[optimizer].command` with a coding-agent command. VeRO passes it the editable target
through `{workspace}`; the evaluation harness and active configuration remain outside
the target and are not included in candidate commits.
