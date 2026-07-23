# Generic C program optimization

This example is the concrete proof that VeRO optimizes programs rather than
only Python agents. The editable target is a C repository with no Python
package or VeRO dependency. A trusted harness outside the target compiles it,
checks numerical correctness, and reports latency.

Initialize the target as its own versioned program, then evaluate and optimize
it through the checked-in `vero.toml`:

```bash
cd examples/c-matmul/target
git init -b main
git add .
git -c user.name=vero -c user.email=vero@localhost commit -m baseline
cd ..

vero evaluate --config vero.toml
vero run --config vero.toml
```

The deterministic producer makes the example suitable for CI and requires no
model credentials. Replace `[optimizer]` with `kind = "vero"` or
`kind = "claude"` plus an `instruction` to use a built-in coding agent.
