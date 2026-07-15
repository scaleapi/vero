# Matrix multiplication program optimization

This is the smallest end-to-end demonstration of VeRO's v0.5 framing: the
target is an ordinary Python matrix multiplication function, not an agent.

- `matmul-kernel` is the editable target and has no dependency on VeRO.
- `matmul-eval` is a trusted external evaluation harness built with
  `scale-vero-tasks`.
- `run.py` connects the target, evaluator, objective, session, and optional
  coding agent.

From the `vero/` package directory:

```bash
# Exercise the complete evaluation pipeline without credentials.
uv run python examples/matmul-kernel/run.py --eval-only

# Let a coding agent optimize the function.
uv run python examples/matmul-kernel/run.py --agent vero
uv run python examples/matmul-kernel/run.py --agent claude
```

`--max-candidates` limits completed optimization attempts. Agent-requested
checkpoints are evaluations too, so use the independent `--max-evaluations`
option when you also want a hard evaluation budget. That budget includes the
baseline, checkpoints, and completed candidates.

Every candidate is edited and evaluated in an isolated Git worktree. The
original target template is unchanged, while reports, agent state, events, and
the best candidate identity are preserved in the printed session directory.
