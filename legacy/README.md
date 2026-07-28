# Legacy (original VeRO paper)

This directory archives the codebase as it existed before the v0.5 redesign —
it is the code for the **original VeRO paper**. It is kept for reference and
reproducibility only; it is not used by the current system, whose code lives at
the repository root (`vero/`, `harness-engineering-bench/`).

**Reproducing the paper?** Prefer the frozen ref, which is the repository exactly
as it stood at publication rather than relocated under this directory:

```bash
git checkout paper-v1          # tag; the paper/v1 branch points at the same commit
```

**Installing anything from here?** Give it its own virtualenv. `legacy/vero` is
`scale-vero` 0.4.7 and the current `vero/` is `scale-vero` 0.5.0; both import as
`vero`, so they cannot share an environment.
