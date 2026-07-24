# Optimize the circle packing

Improve `packing.py` in `/work/agent` so the program packs **26 non-overlapping
circles inside the unit square** with the **largest possible sum of radii**.

`packing.py` must keep a callable `run_packing()` that returns
`(centers, radii, reported_sum)`, where `centers` is 26 `[x, y]` pairs, `radii`
is 26 non-negative floats, and `reported_sum` equals `sum(radii)`. Circles must
stay fully inside the unit square and must not overlap. The current baseline
scores about 0.96; the best known result is ~2.635.

## Workflow

1. Edit `/work/agent/packing.py`.
2. Commit your change with Git.
3. Score the current commit on the validation set:

   ```bash
   evals run --backend cmd --evaluation-set circle-packing \
     --partition validation --start 0 --stop 1
   ```

4. Use `evals status` to see remaining evaluation budget.

The trusted sidecar owns scoring, budget, and final candidate selection. Only
`sum_radii` (with a `valid == 1` constraint) counts. Iterate: try an
arrangement, measure it, keep what improves the validated sum.
