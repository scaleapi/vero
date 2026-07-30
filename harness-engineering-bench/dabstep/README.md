# DABstep

Multi-step data analysis over a fixed set of Adyen payment files. The agent
explores `/app/data/`, reads the shipped documentation, and writes one exact
answer to `/app/answer.txt`.

Proposed as the tau3 replacement in
[`../tau3-replacement-analysis.md`](../tau3-replacement-analysis.md) and picked at
the 2026-07-29 sync.

## Pinned source

`adyen/dabstep@sha256:0edf62c0bdf7003b1d1f934f1547df1c051877e076d5b6f6a2d99caf8b6432b3`

Harbor-native, so nothing is vendored locally: unlike officeqa and
browsecomp-plus there is no `tasks/` tree to fetch, and the benchmark compiles on
a fresh checkout. The image `curl`s its seven data files at build time and needs
no network at run time.

## Split

165 of the 450 tasks as 33 / 66 / 66, in **two difficulty mixes**. 165 matches
browsecomp-plus and holds the finalize wall at 9 waves, `ceil(66 x 3 / 24)`; a
20/40/40 over all 450 would cost 23.

| | mix | dev | val | test | config |
|---|---|---|---|---|---|
| `partitions/` | 16/84, the dataset's own ratio | 5 easy / 28 hard | 10 / 56 | 11 / 55 | `baseline/build.yaml` |
| `partitions/reweighted/` | 44/56 | 14 / 19 | 29 / 37 | 29 / 37 | `baseline/build.reweighted.yaml` |

The dataset is 72 easy and 378 hard, and the two are far apart: o4-mini scores
76.4% on easy against 14.55% on hard, with the next best model at 13.76% on hard.
So the proportional mix puts about **5 easy tasks in a 33-case development
partition** and should seed near 0.12, which may be too thin a signal for the
optimizer to climb. The reweighted mix takes every easy task plus 93 hard and
roughly doubles the seed. 44% is the ceiling at this size because only 72 easy
tasks exist; a clean 50/50 would mean 144 cases and drop held-out from 66 to 58.

**Proportional is canonical** because every other subsample in the suite preserves
its population: swe-bench-pro's sample is proportional by repository and
browsecomp-plus is a plain seeded shuffle. Reporting from the reweighted mix means
stating in the paper that we chose the difficulty, so it stays a variant until a
probe shows the canonical mix has no headroom. That ordering is Varun's call from
the 2026-07-29 thread: run the strongest optimizer (opus + claude-code) against
the proportional mix first and reweight only if it finds nothing.

Both mixes record their quota as `mix` and `difficulty_quotas` in their manifest,
and both are deterministic. Regenerate or verify:

```bash
cd harness-engineering-bench/dabstep
uvx --from 'harbor[modal]==0.20.0' python scripts/partition_dabstep.py \
  --tasks-dir <local export of the dataset> --check
uvx --from 'harbor[modal]==0.20.0' python scripts/partition_dabstep.py \
  --tasks-dir <local export of the dataset> --mix reweighted --check
```

The export is only needed for the `difficulty` tag; canonical names and refs come
from the registry. `scripts/partition_dabstep.py` explains why this is not an
entry in `../scripts/partition_dataset.py`.

The two test partitions overlap in only 9 of 66 cases, so a baseline pinned on one
mix says nothing about the other. Each needs its own K=3 pin.

## Scoring

The upstream DABstep `scorer.py`: `math.isclose` at rel_tol 1e-4, order-insensitive
list compare, and `SequenceMatcher > 0.95` for strings. Binary per case, no judge
model, so this benchmark keeps `harness_user: harness` isolation.

Answer formatting decides a real share of cases, which is genuine harness surface:
the seed prompt covers units, separators and the "Not Applicable" sentinel, and an
optimizer can do better.

## Open before this is reported

- **`baseline_reward` is unset and `score_baseline: true`.** Every run currently
  pays an extra full held-out pass and the reward is not reproducible. Run
  `../scripts/rescore_candidate.py --seed` three times, put the mean in
  `baseline/build.yaml`, and set `score_baseline: false`.
- **First cold build.** `build_timeout_sec` is 600 while the build downloads from
  an external host, and `error_rate_threshold: 0.1` means a rate-limited host
  aborts the whole evaluation rather than one case. Watch the first build with 24
  sandboxes pulling the same endpoint.
- The `allow_internet` key is absent from all 450 tasks, so "no network at run
  time" rests on a Harbor default rather than anything the dataset declares.
