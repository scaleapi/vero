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

165 of the 450 tasks, as 33 / 66 / 66.

The dataset is 72 easy and 378 hard. A proportional draw would seed near the
floor: o4-mini scores 76.4% on easy and 14.55% on hard, and the next best model
manages 13.76% on hard. This split therefore takes **every easy task plus a
deterministic sample of 93 hard ones**, a 44/56 mix that propagates to every
partition (14/19, 29/37, 29/37). The mix is recorded as `difficulty_quotas` in
`partitions/manifest.json` so the knob is visible rather than implicit, and it is
the first thing to revisit if the seed lands badly.

165 also holds the finalize wall at 9 waves, `ceil(66 x 3 / 24)`, matching
browsecomp-plus. A proportional 20/40/40 over all 450 would cost 23.

Regenerate or verify:

```bash
cd harness-engineering-bench/dabstep
uvx --from 'harbor[modal]==0.20.0' python scripts/partition_dabstep.py \
  --tasks-dir <local export of the dataset> --check
```

The export is only needed for the `difficulty` tag; canonical names and refs come
from the registry. `scripts/partition_dabstep.py` explains why this is not an
entry in `../scripts/partition_dataset.py`.

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
