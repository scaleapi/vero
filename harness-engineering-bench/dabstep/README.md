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

165 of the 450 tasks as 33 / 66 / 66, keeping the dataset's own 16/84 difficulty
ratio (dev 5 easy / 28 hard, val 10 / 56, test 11 / 55). 165 matches gaia and
browsecomp-plus and holds the finalize wall at 9 waves, `ceil(66 x 3 / 24)`; a
20/40/40 over all 450 would cost 23.

Proportional, like every other subsample in the suite: swe-bench-pro's sample is
proportional by repository and browsecomp-plus is a plain seeded shuffle.

A difficulty-reweighted 44/56 variant was built and dropped. The worry it addressed
was that the natural mix would seed near the floor, since the dataset is 84% hard
and published frontier scores on hard are around 14.5%. Measurement disagreed: the
seed reads 0.596 on held-out and 0.636 on development, so there was nothing to fix,
and reweighting would have meant being the only benchmark that chooses its own
difficulty. `git log` has it if the question returns.

Regenerate or verify:

```bash
cd harness-engineering-bench/dabstep
uvx --from 'harbor[modal]==0.20.0' python scripts/partition_dabstep.py \
  --tasks-dir <local export of the dataset> --check
```

The export is only needed for the `difficulty` tag; canonical names and refs come
from the registry. `scripts/partition_dabstep.py` explains why this is not an entry
in `../scripts/partition_dataset.py`, and still carries the reweighted quota behind
`--mix` if it is ever wanted.

## Baseline

**0.596 ±0.026** on the 66-case held-out partition: pooled mean over K=3 rounds
(0.621 / 0.561 / 0.606), 198 trials, zero exceptions, measured 2026-07-30 with
`../scripts/rescore_candidate.py --seed`.

Two things worth carrying forward. The seed's run-to-run spread is about 0.05, so
treat any single-round delta under that as unresolved. And the pin belongs to the
seed as of the turn-cap-40 commit: change `agent.py` at all and it has to be redone.

The seed turn cap is 40, not the 24 inherited from officeqa. At 24 the seed
truncated 18 of 33 development cases and force-answered 17; measured 9.1s per turn
against the declared 1800s clock, so 24 was leaving the optimizer a large
one-integer win. 40 matches swe-atlas-qna and brings truncation to 3 of 33.

## Scoring

The upstream DABstep `scorer.py`: `math.isclose` at rel_tol 1e-4, order-insensitive
list compare, and `SequenceMatcher > 0.95` for strings. Binary per case, no judge
model, so this benchmark keeps `harness_user: harness` isolation.

Answer formatting decides a real share of cases, which is genuine harness surface:
the seed prompt covers units, separators and the "Not Applicable" sentinel, and an
optimizer can do better.

## Open before this is reported

- **Re-pin if the seed changes.** The 0.596 above belongs to `agent.py` as it
  stands. Any edit, including propagating fixes from the other seeds, invalidates it.
- **First cold build.** `build_timeout_sec` is 600 while the build downloads from
  an external host, and `error_rate_threshold: 0.1` means a rate-limited host
  aborts the whole evaluation rather than one case. Watch the first build with 24
  sandboxes pulling the same endpoint.
- The `allow_internet` key is absent from all 450 tasks, so "no network at run
  time" rests on a Harbor default rather than anything the dataset declares.
