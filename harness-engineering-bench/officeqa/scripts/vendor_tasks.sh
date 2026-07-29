#!/usr/bin/env bash
# Vendor the OfficeQA task dirs into ./tasks/ (gitignored).
#
# OfficeQA is not in the Harbor hub — it lives only as raw task dirs in the
# harbor-datasets git repo, so we fetch them locally and point build.yaml's
# task_source at them. The repo is large (~4GB), so we sparse-checkout only
# datasets/officeqa. Each task.toml (schema 1.0) is then patched with a canonical
# [task].name, which vero's local task staging requires.
#
# Usage:  bash scripts/vendor_tasks.sh
# Result: harness-engineering-bench/officeqa/tasks/officeqa-uid*/  (246 tasks)
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"          # harness-engineering-bench/officeqa
dest="$here/tasks"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "Sparse-cloning datasets/officeqa from harbor-datasets ..."
git clone --depth 1 --filter=blob:none --no-checkout \
  https://github.com/harbor-framework/harbor-datasets.git "$work/hd"
git -C "$work/hd" sparse-checkout init --cone
git -C "$work/hd" sparse-checkout set datasets/officeqa
git -C "$work/hd" checkout

src="$work/hd/datasets/officeqa"
count=$(find "$src" -maxdepth 1 -type d -name 'officeqa-uid*' | wc -l | tr -d ' ')
# Assert the exact count, not merely "some". A partial or interrupted fetch
# produces a directory that validates fine and then silently scores a subset of
# the benchmark, which is worse than failing here. 246 is a property of the
# dataset, recorded in CONFIGURATION.md.
expected=246
[ "$count" -eq "$expected" ] || {
  echo "ERROR: fetched $count officeqa tasks, expected $expected." >&2
  echo "  A partial fetch would score a subset of the benchmark without saying so." >&2
  echo "  If upstream genuinely changed, update \$expected here and the count in" >&2
  echo "  CONFIGURATION.md and scripts/task_data.py together." >&2
  exit 1
}

echo "Copying $count task dirs -> $dest ..."
rm -rf "$dest"; mkdir -p "$dest"
cp -R "$src"/officeqa-uid* "$dest/"

echo "Patching task.toml with canonical [task].name ..."
python3 - "$dest" <<'PY'
import glob, os, re, sys
root = sys.argv[1]
patched = 0
for d in sorted(glob.glob(os.path.join(root, "officeqa-uid*"))):
    uid = os.path.basename(d)
    f = os.path.join(d, "task.toml")
    txt = open(f).read()
    if re.search(r'(?m)^\[task\]', txt):
        continue
    block = f'[task]\nname = "officeqa/{uid}"\n\n'
    lines = txt.splitlines(keepends=True)
    if lines and lines[0].lstrip().startswith("version"):
        new = lines[0] + "\n" + block + "".join(lines[1:])
    else:
        new = block + txt
    open(f, "w").write(new)
    patched += 1
print(f"  patched {patched} task.toml files")
PY

echo "Done: $(find "$dest" -maxdepth 1 -type d -name 'officeqa-uid*' | wc -l | tr -d ' ') tasks in $dest"
