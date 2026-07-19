#!/usr/bin/env bash
# chunk.sh — read a large trace in byte-windowed chunks so it never has to be
# loaded into context whole. Reproduces IG's get_trace_chunk over the filesystem.
#
#   chunk.sh <file>            -> print chunk count + total size (no content)
#   chunk.sh <file> <k>        -> print chunk k (0-indexed)
#   chunk.sh <file> <k> <size> -> use a custom chunk size (bytes; default 20000)
#
# Chunks are cut on byte boundaries, so a JSON token may straddle two chunks —
# that's fine for scanning/grepping; use `jq` on the whole file when you need
# structure. Prefer grepping the file directly first; only chunk to *read* a
# specific region of a trace too big to cat.
set -euo pipefail

file=${1:?usage: chunk.sh <file> [chunk_index] [chunk_size_bytes]}
size=${3:-20000}

if [[ ! -f "$file" ]]; then
  echo "chunk.sh: no such file: $file" >&2
  exit 1
fi

bytes=$(wc -c <"$file" | tr -d ' ')
# ceil division
count=$(( (bytes + size - 1) / size ))

if [[ $# -lt 2 ]]; then
  echo "file:   $file"
  echo "bytes:  $bytes"
  echo "chunks: $count  (chunk size ${size} bytes; index 0..$((count > 0 ? count - 1 : 0)))"
  exit 0
fi

k=$2
if (( k < 0 || k >= count )); then
  echo "chunk.sh: chunk index $k out of range 0..$((count - 1))" >&2
  exit 1
fi

# dd reads exactly one window; skip in whole blocks of `size`.
dd if="$file" bs="$size" skip="$k" count=1 2>/dev/null
