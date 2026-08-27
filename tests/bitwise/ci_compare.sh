#!/bin/bash
# Build two refs of the current repo and diff their harness manifests.
#   ci_compare.sh <base-ref> <head-ref>
#
# Both builds get an rpath that puts their own src/.libs ahead of any installed
# libmeep; otherwise both sides load the same installed library and the
# comparison silently passes no matter what changed.
set -euo pipefail

BASE="$1"; HEAD_REF="$2"
OUT="${BITWISE_OUT:-/tmp/bitwise}"
EXTRA_CONFIGURE="${EXTRA_CONFIGURE:-}"
RANKS="${RANKS:-1 2}"
mkdir -p "$OUT"

SRC="$PWD"
# Drive both sides with the head's harness so the config list matches.
HARNESS="$SRC/tests/bitwise"

build_ref() {
  local ref="$1" dir="$OUT/tree-$1"
  dir="$OUT/tree-$(git rev-parse --short "$ref")"
  if [ ! -f "$dir/.built" ]; then
    rm -rf "$dir"
    git worktree add --detach "$dir" "$ref"
    ( cd "$dir"
      sh autogen.sh --enable-maintainer-mode --enable-shared --with-mpi \
         --without-scheme --with-python $EXTRA_CONFIGURE \
         LDFLAGS="-Wl,-rpath,$dir/src/.libs" >/dev/null
      make -j"$(nproc)" >/dev/null )
    touch "$dir/.built"
  fi
  echo "$dir"
}

BASE_DIR="$(build_ref "$BASE")"
HEAD_DIR="$(build_ref "$HEAD_REF")"

run() {
  local dir="$1" out="$2"
  PYTHONPATH="$dir/python:${PYTHONPATH:-}" \
  LD_LIBRARY_PATH="$dir/src/.libs:${LD_LIBRARY_PATH:-}" \
    python3 "$HARNESS/run_matrix.py" --out "$out" --ranks $RANKS --python python3
}

run "$BASE_DIR" "$OUT/base.json"
run "$HEAD_DIR" "$OUT/head.json"
python3 "$HARNESS/compare_manifests.py" "$OUT/base.json" "$OUT/head.json"
