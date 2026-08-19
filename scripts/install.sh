#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
"$PYTHON" -m pip install \
  "$HERE/engines/isgn" \
  "$HERE/engines/m2n" \
  "$HERE/engines/fec" \
  "$HERE"
dor stack-info
