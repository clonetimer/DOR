#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
"$PYTHON" -m pip install \
  "$HERE/wheels/isgn_toolkit-2.2.0-py3-none-any.whl" \
  "$HERE/wheels/m2n_toolkit-2.3.1-py3-none-any.whl" \
  "$HERE/wheels/fec_toolkit-2.4.0-py3-none-any.whl" \
  "$HERE/wheels/dor_diagnostic_stack-0.6.0-py3-none-any.whl"
dor stack-info
