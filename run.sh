#!/usr/bin/env bash
# Launch ManiaScope — the native osu!mania skillset difficulty viewer.
#
# Python entry point; uses a matching prebuilt calculator when available.
# Follows the map and rate selected in osu!lazer
# (via a local tosu helper; lazer's log as the rate-less fallback).
#
# Env overrides:
#   LAZER_DATA   osu!lazer data dir (default ~/.var/app/sh.ppy.osu/data/osu)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Small matrix fits do not benefit from every laptop core starting a BLAS thread.
export OPENBLAS_NUM_THREADS="${MANIASCOPE_BLAS_THREADS:-1}"
export OMP_NUM_THREADS="${MANIASCOPE_BLAS_THREADS:-1}"

LOGDIR="$HOME/.cache/maniascope"
mkdir -p "$LOGDIR"
exec >>"$LOGDIR/run.log" 2>&1
echo "=== launch $(date '+%F %T') ==="

exec python3 "$HERE/skillsets.py"
