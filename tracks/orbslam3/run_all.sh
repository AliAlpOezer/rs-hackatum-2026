#!/usr/bin/env bash
# Convert, check, run and benchmark every R&S sequence in one go.
#
#   tracks/orbslam3/run_all.sh <dataset_extracted> <work_dir>
#   MODES="stereo stereo_inertial" SEQS="seq01 seq10" tracks/orbslam3/run_all.sh ...
#
# <work_dir> gets euroc/<seq> (converted), data/<seq>/gt.txt (camera-frame GT),
# results/<seq>/orbslam3_<mode>.txt, reports/check_stereo.txt and reports/summary.md.
# Steps already done are skipped (delete the folder to redo). Needs setup.sh first.
# Env: MODES (default: stereo stereo_inertial mono_inertial), SEQS (default: all seq*),
#      CONVERT_ARGS (default: --gray), ORB_SLAM3_DIR (passed on to run.sh).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
[[ $# -eq 2 ]] || { sed -n '2,12p' "$0"; exit 1; }
SRC="$(cd "$1" && pwd)"
WORK="$(mkdir -p "$2" && cd "$2" && pwd)"
MODES="${MODES:-stereo stereo_inertial mono_inertial}"
SEQS="${SEQS:-$(cd "$SRC" && ls -d seq* | xargs)}"
PY=(uv run --project "$ROOT" python)
mkdir -p "$WORK/reports"

for seq in $SEQS; do
  echo "=== $seq"
  if [[ ! -f "$WORK/euroc/$seq/conversion.yaml" ]]; then
    "${PY[@]}" "$ROOT/data/tools/zed_to_euroc.py" "$SRC/$seq" "$WORK/euroc/$seq" ${CONVERT_ARGS:---gray}
  fi
  if [[ -d "$WORK/euroc/$seq/mav0/state_groundtruth_estimate0" && ! -f "$WORK/data/$seq/gt.txt" ]]; then
    "${PY[@]}" "$ROOT/data/tools/euroc_to_seq.py" "$WORK/euroc/$seq" "$WORK/data/$seq" > /dev/null
  fi
  if ! grep -q "^$seq:" "$WORK/reports/check_stereo.txt" 2>/dev/null; then
    "${PY[@]}" "$ROOT/data/tools/check_stereo.py" "$WORK/euroc/$seq" | tee -a "$WORK/reports/check_stereo.txt"
  fi
  for mode in $MODES; do
    if [[ -f "$WORK/results/$seq/orbslam3_$mode.txt" ]]; then continue; fi
    RESULTS_DIR="$WORK/results" "$HERE/run.sh" "$WORK/euroc/$seq" "$mode" \
      | tee -a "$WORK/reports/runs.txt" || echo "$seq $mode: FAILED" | tee -a "$WORK/reports/runs.txt"
  done
done

"${PY[@]}" "$ROOT/eval/benchmark.py" "$WORK/results" --gt-root "$WORK/data" --frames-root "$WORK/euroc" \
  -o "$WORK/reports/summary.md" > /dev/null
echo "benchmark: $WORK/reports/summary.md, stereo check: $WORK/reports/check_stereo.txt"
