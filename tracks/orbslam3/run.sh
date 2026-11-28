#!/usr/bin/env bash
# Run ORB-SLAM3 on one sequence converted by data/tools/zed_to_euroc.py.
#
#   tracks/orbslam3/run.sh <euroc_seq_dir> [stereo|stereo_inertial|mono_inertial]   (default stereo_inertial)
#
# Writes results/<seq>/orbslam3_<mode>.txt (TUM, T_world_cam, seconds). The raw ORB-SLAM3
# output (EuRoC format, ns, body pose for inertial modes) and the log go to results/<seq>/raw/,
# so eval/ globbing results/<seq>/*.txt only sees contract files.
# stereo and mono_inertial open the Pangolin viewer; without $DISPLAY they run under xvfb-run.
# They often segfault in viewer teardown after saving; that is reported, not treated as failure.
# Env overrides: ORB_SLAM3_DIR, RESULTS_DIR, PYTHON.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ORB_DIR="${ORB_SLAM3_DIR:-$HERE/ORB_SLAM3}"
[[ $# -ge 1 ]] || { sed -n '2,10p' "$0"; exit 1; }
SEQ="$(cd "$1" && pwd)"
MODE="${2:-stereo_inertial}"
NAME="$(basename "$SEQ")"
OUT="${RESULTS_DIR:-$ROOT/results}/$NAME"

case "$MODE" in
  stereo)          EXE=Examples/Stereo/stereo_euroc;                     FRAME=camera; VIEWER=1 ;;
  stereo_inertial) EXE=Examples/Stereo-Inertial/stereo_inertial_euroc;   FRAME=body;   VIEWER=0 ;;
  mono_inertial)   EXE=Examples/Monocular-Inertial/mono_inertial_euroc;  FRAME=body;   VIEWER=1 ;;
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac
SETTINGS="$SEQ/orbslam3_$MODE.yaml"
for f in "$ORB_DIR/$EXE" "$ORB_DIR/Vocabulary/ORBvoc.txt" "$SETTINGS" "$SEQ/timestamps.txt" "$SEQ/mav0"; do
  [[ -e "$f" ]] || { echo "missing $f (run setup.sh / zed_to_euroc.py first)" >&2; exit 1; }
done

if [[ -n "${PYTHON:-}" ]]; then PY=($PYTHON)
elif command -v uv >/dev/null; then PY=(uv run --project "$ROOT" python)
else PY=(python3); fi
WRAP=()
if [[ "$VIEWER" == 1 && -z "${DISPLAY:-}" ]]; then WRAP=(xvfb-run -a -s "-screen 0 1280x1024x24"); fi

mkdir -p "$OUT/raw"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
LOG="$OUT/raw/orbslam3_$MODE.log"
echo "ORB-SLAM3 $MODE on $NAME -> $OUT"
set +e
(cd "$WORK" && "${WRAP[@]}" "$ORB_DIR/$EXE" "$ORB_DIR/Vocabulary/ORBvoc.txt" "$SETTINGS" \
  "$SEQ" "$SEQ/timestamps.txt" "$NAME") > "$LOG" 2>&1
STATUS=$?
set -e

# What to record in notes.md: crash, init delay, tracked frames, IMU init, resets, maps
count() { grep -c "$1" "$LOG" || true; }
N_FRAMES="$(wc -l < "$SEQ/timestamps.txt")"
echo "exit status: $STATUS (log: $LOG)"
if [[ -s "$WORK/f_$NAME.txt" ]]; then
  awk -v f0="$(head -1 "$SEQ/timestamps.txt")" -v n="$N_FRAMES" 'NR==1 {t0=$1} END {
    printf "first pose %.2f s after the first frame, %d/%d frames in the saved map\n", (t0-f0)/1e9, NR, n}' \
    "$WORK/f_$NAME.txt"
fi
echo "'not enough acceleration': $(count 'not enough acceleration'), resets: $(count 'Reseting active map')," \
     "maps: $(grep -o 'There are [0-9]* maps' "$LOG" | head -1 | grep -o '[0-9]*' || echo '?')"
[[ "$FRAME" == body ]] && echo "IMU init stages done: VIBA1 $(count 'end VIBA 1'), VIBA2 $(count 'end VIBA 2')"

if [[ ! -s "$WORK/f_$NAME.txt" ]] || ! grep -q "End of saving trajectory" "$LOG"; then
  echo "FAILED: no complete trajectory written (exit $STATUS)" >&2; exit 1
fi
if [[ "$STATUS" != 0 ]]; then
  echo "note: exit $STATUS came after the trajectory was saved (viewer teardown); trajectory is complete"
fi
RAW="$OUT/raw/orbslam3_${MODE}_f.txt"
cp "$WORK/f_$NAME.txt" "$RAW"
cp "$WORK/kf_$NAME.txt" "$OUT/raw/orbslam3_${MODE}_kf.txt" 2>/dev/null || true
CONVERT=("${PY[@]}" "$HERE/orbslam3_to_tum.py" "$RAW" "$OUT/orbslam3_$MODE.txt" --frame "$FRAME")
[[ "$FRAME" == body ]] && CONVERT+=(--settings "$SETTINGS")
"${CONVERT[@]}"
