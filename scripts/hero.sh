#!/usr/bin/env bash
# The hero pair in one command: intake check, copy into data/own/<date>/, cost and time estimate against the caps,
# the live run (--keep, our footage credit), walkaround verify, a short verdict summary, the run made into the first
# sample (the friend_car slot of samples/index.json), then the commands for the video recapture and the Railway
# redeploy, printed for the lead to run. It never commits, pushes, deploys or prints a key.
#
#   scripts/hero.sh PICKUP.MOV RETURN.MOV [RUN_ID]            the whole thing (RUN_ID defaults to hero_MMDD)
#   scripts/hero.sh --dry-run PICKUP.MOV RETURN.MOV [RUN_ID]  every step before the first model call, then stop
#   scripts/hero.sh --after RUN_ID                             the steps after the run, on a finished run
#
# Options: --dest DIR (where the clips, intake frames, run log and contact sheets go; default data/own/<today>),
# --samples-dir DIR (make the sample in DIR, seeded with samples/index.json, instead of samples/), --capture-test DIR
# (record one scene of the run into DIR with video/capture.sh), --no-sample, --force (go on when the intake or the
# estimate says no). Env: TIRE_CODE=225/60R16 (the wheel ruler states sizes), PLATE_BLOCK=12 (pixelate every plate and
# face box again in the sample), VIDEO_DIR, RAILWAY_DIR (a clone linked to the Railway service walkaround-web).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
WA="$REPO/.venv/bin/walkaround"
VIDEO_DIR="${VIDEO_DIR:-$(cd "$REPO/.." && pwd)/video}"
RAILWAY_DIR="${RAILWAY_DIR:-/path/to/railway-clone}"
CREDIT="Footage: filmed by us with the car owner's permission; the marks are blue tape stand-ins"
TITLE="A friend's own car, filmed at pickup and at return"
SLOT=friend_car
NOKEYS=(MODEL_API_KEY= BASETEN_API_KEY= BASETEN_HACKATHON_API_KEY= HF_TOKEN=)

DRY=0; AFTER=""; DEST=""; SAMPLES_DIR=""; CAPTURE_TEST=""; NO_SAMPLE=0; FORCE=0; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --after) AFTER="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --samples-dir) SAMPLES_DIR="$2"; shift 2 ;;
    --capture-test) CAPTURE_TEST="$2"; shift 2 ;;
    --no-sample) NO_SAMPLE=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) POS+=("$1"); shift ;;
  esac
done

say() { printf '\n== %s\n' "$*"; }
die() { echo "STOP: $*" >&2; exit 1; }
abspath() { (cd "$(dirname "$1")" && printf '%s/%s\n' "$(pwd)" "$(basename "$1")"); }
for tool in ffmpeg ffprobe shasum; do command -v "$tool" >/dev/null || die "$tool is not on PATH"; done
[ -x "$WA" ] || die "no $WA (make the venv first)"
DEST="${DEST:-$REPO/data/own/$(date +%F)}"
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
cd "$REPO"

run_status() { "$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$REPO/runs/$1/run.json" 2>/dev/null || true; }

if [ -n "$AFTER" ]; then
  RUN_ID="$AFTER"
  [ -f "runs/$RUN_ID/run.json" ] || die "no run $RUN_ID in runs/"
else
  [ "${#POS[@]}" -ge 2 ] || die "usage: scripts/hero.sh [--dry-run] PICKUP.MOV RETURN.MOV [RUN_ID]  (or --after RUN_ID)"
  SRC_P="$(abspath "${POS[0]}")"; SRC_R="$(abspath "${POS[1]}")"
  RUN_ID="${POS[2]:-hero_$(date +%m%d)}"
  [[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]] || die "run id $RUN_ID: letters, digits, _ and - only"
  for f in "$SRC_P" "$SRC_R"; do [ -f "$f" ] || die "not a file: $f"; done
  STATUS="$(run_status "$RUN_ID")"
  case "$STATUS" in
    "") ;;
    done|partial|failed) die "run $RUN_ID already exists ($STATUS); pass another run id, or --after $RUN_ID" ;;
    *) echo "run $RUN_ID exists unfinished ($STATUS): the live run resumes it" ;;
  esac

  say "1. Intake check"
  mkdir -p "$DEST/intake"
  bad=0
  for f in "$SRC_P" "$SRC_R"; do
    name="$(basename "$f")"; stem="${name%.*}"
    if ! "$PY" -I tools/data_hunt/intake_check.py "$f" "$DEST/intake/$stem" --min-sec 45 --max-sec 60 > "$DEST/intake/$stem.json"; then bad=1; fi
    "$PY" scripts/hero_check.py intake "$DEST/intake/$stem.json" | sed 's/^/  /'
  done
  SHA_P="$("$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("sha256",""))' "$DEST/intake/$(basename "${SRC_P%.*}").json")"
  SHA_R="$("$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("sha256",""))' "$DEST/intake/$(basename "${SRC_R%.*}").json")"
  [ "$SHA_P" != "$SHA_R" ] || die "pickup and return are the same file"
  echo "  frames for a look: $DEST/intake/"
  if [ "$bad" = 1 ]; then
    [ "$FORCE" = 1 ] && echo "  a hard check failed; going on (--force)" || die "a hard check failed (film again, or --force)"
  fi

  say "2. Copy into $DEST"
  copy_in() {  # $1 source, $2 its sha256; prints the path in DEST
    local src="$1" sha="$2" dst
    dst="$DEST/$(basename "$src")"
    if [ "$src" != "$dst" ]; then
      if [ -e "$dst" ]; then
        [ "$(shasum -a 256 "$dst" | cut -d' ' -f1)" = "$sha" ] || die "$dst exists with other bytes"
      else
        cp -pc "$src" "$dst" 2>/dev/null || cp -p "$src" "$dst"
        [ "$(shasum -a 256 "$dst" | cut -d' ' -f1)" = "$sha" ] || die "copy of $src does not match its hash"
      fi
    fi
    echo "$dst"
  }
  P="$(copy_in "$SRC_P" "$SHA_P")"; R="$(copy_in "$SRC_R" "$SHA_R")"
  echo "  pickup $P"; echo "  return $R (hashes match the originals)"

  say "3. Estimate against the caps (ledger rows of lexi_pair2 and synth_pair2_r9)"
  est=0; "$PY" scripts/hero_check.py estimate "$P" "$R" | sed 's/^/  /' || est=$?
  if [ "$est" != 0 ]; then
    [ "$FORCE" = 1 ] && echo "  at risk; going on (--force)" || die "the run could stop on a cap: raise it in config/thresholds.json (with a source) or pass --force"
  fi

  RUN_ARGS=(run "$P" "$R" --run-id "$RUN_ID" --keep --footage-credit "$CREDIT")
  [ -n "${TIRE_CODE:-}" ] && RUN_ARGS+=(--tire-code "$TIRE_CODE")
  say "4. Live run check (no LIVE, no keys: walkaround only says what it would do)"
  check="$(env -u LIVE "${NOKEYS[@]}" "$WA" "${RUN_ARGS[@]}" 2>&1 || true)"
  echo "  $check"
  case "$check" in *"would run a pair"*) ;; *) die "walkaround refused the arguments" ;; esac
  LIVE_CMD="LIVE=1 .venv/bin/walkaround run \"$P\" \"$R\" --run-id $RUN_ID --keep --footage-credit \"$CREDIT\"${TIRE_CODE:+ --tire-code $TIRE_CODE}"
  if [ "$DRY" = 1 ]; then
    say "Dry run: stopped before any model call. The live run would be:"
    echo "  cd $REPO && $LIVE_CMD"
    exit 0
  fi

  say "5. Live run $RUN_ID (events in $DEST/$RUN_ID.run.log)"
  started=$(date +%s)
  : > "$DEST/$RUN_ID.run.log"
  "$PY" -u scripts/hero_check.py watch "$DEST/$RUN_ID.run.log" $$ &
  watch_pid=$!
  set +e
  LIVE=1 "$WA" "${RUN_ARGS[@]}" > "$DEST/$RUN_ID.run.log" 2>&1  # in the foreground, so Ctrl-C stops the run
  rc=$?
  set -e
  sleep 2; kill "$watch_pid" 2>/dev/null || true; wait "$watch_pid" 2>/dev/null || true
  echo "  walkaround exit $rc after $(( ($(date +%s) - started) / 60 )) min $(( ($(date +%s) - started) % 60 )) s"
  STATUS="$(run_status "$RUN_ID")"
  case "$STATUS" in
    done) ;;
    partial) echo "  PARTIAL: read run.json errors below before using it as the sample" ;;
    *) tail -n 5 "$DEST/$RUN_ID.run.log" | cut -c1-300; die "run $RUN_ID ended $STATUS" ;;
  esac
fi

say "6. Verify (no network)"
vout="$(env "${NOKEYS[@]}" "$WA" verify "$RUN_ID" 2>&1 || true)"
echo "$vout" | tail -n 2 | sed 's/^/  /'
VERIFIED=0
case "$vout" in *"rebuild identical"*) VERIFIED=1 ;; esac

say "7. Verdict summary"
"$PY" scripts/hero_check.py summary "$RUN_ID" | sed 's/^/  /'

if [ "$VERIFIED" != 1 ]; then
  [ "$FORCE" = 1 ] && echo "  verify did not print rebuild identical; going on (--force)" || die "verify did not print rebuild identical: no sample made"
fi

if [ "$NO_SAMPLE" = 0 ]; then
  say "8. Sample (first in samples/index.json, slot $SLOT)"
  SARGS=()
  if [ -n "$SAMPLES_DIR" ]; then
    mkdir -p "$SAMPLES_DIR"; SAMPLES_DIR="$(cd "$SAMPLES_DIR" && pwd)"
    [ -f "$SAMPLES_DIR/index.json" ] || cp samples/index.json "$SAMPLES_DIR/index.json"
    SARGS+=(--samples-dir "$SAMPLES_DIR")
  fi
  SDIR="${SAMPLES_DIR:-$REPO/samples}"
  [ -n "${PLATE_BLOCK:-}" ] && SARGS+=(--plate-block "$PLATE_BLOCK")
  read -r -a CLIPS <<< "$("$PY" scripts/hero_check.py clips "$RUN_ID")"
  MS=(scripts/make_sample.py "$RUN_ID" --slot "$SLOT" --first --title "$TITLE")
  if ! "$PY" "${MS[@]}" ${CLIPS[@]+"${CLIPS[@]}"} ${SARGS[@]+"${SARGS[@]}"} --dry-run > "$DEST/$RUN_ID.sample_plan.txt" 2>&1; then
    echo "  over the size budget with every clip of the panel; keeping only the focus marks and the control"
    read -r -a CLIPS <<< "$("$PY" scripts/hero_check.py clips "$RUN_ID" --minimal)"
  fi
  echo "  clips: ${CLIPS[*]:-none}"
  "$PY" "${MS[@]}" ${CLIPS[@]+"${CLIPS[@]}"} ${SARGS[@]+"${SARGS[@]}"} | sed 's/^/  /'
  if [ -n "$SAMPLES_DIR" ]; then
    "$PY" scripts/make_sample.py --check "$SDIR/$RUN_ID" | sed 's/^/  /'
  else
    "$PY" scripts/make_sample.py --check | sed 's/^/  /'
  fi
  for role in pickup return; do
    v="$SDIR/$RUN_ID/media/$role.proxy_blurred.mp4"
    [ -f "$v" ] && ffmpeg -v error -y -i "$v" -vf "fps=1/3,scale=480:-2,tile=5x4" -frames:v 1 "$DEST/${RUN_ID}_${role}_contact.jpg" || true
  done
  echo "  look for a readable plate or face before committing: $DEST/${RUN_ID}_*_contact.jpg and $SDIR/$RUN_ID/crops/"
  echo "  if one shows: PLATE_BLOCK=12 scripts/hero.sh --after $RUN_ID"
fi

if [ -n "$CAPTURE_TEST" ]; then
  say "9. Capture test: one scene of $RUN_ID into $CAPTURE_TEST"
  mkdir -p "$CAPTURE_TEST"
  (cd "$VIDEO_DIR" && PAIR="$RUN_ID" MOMENT="$RUN_ID" CAPTURE_DIR="$(cd "$CAPTURE_TEST" && pwd)" ./capture.sh --only PAIR__masks 2>&1 | tail -n 3 | sed 's/^/  /')
  ls -l "$CAPTURE_TEST" | tail -n +2 | sed 's/^/  /'
fi

say "Next, for the lead (nothing below was run)"
cat <<EOF
  Video recapture with this run (video/kit_v4/RECAPTURE_LIST.md; P3 masks, P10 coverage; a clean clone keeps the take
  steady while the repo changes):
    git -C $VIDEO_DIR/work/repo_head fetch -q origin main && git -C $VIDEO_DIR/work/repo_head checkout -q --detach FETCH_HEAD
    cd $VIDEO_DIR && WALKAROUND_SRC=$VIDEO_DIR/work/repo_head PAIR=$RUN_ID MOMENT=$RUN_ID CAPTURE_DIR=$VIDEO_DIR/capture_own ./capture.sh --only PAIR__masks,PAIR__coverage
  The tape, moment, crops and blur scenes are pinned to trunk_lid or rear_bumper in video/beats.tsv; this run's page
  opens on the panel named in the summary above, so P4 and P12 need that panel in the beat (or Plan B, pairC_core3).
  Railway redeploy, after the sample is committed and pushed:
    git -C $RAILWAY_DIR pull --ff-only
    cd $RAILWAY_DIR && railway up --detach -m "sample $RUN_ID"
    curl -s https://walkaround-web-production.up.railway.app/api/sample | head -c 200
EOF
