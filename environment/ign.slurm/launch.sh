#!/bin/bash
set -euo pipefail

VERBOSE="${VERBOSE:-false}"

# Usage:
#   GIT_ROOT=/chemin/vers/video_to_ply ./submit.sh

: "${GIT_ROOT:?❌ GIT_ROOT is not set. Example: GIT_ROOT=/path/to/video_to_ply ./submit.sh}"

LAUNCH_SLURM="$GIT_ROOT/environment/ign.slurm/launch_slurm.sh"
RUN_SH="$GIT_ROOT/run.sh"
CONFIG_SH="./config.sh"
LOG_DIR="/mnt/common/hdd/slurm/logs"
SUBMIT_LOG="$LOG_DIR/submit.log"

mkdir -p "$LOG_DIR"

# Log dans terminal + fichier
exec > >(tee -a "$SUBMIT_LOG") 2>&1

echo "========================"
echo "🚀 PRE-SUBMISSION CHECK"
echo "========================"
echo "date         : $(date)"
echo "host         : $(hostname)"
echo "user         : $(whoami)"
echo "pwd          : $(pwd)"
echo "GIT_ROOT     : $GIT_ROOT"
echo "launch.slurm : $LAUNCH_SLURM"
echo "run.sh       : $RUN_SH"
echo "config.sh    : $CONFIG_SH"
echo "log dir      : $LOG_DIR"
echo "submit log   : $SUBMIT_LOG"

[ -d "$GIT_ROOT" ] || { echo "❌ GIT_ROOT not found: $GIT_ROOT"; exit 1; }
[ -f "$LAUNCH_SLURM" ] || { echo "❌ launch.slurm missing: $LAUNCH_SLURM"; exit 1; }
[ -f "$RUN_SH" ] || { echo "❌ run.sh missing: $RUN_SH"; exit 1; }
[ -f "$CONFIG_SH" ] || { echo "❌ config.sh missing: $CONFIG_SH"; exit 1; }

echo
echo "========================"
echo "📤 SUBMITTING JOB"
echo "========================"

OUT=$(sbatch "$LAUNCH_SLURM")
echo "$OUT"

JOB_ID=$(echo "$OUT" | awk "{print \$4}")
if [ -z "${JOB_ID:-}" ]; then
  echo "❌ Could not parse job ID from sbatch output"
  exit 1
fi

STDOUT_LOG="$LOG_DIR/gsplat-$JOB_ID.out"
STDERR_LOG="$LOG_DIR/gsplat-$JOB_ID.err"


echo "✅ Submitted job $JOB_ID"
echo "📄 stdout: $STDOUT_LOG"
echo "📄 stderr: $STDERR_LOG"
echo "🔎 queue: squeue -j $JOB_ID"

echo
echo "========================"
echo "⏳ WAITING FOR LOG FILES"
echo "========================"

for i in $(seq 1 60); do
  if [ -f "$STDOUT_LOG" ] || [ -f "$STDERR_LOG" ]; then
    break
  fi
  sleep 1
done

touch "$STDOUT_LOG" "$STDERR_LOG"

echo
echo "========================"
echo "📡 STREAMING JOB LOGS"
echo "========================"
echo "The script will stop automatically when the Slurm job ends."
echo "Press Ctrl+C to stop following logs manually (the job will continue running)."
echo

if [ "$VERBOSE" = "true" ]; then
  tail -n +1 -f "$STDOUT_LOG" "$STDERR_LOG" &
else
  # Cache les messages "d'analyse" bavards du process
  tail -n +1 -f "$STDOUT_LOG" "$STDERR_LOG" | grep -vE \
    '^\[INFO\]|^\[DEBUG\]|^INFO:|^DEBUG:|it/s|step=|epoch=|loss=|Loading|Saving|Caching|Downloading|Analyzing|Processing|Rendering|Iteration|Checkpoint|Progress' &
fi

TAIL_PID=$!

cleanup() {
  kill "$TAIL_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Attendre la fin du job Slurm
while true; do
  if squeue -j "$JOB_ID" -h | grep -q .; then
    sleep 2
  else
    break
  fi
done

# Stopper le tail et afficher la fin des logs
kill "$TAIL_PID" 2>/dev/null || true
wait "$TAIL_PID" 2>/dev/null || true


echo
echo "========================"
echo "🏁 JOB FINISHED"
echo "========================"

echo "--- Last lines of stdout ---"
tail -n 30 "$STDOUT_LOG" || true
echo
echo "--- Last lines of stderr ---"
tail -n 30 "$STDERR_LOG" || true
echo

# Essayer de récupérer l'état final du job
FINAL_STATE=$(sacct -j "$JOB_ID" --format=State --noheader 2>/dev/null | awk "NF {print \$1; exit}")
EXIT_CODE=$(sacct -j "$JOB_ID" --format=ExitCode --noheader 2>/dev/null | awk "NF {print \$1; exit}")

echo "Final state: ${FINAL_STATE:-unknown}"
echo "Exit code  : ${EXIT_CODE:-unknown}"

# Retourner une erreur si le job a échoué
case "${FINAL_STATE:-}" in
  COMPLETED)
    exit 0
    ;;
  *)
    exit 1
    ;;
esac
