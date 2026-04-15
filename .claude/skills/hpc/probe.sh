#!/bin/bash
# Submit a real `sleep 1` probe job with the given sbatch flags, poll squeue
# until it transitions PENDING -> RUNNING, and report the wait time.
#
# Usage:
#   probe.sh <label> <sbatch_flag> [<sbatch_flag>...]
#
# Example:
#   probe.sh cpu --partition=day --time=5:00:00 --cpus-per-task=2 --mem=12G
#
# Env:
#   TIMEOUT_SEC  How long to wait before scancelling the probe (default: 300s)
#   POLL_SEC     Polling interval in seconds (default: 2)
set -u

LABEL="${1:?usage: probe.sh <label> <sbatch flags...>}"
shift
FLAGS=("$@")

TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
POLL_SEC="${POLL_SEC:-2}"

jobid=""
cleanup() {
  if [[ -n "$jobid" ]]; then
    scancel "$jobid" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

submit_epoch=$(date +%s)
out=$(sbatch "${FLAGS[@]}" --wrap="sleep 1" 2>&1)
rc=$?
if (( rc != 0 )); then
  echo "[$LABEL] submit failed (rc=$rc): $out"
  exit 1
fi

jobid=$(echo "$out" | awk '/^Submitted batch job/ {print $NF; exit}')
if ! [[ "$jobid" =~ ^[0-9]+$ ]]; then
  echo "[$LABEL] could not parse jobid from: $out"
  exit 1
fi

echo "[$LABEL] submitted $jobid (flags: ${FLAGS[*]})"

elapsed=0
while (( elapsed < TIMEOUT_SEC )); do
  state=$(squeue -h -j "$jobid" -o "%T" 2>/dev/null)

  if [[ -z "$state" ]]; then
    # Job no longer in squeue — either ran and completed, or vanished.
    # Use sacct to recover the actual Start time.
    start_str=$(sacct -j "${jobid}.batch" -o Start -X -P -n 2>/dev/null | head -1)
    if [[ -z "$start_str" || "$start_str" == "Unknown" ]]; then
      start_str=$(sacct -j "$jobid" -o Start -X -P -n 2>/dev/null | head -1)
    fi
    if [[ -n "$start_str" && "$start_str" != "Unknown" ]]; then
      start_epoch=$(date -d "$start_str" +%s 2>/dev/null || echo "")
      if [[ -n "$start_epoch" ]]; then
        wait_sec=$((start_epoch - submit_epoch))
        (( wait_sec < 0 )) && wait_sec=0
        echo "[$LABEL] started after ${wait_sec}s (completed; recovered via sacct)"
        exit 0
      fi
    fi
    echo "[$LABEL] job left queue before state read (wait < ${POLL_SEC}s)"
    exit 0
  fi

  if [[ "$state" == "RUNNING" ]]; then
    echo "[$LABEL] started after ~${elapsed}s"
    # Let sleep 1 finish on its own.
    exit 0
  fi

  sleep "$POLL_SEC"
  elapsed=$((elapsed + POLL_SEC))
done

# Timeout — record reason and cancel.
reason=$(squeue -h -j "$jobid" -o "%r" 2>/dev/null)
scancel "$jobid" 2>/dev/null || true
jobid=""
echo "[$LABEL] still PENDING after ${TIMEOUT_SEC}s (reason: ${reason:-unknown}); cancelled"
exit 0
