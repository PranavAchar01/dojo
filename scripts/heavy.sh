#!/bin/bash
# Run a heavy step (training, evaluation, rendering) under the workspace heavy-job lock:
# take the first free ~/helloworld/.heavy-lock[-N] slot, run at low priority, release the slot on exit.
# Skips the step when memory_pressure reports less than 10% free or the disk has less than 15 GiB.
set -u
LOCKROOT="$HOME/helloworld/.heavy-lock"
free_pct=$(memory_pressure | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
disk_gib=$(df -g "$HOME" | awk 'NR==2 {print $4}')
if [ "${free_pct:-0}" -lt 10 ] || [ "${disk_gib:-0}" -lt 15 ]; then
  echo "heavy.sh: skipped (memory free ${free_pct}%, disk ${disk_gib} GiB)" >&2
  exit 75
fi
slot=""
n=1
while [ -z "$slot" ]; do
  cand="$LOCKROOT"; [ "$n" -gt 1 ] && cand="$LOCKROOT-$n"
  if mkdir "$cand" 2>/dev/null; then
    slot="$cand"
  elif [ -f "$cand/owner" ] && ! kill -0 "$(cat "$cand/owner" 2>/dev/null)" 2>/dev/null; then
    rm -r "$cand" && mkdir "$cand" 2>/dev/null && slot="$cand"
  fi
  n=$((n + 1))
done
echo $$ > "$slot/owner"
echo "vla-forge: $*" > "$slot/what"
trap 'rm -r "$slot"' EXIT
# VLA_FORGE_TIMED=1: timed product runs keep nice 19 but skip the background QoS clamp (taskpolicy -b pins the
# process to efficiency cores and slows MuJoCo 3-4x, which would make the 3-minute measurement meaningless).
if [ "${VLA_FORGE_TIMED:-0}" = "1" ]; then
  nice -n 19 "$@"
else
  nice -n 19 taskpolicy -b "$@"
fi
