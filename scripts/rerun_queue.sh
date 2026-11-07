#!/bin/bash
# Re-run old skills one after another under a heavy-lock slot and a memory cap: scripts/rerun_queue.sh <skill id>...
set -u
cd "$(dirname "$0")/.."
LOCKROOT="$HOME/helloworld/.heavy-lock"
slot=""; n=1
while [ -z "$slot" ]; do
  cand="$LOCKROOT"; [ "$n" -gt 1 ] && cand="$LOCKROOT-$n"
  if mkdir "$cand" 2>/dev/null; then slot="$cand"
  elif [ -f "$cand/owner" ] && ! kill -0 "$(cat "$cand/owner" 2>/dev/null)" 2>/dev/null; then
    rm -r "$cand" && mkdir "$cand" 2>/dev/null && slot="$cand"
  fi
  n=$((n + 1))
done
echo $$ > "$slot/owner"; echo "vla-forge rerun: $*" > "$slot/what"
trap 'rm -r "$slot"' EXIT
export VLA_FORGE_DATA_WORKERS=2 VLA_FORGE_EVAL_WORKERS=2 VLA_FORGE_EVAL_CHUNK=3 PYTHONPATH=src HF_HUB_OFFLINE=1
nice -n 19 taskpolicy -b ~/helloworld/jarvis-rice/bin/mem-cap --gb 12 -- ~/helloworld/vla-forge/.venv/bin/python scripts/rerun.py "$@"
