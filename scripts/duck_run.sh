#!/bin/sh
# Run a duck script under the heavy lock, the memory cap and efficiency cores. Usage: duck_run.sh SCRIPT [ARGS]
cd "$(dirname "$0")/.." || exit 1
n=""
for i in "" -2 -3 -4 -5 -6 -7 -8; do
  d="$HOME/helloworld/.heavy-lock$i"
  if [ -d "$d" ] && ! kill -0 "$(cat "$d/owner" 2>/dev/null)" 2>/dev/null; then rm -f "$d/owner"; rmdir "$d" 2>/dev/null; fi
  if mkdir "$d" 2>/dev/null; then n="$d"; break; fi
done
[ -n "$n" ] || { echo "no free heavy slot" >&2; exit 1; }
echo $$ > "$n/owner"
PYTHONPATH=src nice -n 19 taskpolicy -b "$HOME/helloworld/jarvis-rice/bin/mem-cap" --gb 6 -- \
  "$HOME/helloworld/vla-forge/.venv/bin/python" "$@"
rc=$?
rm -f "$n/owner"; rmdir "$n"
exit $rc
