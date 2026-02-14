#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <pane_id> <command...>" >&2
  exit 1
fi

pane_id="$1"
shift
command="$*"

if [[ "$pane_id" != %* ]]; then
  echo "error: pane_id must look like %N (example: %2)" >&2
  exit 1
fi

tmux send-keys -t "$pane_id" -l -- "$command"
tmux send-keys -t "$pane_id" Enter
