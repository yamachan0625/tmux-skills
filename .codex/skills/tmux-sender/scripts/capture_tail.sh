#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "usage: $0 <pane_id> [lines]" >&2
  exit 1
fi

pane_id="$1"
lines="${2:-80}"

if [[ "$pane_id" != %* ]]; then
  echo "error: pane_id must look like %N (example: %2)" >&2
  exit 1
fi

if ! [[ "$lines" =~ ^[0-9]+$ ]]; then
  echo "error: lines must be an integer" >&2
  exit 1
fi

tmux capture-pane -t "$pane_id" -p -S -"$lines" | tail -n "$lines"
