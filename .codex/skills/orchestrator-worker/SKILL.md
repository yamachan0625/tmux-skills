---
name: orchestrator-worker
description: worker として 1 タスクを実行し、結果を local はファイル、github は Issue コメントへ書き戻す。
---

# Orchestrator Worker

## 目的

manager から受け取った単一タスクを実行して、結果を SoT に反映する。

## SoT

- `local`: `[local].result_dir`（既定 `.agent/results`）と status ファイル
- `github`: Issue コメント（`<!-- agent-result:v1 -->`）
- `github + pr-loop`: PR URL を最終出力へ含め、manager feedback（`<!-- manager-feedback:v1 -->`）を再実装時に取り込む

## 実行

```bash
python3 .codex/skills/orchestrator-worker/scripts/worker_run.py \
  --config .codex/orchestrator/config.toml \
  --run-id 20260214-120000 \
  --mode local \
  --task-key sample-task \
  --attempt 1 \
  --worker-pane %1 \
  --manager-pane %0 \
  --task-file .agent/tasks/sample-task.md
```

github モードでは `--task-file` の代わりに `--issue-number` を使う。

## 実装ポリシー

1. 実行ランタイムは Python（`worker_run.py`）を正とする
2. 実行は `codex exec` を使う
3. status は `.agent/runs/<run_id>/status/*.env` に原子的に書き込む
4. github は `comment_token` で重複投稿を避け、投稿後検証を行う
5. 完了通知は manager pane へ `__ORCH_DONE__` 行を送る
6. 出力言語は `[core].output_language`（既定 `ja`）で制御する
7. `pr-loop` では最終回答に `PR_URL: <url>` を必須で含める
