---
name: orchestrator-manager
description: manager として tmux worker ペインへタスクを配布し、local/github モードで監視・再試行・集約を行う。
---

# Orchestrator Manager

## 目的

pane 0 を manager として使い、pane 1..n の worker にタスクを自律配布する。

## モード

- `local`: SoT はファイル（`[local].queue_dir` / `done_dir` / `failed_dir`）
- `github`: SoT は Issue コメント（worker が `<!-- agent-result:v1 -->` コメントを書き戻す）
- `github + pr-loop`: manager が PR を判定し、`OK -> merge -> done` / `NG -> manager-feedback コメント -> worker 再実装`

## Production 向け制御

1. タイムアウト: `[core].worker_hard_timeout_sec` / `worker_idle_timeout_sec`
2. 全体締切: `[core].run_deadline_sec`
3. 失敗時再試行: `[core].max_attempts` + `retry_backoff_sec`
4. 実行ログ: `.agent/runs/<run_id>/manager.events.log`
5. 出力言語: `[core].output_language`（既定 `ja`）
6. PR ループ: `[github].workflow = "pr-loop"` + `review_label` + `pr_require_checks`
7. DAG: `github_dag_pipeline.py` で依存解決付き Issue 実行

## 前提

1. 設定ファイルを作成する

```bash
cp .codex/orchestrator/config.toml.example .codex/orchestrator/config.toml
```

2. `mode` と `worker_panes`、`github.repo`（github利用時）を設定する

## 実行

```bash
python3 .codex/skills/orchestrator-manager/scripts/manager_dispatch.py \
  --config .codex/orchestrator/config.toml
```

## DAG 実行（GitHub）

```bash
python3 .codex/skills/orchestrator-manager/scripts/github_dag_pipeline.py \
  --config .codex/orchestrator/config.toml \
  --root-issue 123
```

## 実装ポリシー

1. 実行ランタイムは Python（`manager_dispatch.py` / `github_dag_pipeline.py`）を正とする
2. worker 実行は `orchestrator-worker/scripts/worker_run.py` に委譲する
3. 再試行は設定値 `max_attempts` と `retry_backoff_sec` に従う
4. github モードではラベル遷移を manager が管理する
5. timeout が発生した worker は同 run 内で隔離し再利用しない
6. `pr-loop` では manager が `gh pr merge` を試行し、失敗時は feedback を Issue に返して再実装させる
