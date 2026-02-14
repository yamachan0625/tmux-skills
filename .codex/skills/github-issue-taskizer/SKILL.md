---
name: github-issue-taskizer
description: 複雑な要件を GitHub Issue に分解して実行可能タスクとして起票する。要件定義・PRD・仕様メモから「Issue化して」「タスク分割して」「依存関係付きで起票して」と依頼されたときに使用する。root issue 作成、DAG ノード issue 作成、受け入れ条件の明文化までを一貫して行う。
---

# GitHub Issue Taskizer

## 目的

複雑な要件を、依存関係つきの実装タスク Issue 群へ変換する。

## 実装方針

1. 起票は `scripts/taskize_requirements.py` を使う。
2. DAG ノード生成は `orchestrator-manager/scripts/github_dag_pipeline.py` を再利用する。
3. DAG ノードは root issue に対して GitHub REST sub-issues API で親子リンクを作成する。
4. root issue は `sprint_days: 1-7` とレビュー項目を満たすものだけを受け付ける。
5. `[github].project_url` が設定されている場合は root/node issue を Project に追加する。
6. このスキルは「Issue 作成まで」を担当し、実行ディスパッチは担当しない。

## 事前確認

1. `gh auth status` が成功することを確認する。
2. `.codex/orchestrator/config.toml` の `[core].mode = "github"` を確認する。
3. `.codex/orchestrator/config.toml` の `[github].repo` が正しいことを確認する。
4. 要件本文を Markdown で用意する。構成は `references/requirement-checklist.md` に従う。
5. root issue 本文に `sprint_days` と必須見出しを入れる。
6. Project 連携する場合は `[github].project_url` を設定し、`gh` に project 権限を付与する（`gh auth refresh -s project`）。

## 手順

1. 新規 root issue から起票する場合は以下を実行する。

```bash
python3 .codex/skills/github-issue-taskizer/scripts/taskize_requirements.py \
  --config .codex/orchestrator/config.toml \
  --title "検索APIのレート制御を導入する" \
  --body-file /tmp/requirements.md
```

2. 既存 root issue から起票する場合は以下を実行する。

```bash
python3 .codex/skills/github-issue-taskizer/scripts/taskize_requirements.py \
  --config .codex/orchestrator/config.toml \
  --root-issue 123 \
  --max-sprint-days 7
```

3. 固定プラン(JSON)で起票する場合は `--plan-file` を使う。

```bash
python3 .codex/skills/github-issue-taskizer/scripts/taskize_requirements.py \
  --config .codex/orchestrator/config.toml \
  --root-issue 123 \
  --plan-file /tmp/dag-plan.json
```

## 出力

1. root issue
2. DAG ノード issue（依存付き + root への実親子リンク）
3. root issue への DAG サマリコメント

## 運用ルール

1. タスク粒度は「1 issue = 1 PR で完結」を優先する。
2. 依存が不要なタスクは `depends_on` を空にする。
3. 受け入れ条件は必ず検証可能な文章にする。
4. 実行フェーズに進める場合は `orchestrator-manager` スキルを使う。
