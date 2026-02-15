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
5. root issue は単一固定ではなく、入力の複雑さとレビュー目的に応じて複数に自動分割する。
6. 分割単位は「ビジネスとエンジニアの認識齟齬検知」「方向性確認」に適したレビュー単位を優先する。
7. `[github].project_url` が設定されている場合は root/node issue を Project に追加する。
8. このスキルは「Issue 作成まで」を担当し、実行ディスパッチは担当しない。
9. root issue 本文テンプレートは `assets/root-issue.template.md` を唯一の正とする。
10. テンプレート各項目の記入基準は `references/root-issue-template-fields.md` を参照する。
11. 子 issue 本文テンプレートの記入基準は `references/child-issue-template-fields.md` を参照する。

## 事前確認

1. `gh auth status` が成功することを確認する。
2. `.codex/orchestrator/config.toml` の `[core].mode = "github"` を確認する。
3. `.codex/orchestrator/config.toml` の `[github].repo` が正しいことを確認する。
4. 要件本文を Markdown で用意する。構成は `references/requirement-checklist.md` に従う。
5. root issue 本文に `sprint_days` と必須見出しを入れる。
6. Project 連携する場合は `[github].project_url` を設定し、`gh` に project 権限を付与する（`gh auth refresh -s project`）。
   権限がない場合、`taskize_requirements.py` は自動で Project 連携を無効化して起票を継続する。
7. root 分割の目的（レビューで何を確認したいか）を明示する。

## 手順

1. 要件本文から root を自動分割して起票する場合は以下を実行する。

```bash
python3 .codex/skills/github-issue-taskizer/scripts/taskize_requirements.py \
  --config .codex/orchestrator/config.toml \
  --title "PORT Phase1 開発初期タスク一覧" \
  --body-file /tmp/requirements.md \
  --review-purpose "認識齟齬検知と方向性確認を定期レビューで行う" \
  --max-roots 6 \
  --planner-timeout-sec 180
```

2. 既存 root issue から DAG ノードだけ起票する場合は以下を実行する。

```bash
python3 .codex/skills/github-issue-taskizer/scripts/taskize_requirements.py \
  --config .codex/orchestrator/config.toml \
  --root-issue 123 \
  --max-sprint-days 7
```

3. 固定プラン(JSON)で起票する場合は `--plan-file` を使う。
`roots[]` を持つ multi-root 形式と `nodes[]` の single-root 形式の両方を受け付ける。
自動プランナー失敗時はローカルヒューリスティック分割へフォールバックする。

```bash
python3 .codex/skills/github-issue-taskizer/scripts/taskize_requirements.py \
  --config .codex/orchestrator/config.toml \
  --title "PORT Phase1 開発初期タスク一覧" \
  --body-file /tmp/requirements.md \
  --plan-file /tmp/root-plan.json
```

## 出力

1. root issue（1件以上、入力に応じて自動分割）
2. DAG ノード issue（依存付き + 各 root への実親子リンク）
3. 各 root issue への DAG サマリコメント
4. run artifacts（`.agent/taskize-runs/<run_id>/` に root plan と summary を保存）

## 運用ルール

1. タスク粒度は「1 issue = 1 PR で完結」を優先する。
2. root 分割はレビュー目的に合わせる。レビュー周期は 1 日でもよい。
3. 依存が不要なタスクは `depends_on` を空にする。
4. root / node のタイトルは日本語で作成する。
5. 受け入れ条件はチェックボックス形式で、全項目を定量評価可能にする（数値・閾値・件数を必須）。
6. 受け入れ条件の書式は固定しない。タスクに応じた表現で、定量評価できる条件を記述する。
7. 実行フェーズに進める場合は `orchestrator-manager` スキルを使う。
