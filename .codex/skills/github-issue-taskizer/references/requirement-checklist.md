# Requirement Checklist

要件を Issue 化する前に、最低限以下を明示する。

## Root Issue Timebox Policy

1. 親 issue は 1 スプリント内で完了できる単位にする
2. `sprint_days` は 1-7 の整数にする
3. 非エンジニアがレビュー可能な内容を明示する

## Root 分割原則（複数 root の場合）

1. 分割基準は「認識齟齬検知」と「方向性確認」のレビュー目的を優先する
2. 入力がシンプルなら root は 1 件に留める
3. 入力が混在・大量なら root を複数化し、各 root をレビュー可能な単位にする
4. レビュー周期は手段なので 1 日単位でも許容する

## 必須項目

1. 背景と目的
2. スコープ（やること）
3. 非スコープ（やらないこと）
4. 完了条件（Definition of Done）
5. 制約（期限、技術、運用、コスト）
6. リスクまたは未確定事項
7. レビュー目的（何のズレを検知し、何を意思決定するか）

## タスク分割の基準

1. 各タスクは 1 PR で完結する粒度にする
2. 依存関係は `depends_on` に明示する
3. 受け入れ条件は検証可能な文にする
4. 同時並行可能なタスクは依存を付けない
5. 設計調査だけのタスクを常設せず、実装成果物を必須にする
6. 受け入れ条件はチェックボックス形式（`- [ ]`）で記述する
7. 受け入れ条件の書式は固定しないが、各項目を定量評価可能（数値/閾値/件数あり）にする

## 子 Issue Template

`orchestrator-manager/assets/dag-node-issue-body.template.md` を利用する。
各項目の具体的な記入基準は `references/child-issue-template-fields.md` を参照する。

## タイトル品質

1. 動詞で始める（例: `Add`, `Refactor`, `Implement`）
2. 40-70 文字程度で対象と目的を含める
3. 抽象語だけのタイトルを避ける（例: `対応する`, `改善`）

## Root Issue Template

`assets/root-issue.template.md` を唯一のテンプレートとして利用する。
各項目の具体的な記入基準は `references/root-issue-template-fields.md` を参照する。
