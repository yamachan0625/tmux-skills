# Plan Schema

Create a JSON file and pass it to:

```bash
python3 scripts/create_issue_graph.py --plan /tmp/issue-plan.json
```

## Required structure

```json
{
  "repo": "OWNER/REPO",
  "parents": [
    {
      "id": "parent-auth",
      "title": "認証基盤の改善",
      "background": "現行の認証処理で運用負荷が高い。",
      "purpose": "認証基盤を安定化し障害復旧時間を短縮する。",
      "scope": [
        "ログイン経路の認証処理",
        "トークン更新処理"
      ],
      "acceptance_criteria": [
        {
          "scenario": "正常なログインが成功する",
          "given": "有効なユーザー資格情報がある",
          "when": "ログインAPIを呼び出す",
          "then": "200が返りアクセストークンが発行される"
        },
        {
          "scenario": "無効な資格情報で失敗する",
          "given": "無効なパスワードを入力する",
          "when": "ログインAPIを呼び出す",
          "then": "401が返りトークンは発行されない"
        }
      ],
      "todos": [
        "子Issueがすべて完了したことを確認する",
        "リリース判定を実施して親Issueをクローズする"
      ],
      "questions": [
        "親Issueの完了判定コメントの定型文はありますか？"
      ],
      "out_of_scope": [
        "認証方式の全面刷新"
      ],
      "labels": [
        "feature"
      ],
      "children": [
        {
          "id": "auth-api",
          "title": "ログインAPIの失敗時ハンドリングを改善する",
          "background": "失敗時のエラーレスポンスが不統一。",
          "purpose": "API利用者が失敗原因を正確に判別できるようにする。",
          "scope": [
            "ログインAPIエラーレスポンス",
            "認証失敗ログ出力"
          ],
          "acceptance_criteria": [
            {
              "scenario": "正常系レスポンス",
              "given": "有効な資格情報がある",
              "when": "ログインAPIを呼び出す",
              "then": "200とトークンが返る"
            },
            {
              "scenario": "異常系レスポンス",
              "given": "無効な資格情報を送信する",
              "when": "ログインAPIを呼び出す",
              "then": "401と標準エラーコードが返る"
            }
          ],
          "todos": [
            "失敗系のテストを先に作成する",
            "API処理を修正する",
            "テストを通す"
          ],
          "questions": [
            "標準エラーコード一覧のソースはどれですか？"
          ],
          "depends_on": [
            "auth-schema"
          ]
        },
        {
          "id": "auth-schema",
          "title": "認証エラーコードのDBマッピングを追加する",
          "background": "エラーコード管理がアプリ層に散在している。",
          "purpose": "エラーコード定義を一元化する。",
          "scope": [
            "エラーコード管理テーブル",
            "マイグレーション"
          ],
          "acceptance_criteria": [
            {
              "scenario": "マイグレーション適用",
              "given": "最新スキーマのDBがある",
              "when": "マイグレーションを実行する",
              "then": "エラーコードテーブルが作成される"
            },
            {
              "scenario": "既存機能への非影響",
              "given": "既存認証処理が稼働している",
              "when": "デプロイ後に認証APIを実行する",
              "then": "既存挙動が維持される"
            },
            {
              "scenario": "ロールバック可能",
              "given": "マイグレーション適用済み",
              "when": "ダウングレードを実行する",
              "then": "スキーマが元に戻る"
            }
          ],
          "todos": [
            "マイグレーションの失敗テストを先に追加する",
            "マイグレーションを実装する",
            "適用とロールバックを検証する"
          ],
          "questions": [
            "ロールバックの許容停止時間は何分ですか？"
          ]
        }
      ]
    }
  ]
}
```

## Field notes

- `repo` is optional. When omitted, script resolves current repository.
- `parents[*].children` must contain at most 10 items.
- `id` is optional but recommended. Use stable IDs and refer to them from `depends_on`.
- `depends_on` supports child `id` or exact child `title`.
- `todos` is required. `tasks` is accepted for backward compatibility.
- `questions` is optional but required when requirements are undecidable from input.
- `labels` is optional. Script also infers labels automatically.
- `out_of_scope` is optional.
- Do not reuse identical `acceptance_criteria` or identical `todos` across different issues.
- Avoid generic wording. Each issue should include concrete objects (API/table/metric/rule).

## Validation behavior

- Script validates required sections.
- Script validates minimum Gherkin scenario count by issue type.
- Script checks duplicate titles before creation.
- Script stops entirely when duplicates are found.
