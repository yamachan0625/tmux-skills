# Label Mapping

`create_issue_graph.py` infers canonical labels from issue text, then maps them to existing repository labels.

## Canonical labels

- `bug`
- `feature`
- `chore`
- `docs`
- `api`
- `ui`
- `db`
- `test`
- `security`
- `performance`
- `infra`
- `spike`
- `release`

## Mapping behavior

1. Detect canonical labels from keywords in title/background/purpose/scope/tasks.
2. Include explicit labels in plan when they already exist in repo.
3. Resolve canonical label by alias list (example: `feature` -> `feature`, `enhancement`, `type:feature`).
4. Keep only labels that actually exist in the target repository.

The script does not auto-create labels. If label set is empty after mapping, issue is created without labels.

## Keyword examples

- Bug: `不具合`, `障害`, `error`, `bug`
- Docs: `ドキュメント`, `仕様書`, `readme`
- DB: `migration`, `database`, `schema`, `テーブル`
- Security: `認可`, `認証`, `権限`, `security`
- Spike: `調査`, `検証`, `spike`, `poc`
- Release: `リリース`, `deploy`, `rollout`, `rollback`
