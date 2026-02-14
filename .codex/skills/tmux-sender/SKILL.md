---
name: tmux-sender
description: tmux の別ペインにコマンドを送信する。ペイン指定は pane_id(%N) を使い、送信・取得を安全に実行する。
---

# tmux コマンド送信スキル

## 目的

tmux の別ペインへ安全にコマンドを送信し、必要なら出力を取得する。

## ルール

1. ターゲットは `pane_id(%N)` を使う。`pane_index` は使わない。
2. コマンド送信は `scripts/send_command.sh` を使う。`send-keys -l` でリテラル送信する。
3. 実行確認は `scripts/capture_tail.sh` で行う。

## 手順

1. `tmux list-panes -a -F '#S:#I.#P pane_id=#{pane_id} active=#{pane_active}'` で対象ペインを確認
2. `.codex/skills/tmux-sender/scripts/send_command.sh <pane_id> "<command>"` で送信
3. `.codex/skills/tmux-sender/scripts/capture_tail.sh <pane_id> 60` で結果確認

## 例

```bash
.codex/skills/tmux-sender/scripts/send_command.sh %2 'codex exec "レビューして"'
.codex/skills/tmux-sender/scripts/capture_tail.sh %2 80
```
