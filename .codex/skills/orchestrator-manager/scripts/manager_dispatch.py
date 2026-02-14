#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
COMMON_DIR = REPO_ROOT / ".codex/orchestrator"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from orchestrator_common import (  # noqa: E402
    OrchestratorError,
    acquire_lock,
    as_bool,
    as_int,
    as_list_of_int,
    as_list_of_str,
    as_str,
    ensure_command,
    extract_first_url,
    extract_pr_number_from_url,
    file_mtime_epoch,
    has_state_in_env,
    load_toml,
    log,
    log_event,
    now_epoch,
    parse_env_file,
    run_cmd,
    run_cmd_text,
    safe_key,
    section,
    send_tmux_command,
    tmux_interrupt_pane,
)


@dataclass
class Task:
    key: str
    source: str
    kind: str
    issue: str
    attempt: int
    ready_epoch: int
    safe_key: str
    state: str = "pending"
    worker: str = ""
    pr_url: str = ""
    pr_number: str = ""
    review_cycles: int = 0


@dataclass
class WorkerSlot:
    pane: str
    task_index: int | None = None
    started_epoch: int = 0
    last_progress_epoch: int = 0
    disabled: bool = False
    disabled_reason: str = ""


@dataclass
class PRReviewResult:
    outcome: str
    reason: str
    ref: str
    details: str
    pr_url: str
    pr_number: str


class ManagerDispatch:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo_root = REPO_ROOT
        self.config_path = Path(args.config)
        if not self.config_path.is_absolute():
            self.config_path = self.repo_root / self.config_path
        if not self.config_path.exists():
            raise OrchestratorError(
                f"config not found: {self.config_path}\n"
                "copy .codex/orchestrator/config.toml.example to .codex/orchestrator/config.toml first"
            )

        ensure_command("tmux")
        ensure_command("codex")

        self.cfg = load_toml(self.config_path)
        core = section(self.cfg, "core")

        self.mode = as_str(core.get("mode"), "local")
        if args.mode:
            self.mode = args.mode
        if self.mode not in {"local", "github"}:
            raise OrchestratorError(f"unsupported mode: {self.mode}")

        self.run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
        self.limit = args.limit

        self.manager_pane = as_str(core.get("manager_pane"), "%0")
        self.poll_interval_sec = as_int(core.get("poll_interval_sec"), 2)
        self.max_attempts = as_int(core.get("max_attempts"), 3)
        self.worker_hard_timeout_sec = as_int(core.get("worker_hard_timeout_sec"), 3600)
        self.worker_idle_timeout_sec = as_int(core.get("worker_idle_timeout_sec"), 600)
        self.run_deadline_sec = as_int(core.get("run_deadline_sec"), 14400)
        self.timeout_retriable = as_bool(core.get("timeout_retriable"), True)
        self.backoff_secs = as_list_of_int(core.get("retry_backoff_sec"), [30, 120])

        worker_py = self.repo_root / ".codex/skills/orchestrator-worker/scripts/worker_run.py"
        if not worker_py.exists() or not worker_py.is_file():
            raise OrchestratorError("worker entrypoint not found: worker_run.py")
        self.worker_cmd_prefix = [sys.executable, str(worker_py)]

        # In-host guard against concurrent managers processing the same queue.
        lock_name = f"manager.{self.mode}.lock"
        self.lock_handle = acquire_lock(self.repo_root / ".agent/locks" / lock_name)

        self.run_dir = self.repo_root / ".agent/runs" / self.run_id
        (self.run_dir / "status").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "results").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "events").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.manager_event_file = self.run_dir / "manager.events.log"
        self.manager_event_file.touch(exist_ok=True)

        self.local_queue_dir = ""
        self.local_done_dir = ""
        self.local_failed_dir = ""

        github = section(self.cfg, "github")
        self.github_repo = as_str(github.get("repo"), "")
        self.github_queue_label = as_str(github.get("queue_label"), "agent:queued")
        self.github_running_label = as_str(github.get("running_label"), "agent:running")
        self.github_review_label = as_str(github.get("review_label"), "agent:review")
        self.github_done_label = as_str(github.get("done_label"), "agent:done")
        self.github_failed_label = as_str(github.get("failed_label"), "agent:failed")
        self.github_needs_human_label = as_str(github.get("needs_human_label"), "agent:needs-human")
        self.github_scope_label = as_str(github.get("scope_label"), "")
        self.github_workflow = as_str(github.get("workflow"), "issue-only")
        self.github_close_on_done = as_bool(github.get("close_on_done"), True)
        if self.github_workflow == "pr-loop":
            self.github_close_on_done = False
        self.github_pr_merge_method = as_str(github.get("pr_merge_method"), "squash")
        if self.github_pr_merge_method not in {"merge", "squash", "rebase"}:
            self.github_pr_merge_method = "squash"
        self.github_pr_delete_branch = as_bool(github.get("pr_delete_branch"), True)
        self.github_pr_require_checks = as_bool(github.get("pr_require_checks"), True)
        self.github_review_retry_backoff_sec = as_int(github.get("review_retry_backoff_sec"), 60)
        self.github_review_wait_max_cycles = as_int(github.get("review_wait_max_cycles"), 20)

        if self.mode == "local":
            local = section(self.cfg, "local")
            self.local_queue_dir = as_str(local.get("queue_dir"), ".agent/tasks")
            self.local_done_dir = as_str(local.get("done_dir"), ".agent/done")
            self.local_failed_dir = as_str(local.get("failed_dir"), ".agent/failed")
            (self.repo_root / self.local_queue_dir).mkdir(parents=True, exist_ok=True)
            (self.repo_root / self.local_done_dir).mkdir(parents=True, exist_ok=True)
            (self.repo_root / self.local_failed_dir).mkdir(parents=True, exist_ok=True)
        else:
            ensure_command("gh")
            if not self.github_repo:
                raise OrchestratorError("[github].repo is required")

        workers = as_list_of_str(core.get("worker_panes"), [])
        if not workers:
            panes = run_cmd_text(["tmux", "list-panes", "-a", "-F", "#{pane_id}"], check=True)
            workers = [x.strip() for x in panes.splitlines() if x.strip() and x.strip() != self.manager_pane]
        if not workers:
            raise OrchestratorError("no worker panes configured")

        self.workers: list[WorkerSlot] = [WorkerSlot(pane=pane) for pane in workers]
        self.tasks: list[Task] = []
        self.summary: dict[str, tuple[str, str]] = {}

        self.run_started_epoch = now_epoch()
        self.run_deadline_epoch = (
            self.run_started_epoch + self.run_deadline_sec if self.run_deadline_sec > 0 else 0
        )

    def gh_text(self, args: list[str], *, check: bool = True, default: str = "") -> str:
        return run_cmd_text(["gh", *args], check=check, default=default)

    def gh_ok(self, args: list[str]) -> bool:
        cp = run_cmd(["gh", *args], check=False, capture_output=True)
        return cp.returncode == 0

    def _gh_edit_add(self, issue: str, label: str) -> None:
        if not label:
            return
        self.gh_ok(["issue", "edit", "-R", self.github_repo, issue, "--add-label", label])

    def _gh_edit_remove(self, issue: str, label: str) -> None:
        if not label:
            return
        self.gh_ok(["issue", "edit", "-R", self.github_repo, issue, "--remove-label", label])

    def github_mark_running(self, issue: str) -> None:
        self._gh_edit_add(issue, self.github_running_label)
        self._gh_edit_remove(issue, self.github_queue_label)
        self._gh_edit_remove(issue, self.github_review_label)
        self._gh_edit_remove(issue, self.github_failed_label)
        self._gh_edit_remove(issue, self.github_needs_human_label)

    def github_mark_queued(self, issue: str) -> None:
        self._gh_edit_add(issue, self.github_queue_label)
        self._gh_edit_remove(issue, self.github_running_label)
        self._gh_edit_remove(issue, self.github_review_label)
        self._gh_edit_remove(issue, self.github_done_label)
        self._gh_edit_remove(issue, self.github_failed_label)
        self._gh_edit_remove(issue, self.github_needs_human_label)

    def github_mark_review(self, issue: str) -> None:
        self._gh_edit_add(issue, self.github_review_label)
        self._gh_edit_remove(issue, self.github_running_label)
        self._gh_edit_remove(issue, self.github_queue_label)
        self._gh_edit_remove(issue, self.github_failed_label)
        self._gh_edit_remove(issue, self.github_needs_human_label)

    def github_mark_done(self, issue: str) -> None:
        self._gh_edit_add(issue, self.github_done_label)
        self._gh_edit_remove(issue, self.github_running_label)
        self._gh_edit_remove(issue, self.github_review_label)
        self._gh_edit_remove(issue, self.github_failed_label)
        self._gh_edit_remove(issue, self.github_needs_human_label)
        if self.github_close_on_done:
            self.gh_ok(["issue", "close", "-R", self.github_repo, issue, "--reason", "completed"])

    def github_mark_failed(self, issue: str) -> None:
        self._gh_edit_add(issue, self.github_failed_label)
        self._gh_edit_add(issue, self.github_needs_human_label)
        self._gh_edit_remove(issue, self.github_running_label)
        self._gh_edit_remove(issue, self.github_review_label)
        self._gh_edit_remove(issue, self.github_queue_label)
        self._gh_edit_remove(issue, self.github_done_label)

    def add_task(
        self,
        *,
        key: str,
        source: str,
        kind: str,
        issue: str,
        attempt: int,
        ready: int,
        pr_url: str = "",
        pr_number: str = "",
        review_cycles: int = 0,
    ) -> None:
        self.tasks.append(
            Task(
                key=key,
                source=source,
                kind=kind,
                issue=issue,
                attempt=attempt,
                ready_epoch=ready,
                safe_key=safe_key(key),
                pr_url=pr_url,
                pr_number=pr_number,
                review_cycles=review_cycles,
            )
        )

    def summary_set(self, key: str, state: str, ref: str) -> None:
        self.summary[key] = (state, ref)

    def backoff_for_attempt(self, failed_attempt: int) -> int:
        idx = max(0, failed_attempt - 1)
        if idx >= len(self.backoff_secs):
            idx = len(self.backoff_secs) - 1
        return self.backoff_secs[idx]

    def clear_worker_slot(self, wi: int | None) -> None:
        if wi is None or wi < 0:
            return
        slot = self.workers[wi]
        slot.task_index = None
        slot.started_epoch = 0
        slot.last_progress_epoch = 0

    def disable_worker_slot(self, wi: int, reason: str) -> None:
        slot = self.workers[wi]
        slot.disabled = True
        slot.disabled_reason = reason
        log(f"worker disabled pane={slot.pane} reason={reason}")
        log_event(self.manager_event_file, "warn", "worker_disabled", f"pane={slot.pane} reason={reason}")

    def finalize_task_done(self, ti: int, wi: int | None, ref: str) -> None:
        task = self.tasks[ti]
        task.state = "done"
        task.worker = ""
        worker_desc = "manager-review"
        if wi is not None and wi >= 0:
            worker_desc = self.workers[wi].pane
            self.clear_worker_slot(wi)

        if self.mode == "local":
            src = Path(task.source)
            if src.exists():
                shutil.move(str(src), str(self.repo_root / self.local_done_dir / src.name))
        else:
            self.github_mark_done(task.issue)

        self.summary_set(task.key, "done", ref)
        log(f"done task={task.key} attempt={task.attempt} worker={worker_desc}")
        log_event(
            self.manager_event_file,
            "info",
            "task_done",
            f"task={task.key} attempt={task.attempt} worker={worker_desc} ref={ref}",
        )

    def finalize_task_failed(self, ti: int, wi: int | None, retriable: bool, reason: str, ref: str) -> None:
        task = self.tasks[ti]
        task.worker = ""
        if wi is not None and wi >= 0:
            self.clear_worker_slot(wi)

        if retriable and task.attempt < self.max_attempts:
            next_kind = "github" if task.kind == "github-review" else task.kind
            next_attempt = task.attempt + 1
            backoff = self.backoff_for_attempt(task.attempt)
            ready_at = now_epoch() + backoff
            self.add_task(
                key=task.key,
                source=task.source,
                kind=next_kind,
                issue=task.issue,
                attempt=next_attempt,
                ready=ready_at,
            )
            if self.mode == "github":
                self.github_mark_queued(task.issue)
            task.state = "failed-retrying"
            log(f"retry scheduled task={task.key} next_attempt={next_attempt} backoff={backoff}s reason={reason}")
            log_event(
                self.manager_event_file,
                "warn",
                "task_retry",
                f"task={task.key} current_attempt={task.attempt} next_attempt={next_attempt} backoff={backoff} reason={reason}",
            )
            return

        task.state = "failed"
        if self.mode == "local":
            src = Path(task.source)
            if src.exists():
                shutil.move(str(src), str(self.repo_root / self.local_failed_dir / src.name))
        else:
            self.github_mark_failed(task.issue)

        fail_ref = f"reason={reason}"
        if ref:
            fail_ref = f"{fail_ref} ref={ref}"
        self.summary_set(task.key, "failed", fail_ref)

        if wi is not None and wi >= 0:
            worker_desc = self.workers[wi].pane
            log(f"failed task={task.key} attempt={task.attempt} worker={worker_desc} reason={reason}")
            log_event(
                self.manager_event_file,
                "error",
                "task_failed",
                f"task={task.key} attempt={task.attempt} worker={worker_desc} reason={reason}",
            )
        else:
            log(f"failed task={task.key} attempt={task.attempt} reason={reason}")
            log_event(
                self.manager_event_file,
                "error",
                "task_failed",
                f"task={task.key} attempt={task.attempt} reason={reason}",
            )

    def github_post_manager_feedback(
        self, issue: str, task_key: str, attempt: int, reason: str, pr_url: str, pr_number: str, details: str
    ) -> str:
        safe = safe_key(task_key)
        feedback_file = self.run_dir / "logs" / f"{safe}.attempt{attempt}.manager-feedback.md"
        feedback_token = f"manager_feedback_token: {self.run_id}:{safe}:{attempt}:{now_epoch()}"
        body = "\n".join(
            [
                "<!-- manager-feedback:v1 -->",
                feedback_token,
                f"run_id: {self.run_id}",
                f"task_key: {task_key}",
                f"attempt: {attempt}",
                f"reason: {reason}",
                f"pr_url: {pr_url}",
                f"pr_number: {pr_number}",
                "",
                "修正をお願いします。以下を反映して再度 PR を更新してください。",
                "",
                details or "- manager review failed. Please inspect PR and update.",
                "",
            ]
        )
        feedback_file.write_text(body, encoding="utf-8")

        cp = run_cmd(
            ["gh", "issue", "comment", "-R", self.github_repo, issue, "--body-file", str(feedback_file)],
            check=False,
            capture_output=True,
        )
        if cp.returncode != 0:
            error_file = self.run_dir / "logs" / f"{safe}.attempt{attempt}.manager-feedback-error.log"
            error_file.write_text((cp.stdout or "") + (cp.stderr or ""), encoding="utf-8")
            return ""
        return extract_first_url((cp.stdout or "") + (cp.stderr or ""))

    def github_review_pr(
        self, issue: str, task_key: str, attempt: int, pr_url: str, pr_number: str
    ) -> PRReviewResult:
        result = PRReviewResult(
            outcome="failed",
            reason="pr_review_failed",
            ref=pr_url,
            details="",
            pr_url=pr_url,
            pr_number=pr_number,
        )

        if not result.pr_number and result.pr_url:
            result.pr_number = extract_pr_number_from_url(result.pr_url)
        if not result.pr_number:
            result.outcome = "needs_worker"
            result.reason = "missing_pr_url"
            result.details = "- 最終回答に `PR_URL: https://github.com/<owner>/<repo>/pull/<number>` を必ず含めてください。"
            return result

        pr_state = self.gh_text(
            ["pr", "view", "-R", self.github_repo, result.pr_number, "--json", "state", "--jq", ".state"],
            check=False,
            default="",
        )
        if not pr_state:
            result.outcome = "needs_worker"
            result.reason = "pr_not_found"
            result.details = (
                f"- PR #{result.pr_number} を取得できませんでした。"
                "PR URL の提示とリポジトリ整合性を確認してください。"
            )
            return result

        pr_url_live = self.gh_text(
            ["pr", "view", "-R", self.github_repo, result.pr_number, "--json", "url", "--jq", ".url"],
            check=False,
            default=result.pr_url,
        )
        if pr_url_live:
            result.pr_url = pr_url_live
        result.ref = result.pr_url

        if pr_state == "MERGED":
            result.outcome = "done"
            result.reason = "none"
            return result

        if pr_state != "OPEN":
            result.outcome = "needs_worker"
            result.reason = "pr_not_open"
            result.details = f"- PR #{result.pr_number} が OPEN ではありません（state={pr_state}）。再度有効な PR を作成してください。"
            return result

        is_draft = self.gh_text(
            ["pr", "view", "-R", self.github_repo, result.pr_number, "--json", "isDraft", "--jq", ".isDraft"],
            check=False,
            default="false",
        ).lower()
        if is_draft == "true":
            result.outcome = "needs_worker"
            result.reason = "pr_is_draft"
            result.details = f"- PR #{result.pr_number} が Draft のためマージできません。Ready for review にしてください。"
            return result

        merge_state = self.gh_text(
            [
                "pr",
                "view",
                "-R",
                self.github_repo,
                result.pr_number,
                "--json",
                "mergeStateStatus",
                "--jq",
                ".mergeStateStatus",
            ],
            check=False,
            default="",
        )
        if merge_state in {"DIRTY", "BLOCKED", "BEHIND", "UNKNOWN"}:
            result.outcome = "needs_worker"
            result.reason = f"pr_merge_state_{merge_state.lower()}"
            result.details = (
                f"- PR #{result.pr_number} は現在 mergeStateStatus={merge_state} です。"
                "コンフリクト解消またはブランチ更新を実施してください。"
            )
            return result

        if self.github_pr_require_checks:
            checks_cp = run_cmd(
                ["gh", "pr", "checks", "-R", self.github_repo, result.pr_number, "--required"],
                check=False,
                capture_output=True,
            )
            checks_output = (checks_cp.stdout or "") + (checks_cp.stderr or "")
            checks_output_l = checks_output.lower()
            if checks_cp.returncode == 8:
                result.outcome = "wait"
                result.reason = "pr_checks_pending"
                return result
            if checks_cp.returncode == 1:
                if (
                    "no required checks reported" in checks_output_l
                    or "no required status checks" in checks_output_l
                ):
                    # No required checks are configured for this branch/repo.
                    # Treat as pass and continue to merge.
                    pass
                else:
                    failed_checks = self.gh_text(
                        [
                            "pr",
                            "checks",
                            "-R",
                            self.github_repo,
                            result.pr_number,
                            "--required",
                            "--json",
                            "name,bucket",
                            "--jq",
                            '[.[] | select(.bucket == "fail") | .name] | join(", ")',
                        ],
                        check=False,
                        default="",
                    )
                    result.outcome = "needs_worker"
                    result.reason = "pr_checks_failed"
                    if failed_checks:
                        result.details = f"- 必須チェックが失敗しました: {failed_checks}"
                    else:
                        result.details = (
                            f"- 必須チェックが失敗しました。`gh pr checks {result.pr_number} --required` で詳細を確認して修正してください。"
                        )
                    return result
            elif checks_cp.returncode != 0:
                result.outcome = "failed"
                result.reason = "pr_checks_command_failed"
                result.details = checks_output
                return result

        merge_cmd = ["gh", "pr", "merge", "-R", self.github_repo, result.pr_number]
        if self.github_pr_merge_method == "merge":
            merge_cmd.append("--merge")
        elif self.github_pr_merge_method == "rebase":
            merge_cmd.append("--rebase")
        else:
            merge_cmd.append("--squash")
        if self.github_pr_delete_branch:
            merge_cmd.append("--delete-branch")

        merge_cp = run_cmd(merge_cmd, check=False, capture_output=True)
        merge_output = (merge_cp.stdout or "") + (merge_cp.stderr or "")
        merge_output_l = merge_output.lower()

        if merge_cp.returncode == 0:
            result.outcome = "done"
            result.reason = "none"
            return result
        if "already merged" in merge_output_l or "has been merged" in merge_output_l:
            result.outcome = "done"
            result.reason = "none"
            return result
        if "required status check" in merge_output_l or "pending check" in merge_output_l or "checks are still running" in merge_output_l:
            result.outcome = "wait"
            result.reason = "pr_checks_pending"
            return result
        if re.search(r"not mergeable|conflict|behind|blocked|draft", merge_output_l):
            result.outcome = "needs_worker"
            result.reason = "pr_merge_blocked"
            result.details = (
                "- PR をマージできませんでした。以下のメッセージを確認して修正してください。\n\n"
                "```\n"
                f"{merge_output.strip()}\n"
                "```"
            )
            return result

        result.outcome = "failed"
        result.reason = "pr_merge_failed"
        result.details = merge_output
        return result

    def load_initial_tasks(self) -> None:
        if self.mode == "local":
            queue_dir = self.repo_root / self.local_queue_dir
            loaded = 0
            for file in sorted(queue_dir.glob("*.md")):
                task_key = file.stem
                self.add_task(
                    key=task_key,
                    source=str(file),
                    kind="local",
                    issue="",
                    attempt=1,
                    ready=0,
                )
                loaded += 1
                if loaded >= self.limit:
                    break
            return

        if self.github_scope_label:
            args = [
                "issue",
                "list",
                "-R",
                self.github_repo,
                "-l",
                self.github_queue_label,
                "-l",
                self.github_scope_label,
                "--limit",
                str(self.limit),
                "--json",
                "number",
                "--jq",
                ".[].number",
            ]
        else:
            args = [
                "issue",
                "list",
                "-R",
                self.github_repo,
                "-l",
                self.github_queue_label,
                "--limit",
                str(self.limit),
                "--json",
                "number",
                "--jq",
                ".[].number",
            ]
        out = self.gh_text(args, check=False, default="")
        loaded = 0
        for raw in out.splitlines():
            issue = raw.strip()
            if not issue:
                continue
            self.add_task(
                key=f"issue-{issue}",
                source=issue,
                kind="github",
                issue=issue,
                attempt=1,
                ready=0,
            )
            loaded += 1
            if loaded >= self.limit:
                break

    def run(self) -> int:
        self.load_initial_tasks()
        initial_task_total = len(self.tasks)
        if initial_task_total == 0:
            log(f"no queued tasks found (mode={self.mode})")
            log_event(self.manager_event_file, "info", "no_tasks", f"mode={self.mode}")
            return 0

        log(
            f"run_id={self.run_id} mode={self.mode} workers={len(self.workers)} tasks={initial_task_total}"
        )
        log_event(
            self.manager_event_file,
            "info",
            "run_started",
            f"run_id={self.run_id} mode={self.mode} workers={len(self.workers)} tasks={initial_task_total}",
        )

        while True:
            now = now_epoch()

            if self.run_deadline_epoch > 0 and now >= self.run_deadline_epoch:
                log(f"run deadline exceeded run_id={self.run_id}")
                log_event(self.manager_event_file, "error", "run_deadline_exceeded", f"run_id={self.run_id}")

                for wi, slot in enumerate(self.workers):
                    if slot.task_index is None:
                        continue
                    ti = slot.task_index
                    task = self.tasks[ti]
                    tmux_interrupt_pane(slot.pane)
                    self.disable_worker_slot(wi, "run_deadline_exceeded")
                    stderr = self.run_dir / "logs" / f"{task.safe_key}.attempt{task.attempt}.stderr.log"
                    self.finalize_task_failed(ti, wi, False, "run_deadline_exceeded", str(stderr))

                for ti, task in enumerate(self.tasks):
                    if task.state == "pending":
                        self.finalize_task_failed(ti, None, False, "run_deadline_exceeded", "")
                break

            for wi, slot in enumerate(self.workers):
                if slot.task_index is None:
                    continue
                ti = slot.task_index
                task = self.tasks[ti]

                status_file = self.run_dir / "status" / f"{task.safe_key}.attempt{task.attempt}.env"
                if status_file.exists() and has_state_in_env(status_file):
                    st = parse_env_file(status_file)
                    state = st.get("state", "")
                    retriable = as_bool(st.get("retriable", "false"), False)
                    reason = st.get("failure_reason", "") or "worker_reported_failed"
                    result_file = st.get("result_file", "")
                    sot_result_file = st.get("sot_result_file", "")
                    comment_url = st.get("comment_url", "")
                    pr_url = st.get("pr_url", "")
                    pr_number = st.get("pr_number", "")
                    task.pr_url = pr_url
                    task.pr_number = pr_number

                    ref = result_file
                    if sot_result_file:
                        ref = sot_result_file
                    if self.mode == "github":
                        ref = comment_url

                    if state == "done":
                        if self.mode == "github" and self.github_workflow == "pr-loop":
                            self.github_mark_review(task.issue)
                            review = self.github_review_pr(task.issue, task.key, task.attempt, pr_url, pr_number)
                            task.pr_url = review.pr_url
                            task.pr_number = review.pr_number

                            if review.outcome == "done":
                                self.finalize_task_done(ti, wi, review.ref)
                            elif review.outcome == "wait":
                                self.clear_worker_slot(wi)
                                task.state = "pending"
                                task.kind = "github-review"
                                task.worker = ""
                                task.ready_epoch = now + self.github_review_retry_backoff_sec
                                task.review_cycles += 1
                                if task.review_cycles > self.github_review_wait_max_cycles:
                                    self.finalize_task_failed(ti, None, False, "pr_checks_stuck", review.ref)
                                else:
                                    log(
                                        f"review wait task={task.key} attempt={task.attempt} "
                                        f"issue={task.issue} pr={review.pr_number} cycle={task.review_cycles}"
                                    )
                                    log_event(
                                        self.manager_event_file,
                                        "warn",
                                        "task_review_wait",
                                        f"task={task.key} issue={task.issue} pr={review.pr_number} cycle={task.review_cycles}",
                                    )
                            elif review.outcome == "needs_worker":
                                feedback_ref = self.github_post_manager_feedback(
                                    task.issue,
                                    task.key,
                                    task.attempt,
                                    review.reason,
                                    review.pr_url,
                                    review.pr_number,
                                    review.details,
                                )
                                self.finalize_task_failed(
                                    ti,
                                    wi,
                                    True,
                                    review.reason,
                                    feedback_ref or comment_url,
                                )
                            else:
                                self.finalize_task_failed(ti, wi, False, review.reason, review.ref)
                        else:
                            self.finalize_task_done(ti, wi, ref)
                    elif state == "failed":
                        self.finalize_task_failed(ti, wi, retriable, reason, ref)
                    continue

                events_file = self.run_dir / "events" / f"{task.safe_key}.attempt{task.attempt}.jsonl"
                stderr_file = self.run_dir / "logs" / f"{task.safe_key}.attempt{task.attempt}.stderr.log"
                progress = max(file_mtime_epoch(events_file), file_mtime_epoch(stderr_file))
                if progress > slot.last_progress_epoch:
                    slot.last_progress_epoch = progress

                runtime_sec = now - slot.started_epoch
                idle_sec = now - slot.last_progress_epoch
                timeout_reason = ""
                if self.worker_hard_timeout_sec > 0 and runtime_sec >= self.worker_hard_timeout_sec:
                    timeout_reason = "worker_hard_timeout"
                elif self.worker_idle_timeout_sec > 0 and idle_sec >= self.worker_idle_timeout_sec:
                    timeout_reason = "worker_idle_timeout"

                if timeout_reason:
                    tmux_interrupt_pane(slot.pane)
                    self.disable_worker_slot(wi, timeout_reason)
                    self.finalize_task_failed(
                        ti,
                        wi,
                        self.timeout_retriable,
                        timeout_reason,
                        str(stderr_file),
                    )

            if self.mode == "github" and self.github_workflow == "pr-loop":
                for ti, task in enumerate(self.tasks):
                    if task.state != "pending" or task.kind != "github-review":
                        continue
                    if task.ready_epoch > now:
                        continue

                    self.github_mark_review(task.issue)
                    review = self.github_review_pr(task.issue, task.key, task.attempt, task.pr_url, task.pr_number)
                    task.pr_url = review.pr_url
                    task.pr_number = review.pr_number

                    if review.outcome == "done":
                        self.finalize_task_done(ti, None, review.ref)
                    elif review.outcome == "wait":
                        task.ready_epoch = now + self.github_review_retry_backoff_sec
                        task.review_cycles += 1
                        if task.review_cycles > self.github_review_wait_max_cycles:
                            self.finalize_task_failed(ti, None, False, "pr_checks_stuck", review.ref)
                        else:
                            log(
                                f"review wait task={task.key} attempt={task.attempt} "
                                f"issue={task.issue} pr={review.pr_number} cycle={task.review_cycles}"
                            )
                            log_event(
                                self.manager_event_file,
                                "warn",
                                "task_review_wait",
                                f"task={task.key} issue={task.issue} pr={review.pr_number} cycle={task.review_cycles}",
                            )
                    elif review.outcome == "needs_worker":
                        feedback_ref = self.github_post_manager_feedback(
                            task.issue,
                            task.key,
                            task.attempt,
                            review.reason,
                            review.pr_url,
                            review.pr_number,
                            review.details,
                        )
                        self.finalize_task_failed(ti, None, True, review.reason, feedback_ref or review.ref)
                    else:
                        self.finalize_task_failed(ti, None, False, review.reason, review.ref)

            for wi, slot in enumerate(self.workers):
                if slot.disabled or slot.task_index is not None:
                    continue
                candidate = None
                for ti, task in enumerate(self.tasks):
                    if task.state != "pending":
                        continue
                    if task.kind == "github-review":
                        continue
                    if task.ready_epoch > now:
                        continue
                    candidate = ti
                    break
                if candidate is None:
                    continue

                task = self.tasks[candidate]
                cmd = self.worker_cmd_prefix + [
                    "--config",
                    str(self.config_path),
                    "--run-id",
                    self.run_id,
                    "--mode",
                    self.mode,
                    "--task-key",
                    task.key,
                    "--attempt",
                    str(task.attempt),
                    "--worker-pane",
                    slot.pane,
                    "--manager-pane",
                    self.manager_pane,
                ]
                if self.mode == "local":
                    cmd.extend(["--task-file", task.source])
                else:
                    cmd.extend(["--issue-number", task.issue])
                    self.github_mark_running(task.issue)

                send_tmux_command(slot.pane, cmd)
                task.state = "running"
                task.worker = slot.pane
                slot.task_index = candidate
                slot.started_epoch = now
                slot.last_progress_epoch = now
                log(f"dispatch task={task.key} attempt={task.attempt} pane={slot.pane}")
                log_event(
                    self.manager_event_file,
                    "info",
                    "task_dispatched",
                    f"task={task.key} attempt={task.attempt} pane={slot.pane}",
                )

            pending_count = 0
            pending_worker_count = 0
            running_count = 0
            for task in self.tasks:
                if task.state == "pending":
                    pending_count += 1
                    if task.kind != "github-review":
                        pending_worker_count += 1
                if task.state == "running":
                    running_count += 1
            healthy_workers = sum(1 for slot in self.workers if not slot.disabled)

            if pending_count == 0 and running_count == 0:
                break

            if pending_worker_count > 0 and running_count == 0 and healthy_workers == 0:
                log("no healthy workers left; failing remaining pending tasks")
                log_event(
                    self.manager_event_file,
                    "error",
                    "no_healthy_workers",
                    f"pending_worker={pending_worker_count}",
                )
                for ti, task in enumerate(self.tasks):
                    if task.state == "pending" and task.kind != "github-review":
                        self.finalize_task_failed(ti, None, False, "no_healthy_workers", "")
                if pending_count == pending_worker_count:
                    break

            time.sleep(self.poll_interval_sec)

        summary_file = self.run_dir / "summary.tsv"
        with summary_file.open("w", encoding="utf-8") as f:
            f.write("task_key\tstate\tref\n")
            for key, (state, ref) in self.summary.items():
                f.write(f"{key}\t{state}\t{ref}\n")

        done_count = sum(1 for state, _ in self.summary.values() if state == "done")
        failed_count = len(self.summary) - done_count

        log(
            f"completed run_id={self.run_id} done={done_count} failed={failed_count} total={len(self.summary)}"
        )
        log(f"summary_file={summary_file}")
        log_event(
            self.manager_event_file,
            "info",
            "run_completed",
            f"run_id={self.run_id} done={done_count} failed={failed_count} total={len(self.summary)} summary={summary_file}",
        )
        return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatch tasks to tmux worker panes and monitor execution.",
    )
    parser.add_argument(
        "--config",
        default=".codex/orchestrator/config.toml",
        help="path to config TOML (default: .codex/orchestrator/config.toml)",
    )
    parser.add_argument("--mode", choices=["local", "github"], default=None, help="override [core].mode")
    parser.add_argument("--run-id", default=None, help="fixed run id (default: timestamp)")
    parser.add_argument("--limit", type=int, default=50, help="max number of queued tasks to consume")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    runner = ManagerDispatch(args)
    return runner.run()


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except OrchestratorError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
