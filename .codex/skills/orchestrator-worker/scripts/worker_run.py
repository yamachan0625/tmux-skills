#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
COMMON_DIR = REPO_ROOT / ".codex/orchestrator"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from orchestrator_common import (  # noqa: E402
    OrchestratorError,
    as_bool,
    as_int,
    as_str,
    ensure_command as ensure_command_or_fail,
    extract_first_url,
    extract_pr_number_from_url,
    load_toml,
    run_cmd,
    run_cmd_text,
    safe_key,
    section,
)


class WorkerError(RuntimeError):
    def __init__(self, message: str, *, reason: str, retriable: bool, exit_code: int) -> None:
        super().__init__(message)
        self.reason = reason
        self.retriable = retriable
        self.exit_code = exit_code


def require_command(cmd: str) -> None:
    try:
        ensure_command_or_fail(cmd)
    except OrchestratorError as exc:
        raise WorkerError(
            f"required command not found: {cmd}",
            reason="worker_missing_command",
            retriable=False,
            exit_code=10,
        ) from exc


def build_language_instruction(lang: str) -> str:
    norm = lang.strip().lower()
    if norm in {"ja", "jp", "japanese", "jpn", "日本語"}:
        return "回答・説明・コメントは日本語で記述してください。コード識別子は必要に応じて英語で構いません。"
    if norm in {"en", "english"}:
        return "Write all narrative text, explanations, and comments in English. Keep code identifiers as appropriate."
    return f"Write all narrative text, explanations, and comments in {lang}. Keep code identifiers as appropriate."


def extract_pr_url_from_file(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"https://github\.com/\S+/\S+/pull/\d+", text)
    return m.group(0) if m else ""


def is_transient_error(stderr_file: Path) -> bool:
    if not stderr_file.exists():
        return False
    text = stderr_file.read_text(encoding="utf-8", errors="replace").lower()
    pattern = r"rate.?limit|timeout|temporar|try again|connection reset|503|502|504|network|econnreset|etimedout"
    return re.search(pattern, text) is not None


def decode_base64(raw: str) -> str:
    try:
        return base64.b64decode(raw).decode("utf-8", errors="replace")
    except Exception:
        return ""


class WorkerRun:
    def __init__(self, args: argparse.Namespace) -> None:
        self.repo_root = REPO_ROOT

        self.config_path = Path(args.config)
        if not self.config_path.is_absolute():
            self.config_path = self.repo_root / self.config_path
        if not self.config_path.exists():
            raise WorkerError(
                f"config not found: {self.config_path}",
                reason="worker_config_missing",
                retriable=False,
                exit_code=2,
            )

        require_command("codex")

        try:
            self.cfg = load_toml(self.config_path)
        except Exception as exc:
            raise WorkerError(
                f"failed to load config: {self.config_path}",
                reason="worker_config_invalid",
                retriable=False,
                exit_code=2,
            ) from exc
        core = section(self.cfg, "core")
        self.mode = args.mode or as_str(core.get("mode"), "local")
        if self.mode not in {"local", "github"}:
            raise WorkerError(
                f"unsupported mode: {self.mode}",
                reason="worker_invalid_mode",
                retriable=False,
                exit_code=2,
            )

        self.run_id = args.run_id
        self.task_key = args.task_key
        self.attempt = args.attempt
        self.worker_pane = args.worker_pane
        self.manager_pane = args.manager_pane or ""
        self.task_file = Path(args.task_file) if args.task_file else None
        self.issue_number = str(args.issue_number) if args.issue_number is not None else ""

        if not self.run_id or not self.task_key or self.attempt <= 0 or not self.worker_pane:
            raise WorkerError(
                "missing required args",
                reason="worker_missing_args",
                retriable=False,
                exit_code=2,
            )

        codex_worker = section(self.cfg, "codex.worker")
        self.approval = as_str(codex_worker.get("ask_for_approval"), "never")
        self.sandbox = as_str(codex_worker.get("sandbox"), "danger-full-access")
        self.profile = as_str(codex_worker.get("profile"), "")
        self.model = as_str(codex_worker.get("model"), "")
        self.json_output = as_bool(codex_worker.get("json_output"), True)

        self.output_language = as_str(core.get("output_language"), "ja")
        self.output_language_instruction = as_str(core.get("output_language_instruction"), "")
        if not self.output_language_instruction:
            self.output_language_instruction = build_language_instruction(self.output_language)

        github = section(self.cfg, "github")
        self.github_repo = as_str(github.get("repo"), "")
        self.github_workflow = as_str(github.get("workflow"), "issue-only")
        self.comment_post_max_attempts = as_int(github.get("comment_post_max_attempts"), 2)
        if self.comment_post_max_attempts <= 0:
            self.comment_post_max_attempts = 1
        self.verify_comment_after_post = as_bool(github.get("verify_comment_after_post"), True)

        self.run_dir = self.repo_root / ".agent/runs" / self.run_id
        (self.run_dir / "results").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "events").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "status").mkdir(parents=True, exist_ok=True)

        self.safe_task_key = safe_key(self.task_key)
        self.final_file = self.run_dir / "results" / f"{self.safe_task_key}.attempt{self.attempt}.md"
        self.events_file = self.run_dir / "events" / f"{self.safe_task_key}.attempt{self.attempt}.jsonl"
        self.stderr_file = self.run_dir / "logs" / f"{self.safe_task_key}.attempt{self.attempt}.stderr.log"
        self.status_file = self.run_dir / "status" / f"{self.safe_task_key}.attempt{self.attempt}.env"
        self.comment_error_file = self.run_dir / "logs" / f"{self.safe_task_key}.attempt{self.attempt}.comment-error.log"

        self.state = "failed"
        self.exit_code = 0
        self.retriable = True
        self.failure_reason = "none"
        self.pr_url = ""
        self.pr_number = ""
        self.comment_url = ""
        self.comment_token = ""
        self.comment_verified = False
        self.sot_result_file = ""

    def gh_text(self, args: list[str], *, check: bool = True, default: str = "") -> str:
        try:
            return run_cmd_text(["gh", *args], check=check, default=default)
        except OrchestratorError as exc:
            if check:
                raise WorkerError(
                    f"gh command failed: {' '.join(args)}",
                    reason="worker_command_failed",
                    retriable=True,
                    exit_code=11,
                ) from exc
            return default

    def send_manager_notice(self) -> None:
        if not self.manager_pane or shutil.which("tmux") is None:
            return
        notice = (
            "echo '__ORCH_DONE__ "
            f"run={self.run_id} task={self.safe_task_key} attempt={self.attempt} "
            f"worker={self.worker_pane} state={self.state} exit={self.exit_code} reason={self.failure_reason}'"
        )
        run_cmd(["tmux", "send-keys", "-t", self.manager_pane, "-l", "--", notice], check=False)
        run_cmd(["tmux", "send-keys", "-t", self.manager_pane, "Enter"], check=False)

    def find_comment_url_by_token(self, issue_number: str, token: str) -> str:
        rows = self.gh_text(
            [
                "issue",
                "view",
                "-R",
                self.github_repo,
                issue_number,
                "--json",
                "comments",
                "--jq",
                ".comments[]? | [.url, (.body|@base64)] | @tsv",
            ],
            check=False,
            default="",
        )
        if not rows:
            return ""
        for line in rows.splitlines():
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            url, body_b64 = parts
            body = decode_base64(body_b64)
            if token in body:
                return url
        return ""

    def collect_manager_feedback(self, issue_number: str, limit: int = 3) -> str:
        rows = self.gh_text(
            [
                "issue",
                "view",
                "-R",
                self.github_repo,
                issue_number,
                "--json",
                "comments",
                "--jq",
                ".comments[]? | [.createdAt, (.body|@base64)] | @tsv",
            ],
            check=False,
            default="",
        )
        if not rows:
            return ""
        found: list[str] = []
        for line in rows.splitlines():
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            body = decode_base64(parts[1])
            if "<!-- manager-feedback:v1 -->" in body:
                found.append(body)
        if not found:
            return ""
        return "\n\n".join(found[-limit:])

    def find_open_pr_for_issue(self, issue_number: str) -> str:
        return self.gh_text(
            [
                "pr",
                "list",
                "-R",
                self.github_repo,
                "--state",
                "open",
                "--search",
                f"#{issue_number}",
                "--limit",
                "20",
                "--json",
                "url",
                "--jq",
                ".[0].url // \"\"",
            ],
            check=False,
            default="",
        )

    def write_status_atomic(self) -> None:
        tmp = self.status_file.with_suffix(self.status_file.suffix + f".tmp.{os.getpid()}")
        lines = [
            "status_version=1",
            f"state={self.state}",
            f"exit_code={self.exit_code}",
            f"retriable={'true' if self.retriable else 'false'}",
            f"failure_reason={self.failure_reason}",
            f"worker_pane={self.worker_pane}",
            f"task_key={self.task_key}",
            f"attempt={self.attempt}",
            f"mode={self.mode}",
            f"result_file={self.final_file}",
            f"sot_result_file={self.sot_result_file}",
            f"events_file={self.events_file}",
            f"stderr_file={self.stderr_file}",
            f"task_file={self.task_file or ''}",
            f"issue_number={self.issue_number}",
            f"github_workflow={self.github_workflow}",
            f"pr_url={self.pr_url}",
            f"pr_number={self.pr_number}",
            f"comment_url={self.comment_url}",
            f"comment_token={self.comment_token}",
            f"comment_verified={'true' if self.comment_verified else 'false'}",
            f"comment_error_file={self.comment_error_file}",
        ]
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(self.status_file)

    def build_prompt(self) -> str:
        if self.mode == "local":
            if not self.task_file or not self.task_file.exists():
                raise WorkerError(
                    "local mode requires --task-file and the file must exist",
                    reason="worker_task_file_missing",
                    retriable=False,
                    exit_code=2,
                )
            local_result_dir = as_str(section(self.cfg, "local").get("result_dir"), ".agent/results")
            (self.repo_root / local_result_dir).mkdir(parents=True, exist_ok=True)
            prompt = self.task_file.read_text(encoding="utf-8", errors="replace")
            return prompt

        require_command("gh")
        if not self.issue_number:
            raise WorkerError(
                "github mode requires --issue-number",
                reason="worker_issue_missing",
                retriable=False,
                exit_code=2,
            )
        if not self.github_repo:
            raise WorkerError(
                "[github].repo is required in config",
                reason="worker_repo_missing",
                retriable=False,
                exit_code=2,
            )
        issue_title = self.gh_text(
            ["issue", "view", "-R", self.github_repo, self.issue_number, "--json", "title", "--jq", ".title"]
        )
        issue_body = self.gh_text(
            ["issue", "view", "-R", self.github_repo, self.issue_number, "--json", "body", "--jq", ".body"]
        )
        issue_url = self.gh_text(
            ["issue", "view", "-R", self.github_repo, self.issue_number, "--json", "url", "--jq", ".url"]
        )
        feedback = self.collect_manager_feedback(self.issue_number, 3)
        prompt = f"Issue #{self.issue_number}: {issue_title}\n\n{issue_body}\n\nIssue URL: {issue_url}"
        if feedback:
            prompt += f"\n\n[Recent manager feedback]\n{feedback}"
        if self.github_workflow == "pr-loop":
            branch_name = f"agent/issue-{self.issue_number}-attempt-{self.attempt}"
            prompt += (
                "\n\n[GitHub PR workflow requirement]\n"
                "1. この Issue の要件を満たす実装を行う。\n"
                f"2. ブランチ '{branch_name}' を作成または更新して作業する。\n"
                "3. 変更を commit し、remote へ push する。\n"
                f"4. この Issue と紐づく PR を作成または更新する（本文に 'Closes #{self.issue_number}' を含める）。\n"
                "5. 最終回答に必ず 'PR_URL: <url>' の1行を含める。\n"
                "6. テストや静的チェックがある場合は実行し、結果を要約する。"
            )
        return prompt

    def run_codex(self, prompt: str) -> None:
        prompt = f"{prompt}\n\n[Output language requirement]\n{self.output_language_instruction}\n"
        cmd = ["codex", "-a", self.approval, "-s", self.sandbox, "exec"]
        if self.profile:
            cmd.extend(["--profile", self.profile])
        if self.model:
            cmd.extend(["--model", self.model])
        if self.json_output:
            cmd.append("--json")
        cmd.extend(["-o", str(self.final_file), "-"])

        with self.events_file.open("w", encoding="utf-8") as out, self.stderr_file.open(
            "w", encoding="utf-8"
        ) as err:
            cp = subprocess.run(cmd, input=prompt, text=True, stdout=out, stderr=err, check=False)

        self.exit_code = cp.returncode
        if self.exit_code == 0:
            self.state = "done"
            self.retriable = False
            self.failure_reason = "none"
        else:
            self.state = "failed"
            self.failure_reason = "codex_exec_failed"
            self.retriable = is_transient_error(self.stderr_file)

    def process_pr_workflow(self) -> None:
        if not (self.mode == "github" and self.github_workflow == "pr-loop" and self.state == "done"):
            return
        self.pr_url = extract_pr_url_from_file(self.final_file)
        if not self.pr_url:
            self.pr_url = self.find_open_pr_for_issue(self.issue_number)
        if self.pr_url:
            self.pr_number = extract_pr_number_from_url(self.pr_url)
        if not self.pr_url:
            self.state = "failed"
            self.retriable = True
            self.failure_reason = "missing_pr_url"
            if self.exit_code == 0:
                self.exit_code = 87

    def copy_local_result(self) -> None:
        if self.mode != "local" or not self.final_file.exists():
            return
        local_result_dir = as_str(section(self.cfg, "local").get("result_dir"), ".agent/results")
        target = self.repo_root / local_result_dir / f"{self.safe_task_key}.attempt{self.attempt}.md"
        tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.final_file, tmp)
        tmp.replace(target)
        self.sot_result_file = str(target)

    def post_github_comment(self) -> None:
        if self.mode != "github":
            return
        marker = "<!-- agent-result:v1 -->"
        self.comment_token = f"comment_token: {self.run_id}:{self.safe_task_key}:{self.attempt}"
        comment_file = self.run_dir / "logs" / f"{self.safe_task_key}.attempt{self.attempt}.comment.md"

        lines = [
            marker,
            self.comment_token,
            f"run_id: {self.run_id}",
            f"task_key: {self.task_key}",
            f"attempt: {self.attempt}",
            f"worker_pane: {self.worker_pane}",
            f"status: {self.state}",
            f"exit_code: {self.exit_code}",
            f"failure_reason: {self.failure_reason}",
            f"workflow: {self.github_workflow}",
            f"pr_url: {self.pr_url}",
            f"pr_number: {self.pr_number}",
            "",
        ]
        if self.state == "done" and self.final_file.exists():
            lines.extend(["### Worker Result", "", self.final_file.read_text(encoding="utf-8", errors="replace")])
        else:
            lines.extend(["### Worker Error (stderr tail)", ""])
            if self.stderr_file.exists():
                tail_lines = self.stderr_file.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
                lines.extend(tail_lines)
            else:
                lines.append("stderr file not found")
        comment_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        self.comment_url = self.find_comment_url_by_token(self.issue_number, self.comment_token)
        if not self.comment_url:
            for post_attempt in range(1, self.comment_post_max_attempts + 1):
                cp = run_cmd(
                    [
                        "gh",
                        "issue",
                        "comment",
                        "-R",
                        self.github_repo,
                        self.issue_number,
                        "--body-file",
                        str(comment_file),
                    ],
                    check=False,
                    capture_output=True,
                )
                output = (cp.stdout or "") + (cp.stderr or "")
                if cp.returncode == 0:
                    parsed_url = extract_first_url(output)
                    if self.verify_comment_after_post:
                        verified_url = self.find_comment_url_by_token(self.issue_number, self.comment_token)
                        if verified_url:
                            self.comment_verified = True
                            self.comment_url = verified_url or parsed_url
                            break
                        self.comment_error_file.write_text(
                            f"comment verification failed on attempt {post_attempt}\n{output}",
                            encoding="utf-8",
                        )
                    else:
                        self.comment_verified = True
                        self.comment_url = parsed_url
                        break
                else:
                    self.comment_error_file.write_text(output, encoding="utf-8")
                if post_attempt < self.comment_post_max_attempts:
                    time.sleep((post_attempt + 1) * 2)
        else:
            self.comment_verified = True

        if not self.comment_verified:
            self.state = "failed"
            self.retriable = True
            if self.failure_reason == "none":
                self.failure_reason = "github_comment_verification_failed"
            else:
                self.failure_reason = f"{self.failure_reason}+github_comment_verification_failed"
            if self.exit_code == 0:
                self.exit_code = 86

    def execute(self) -> int:
        try:
            prompt = self.build_prompt()
            self.run_codex(prompt)
            self.process_pr_workflow()
            self.copy_local_result()
            self.post_github_comment()
            self.write_status_atomic()
            self.send_manager_notice()
            return self.exit_code
        except WorkerError as exc:
            if self.exit_code == 0:
                self.exit_code = exc.exit_code
            self.state = "failed"
            self.retriable = exc.retriable
            self.failure_reason = exc.reason
            self.write_status_atomic()
            self.send_manager_notice()
            return self.exit_code
        except Exception:
            if self.exit_code == 0:
                self.exit_code = 1
            self.state = "failed"
            self.retriable = True
            if not self.failure_reason or self.failure_reason == "none":
                self.failure_reason = "worker_internal_error"
            self.write_status_atomic()
            self.send_manager_notice()
            return self.exit_code


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single orchestrator task as worker.")
    parser.add_argument("--run-id", required=True, help="run id")
    parser.add_argument("--task-key", required=True, help="task key")
    parser.add_argument("--attempt", type=int, required=True, help="attempt number")
    parser.add_argument("--worker-pane", required=True, help="worker pane id")
    parser.add_argument("--config", default=".codex/orchestrator/config.toml", help="config path")
    parser.add_argument("--mode", choices=["local", "github"], default=None, help="override mode")
    parser.add_argument("--manager-pane", default=None, help="manager pane id")
    parser.add_argument("--task-file", default=None, help="local mode task markdown file")
    parser.add_argument("--issue-number", type=int, default=None, help="github mode issue number")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    return WorkerRun(parse_args(argv)).execute()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
