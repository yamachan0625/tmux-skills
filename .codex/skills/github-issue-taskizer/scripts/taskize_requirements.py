#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
COMMON_DIR = REPO_ROOT / ".codex/orchestrator"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from orchestrator_common import (  # noqa: E402
    OrchestratorError,
    as_str,
    ensure_command,
    extract_first_url,
    load_toml,
    run_cmd,
    section,
)

HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
SPRINT_DAYS_PATTERN = re.compile(r"(?mi)^\s*sprint_days\s*:\s*(\d{1,2})\s*$")
REQUIRED_HEADING_GROUPS: tuple[tuple[str, ...], ...] = (
    ("Sprint Goal", "スプリントゴール"),
    ("Business Value", "ビジネス価値"),
    ("Review Steps", "レビュー手順"),
    ("Done Criteria", "完了条件"),
)


def normalize_heading(value: str) -> str:
    return " ".join(value.strip().lower().split())


def extract_markdown_headings(body: str) -> set[str]:
    found: set[str] = set()
    for raw in body.splitlines():
        match = HEADING_PATTERN.match(raw)
        if not match:
            continue
        found.add(normalize_heading(match.group(1)))
    return found


def validate_root_issue_body(body: str, *, max_sprint_days: int, source: str) -> int:
    heading_set = extract_markdown_headings(body)
    missing: list[str] = []
    for group in REQUIRED_HEADING_GROUPS:
        if not any(normalize_heading(alias) in heading_set for alias in group):
            missing.append(" / ".join(group))

    sprint_days = -1
    match = SPRINT_DAYS_PATTERN.search(body)
    if not match:
        missing.append(f"sprint_days: <1-{max_sprint_days}>")
    else:
        sprint_days = int(match.group(1))
        if sprint_days < 1 or sprint_days > max_sprint_days:
            missing.append(f"sprint_days must be between 1 and {max_sprint_days}")

    if missing:
        details = "\n".join(f"- {item}" for item in missing)
        raise OrchestratorError(
            f"root issue policy violation ({source})\n"
            "required fields are missing or invalid:\n"
            f"{details}"
        )
    return sprint_days


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create root issue and DAG task issues from complex requirements.",
    )
    parser.add_argument(
        "--config",
        default=".codex/orchestrator/config.toml",
        help="path to orchestrator config TOML",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--root-issue", type=int, help="existing root issue number")
    source_group.add_argument("--title", default="", help="title for a newly created root issue")
    parser.add_argument(
        "--body-file",
        default="",
        help="root issue body markdown file (required when --title is used)",
    )
    parser.add_argument(
        "--extra-root-label",
        action="append",
        default=[],
        help="extra labels to add to a newly created root issue (repeatable)",
    )
    parser.add_argument("--run-id", default=None, help="optional DAG run id")
    parser.add_argument("--plan-file", default=None, help="optional prebuilt DAG plan JSON file")
    parser.add_argument(
        "--max-sprint-days",
        type=int,
        default=7,
        help="max allowed sprint_days in root issue (default: 7)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print actions only")

    args = parser.parse_args(argv)
    if args.max_sprint_days <= 0:
        parser.error("--max-sprint-days must be >= 1")
    if args.root_issue is not None:
        if args.root_issue <= 0:
            parser.error("--root-issue must be >= 1")
        if args.body_file:
            parser.error("--body-file cannot be used with --root-issue")

    if args.title:
        if not args.title.strip():
            parser.error("--title cannot be empty")
        if not args.body_file:
            parser.error("--body-file is required when --title is used")
    return args


def dedupe_labels(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        label = raw.strip()
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


class Taskizer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo_root = REPO_ROOT

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = self.repo_root / config_path
        self.config_path = config_path.resolve()
        if not self.config_path.exists():
            raise OrchestratorError(f"config not found: {self.config_path}")

        ensure_command("gh")
        ensure_command("codex")

        self.cfg = load_toml(self.config_path)
        core = section(self.cfg, "core")
        if as_str(core.get("mode"), "github") != "github":
            raise OrchestratorError("taskize_requirements.py requires [core].mode = github")

        github = section(self.cfg, "github")
        self.repo = as_str(github.get("repo"), "")
        if not self.repo:
            raise OrchestratorError("[github].repo is required")

        self.root_label = as_str(github.get("root_label"), "agent:root")
        self.dag_label = as_str(github.get("dag_label"), "agent:dag")
        self.scope_label = as_str(github.get("scope_label"), "")
        self.max_sprint_days = args.max_sprint_days

        self.pipeline_script = (
            self.repo_root / ".codex/skills/orchestrator-manager/scripts/github_dag_pipeline.py"
        )
        if not self.pipeline_script.exists():
            raise OrchestratorError(f"pipeline script not found: {self.pipeline_script}")

    def resolve_path(self, raw: str) -> Path:
        path = Path(raw)
        if path.is_absolute():
            return path
        cwd_path = (Path.cwd() / path).resolve()
        if cwd_path.exists():
            return cwd_path
        return (self.repo_root / path).resolve()

    def create_root_issue(self) -> tuple[int, str]:
        if self.args.root_issue is not None:
            payload = self.root_issue_payload(self.args.root_issue)
            body = as_str(payload.get("body"), "")
            url = as_str(payload.get("url"), "")
            validate_root_issue_body(
                body,
                max_sprint_days=self.max_sprint_days,
                source=f"existing root issue #{self.args.root_issue}",
            )
            return self.args.root_issue, url

        body_file = self.resolve_path(self.args.body_file)
        if not body_file.exists():
            raise OrchestratorError(f"body file not found: {body_file}")
        body_text = body_file.read_text(encoding="utf-8", errors="replace")
        validate_root_issue_body(
            body_text,
            max_sprint_days=self.max_sprint_days,
            source=f"body-file {body_file}",
        )

        labels = dedupe_labels(
            [
                self.root_label,
                self.dag_label,
                self.scope_label,
                *self.args.extra_root_label,
            ]
        )
        cmd = [
            "gh",
            "issue",
            "create",
            "-R",
            self.repo,
            "--title",
            self.args.title.strip(),
            "--body-file",
            str(body_file),
        ]
        for label in labels:
            cmd.extend(["--label", label])

        if self.args.dry_run:
            print("dry-run:create-root:", " ".join(cmd))
            return 0, ""

        cp = run_cmd(cmd, check=False, capture_output=True)
        if cp.returncode != 0:
            raise OrchestratorError(f"gh issue create failed: {(cp.stderr or '').strip()}")

        created_text = (cp.stdout or "") + "\n" + (cp.stderr or "")
        url = extract_first_url(created_text)
        match = re.search(r"/issues/(\d+)", url)
        if not match:
            match = re.search(r"/issues/(\d+)", created_text)
        if not match:
            raise OrchestratorError("failed to parse root issue number from gh output")
        return int(match.group(1)), url

    def root_issue_payload(self, issue_number: int) -> dict[str, object]:
        cp = run_cmd(
            [
                "gh",
                "issue",
                "view",
                "-R",
                self.repo,
                str(issue_number),
                "--json",
                "body,url",
            ],
            check=False,
            capture_output=True,
        )
        if cp.returncode != 0:
            raise OrchestratorError(f"failed to fetch root issue #{issue_number}: {(cp.stderr or '').strip()}")
        try:
            payload = json.loads(cp.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise OrchestratorError(f"failed to parse root issue payload for #{issue_number}") from exc
        if not isinstance(payload, dict):
            raise OrchestratorError(f"unexpected root issue payload for #{issue_number}")
        return payload

    def run_dag_creation(self, root_issue: int) -> int:
        cmd = [
            sys.executable,
            str(self.pipeline_script),
            "--config",
            str(self.config_path),
            "--root-issue",
            str(root_issue),
            "--create-only",
        ]
        if self.args.run_id:
            cmd.extend(["--run-id", self.args.run_id])
        if self.args.plan_file:
            plan_file = self.resolve_path(self.args.plan_file)
            if not plan_file.exists():
                raise OrchestratorError(f"plan file not found: {plan_file}")
            cmd.extend(["--plan-file", str(plan_file)])

        if self.args.dry_run:
            print("dry-run:create-dag:", " ".join(cmd))
            return 0

        cp = run_cmd(cmd, check=False, capture_output=True)
        if cp.returncode != 0:
            detail = ((cp.stdout or "") + "\n" + (cp.stderr or "")).strip()
            raise OrchestratorError(f"github_dag_pipeline.py failed (rc={cp.returncode})\n{detail}")
        return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    runner = Taskizer(args)
    root_issue, root_url = runner.create_root_issue()
    if root_issue <= 0 and args.dry_run:
        print("dry-run:root-issue will be resolved at execution time")
        return 0
    runner.run_dag_creation(root_issue)
    print(f"repo={runner.repo}")
    print(f"root_issue={root_issue}")
    if root_url:
        print(f"root_url={root_url}")
    print("status=dag_issues_created")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except OrchestratorError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
