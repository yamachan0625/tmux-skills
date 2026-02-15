#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
SKILL_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = SKILL_DIR / "assets"
COMMON_DIR = REPO_ROOT / ".codex/orchestrator"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from orchestrator_common import (  # noqa: E402
    OrchestratorError,
    as_int,
    as_str,
    ensure_command,
    extract_first_url,
    load_toml,
    run_cmd,
    safe_key,
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
DEFAULT_REVIEW_PURPOSE = (
    "ビジネスサイドとエンジニアの認識齟齬を定期的に検知し、"
    "プロジェクト方向性の妥当性を確認できる単位でレビューする。"
)
TASK_LINE_PATTERN = re.compile(r"^\s*[-*]\s+(?:\[[ xX]\]\s*)?(?P<task>.+?)\s*$")
JAPANESE_CHAR_PATTERN = re.compile(r"[ぁ-んァ-ン一-龯]")
QUANTITATIVE_PATTERN = re.compile(
    r"(?:\d+(?:\.\d+)?)|(?:>=|<=|>|<|=)|(?:以上|以下|以内|未満)|(?:件|回|秒|分|時間|%|ms)"
)
EVIDENCE_PATH_PATTERN = re.compile(r"(?:/|\\|\.md|\.json|\.sql|\.yml|\.yaml|\.csv|#)")
PHASE_ORDER = (
    "foundation",
    "implementation",
    "validation",
    "operations",
    "release",
    "generic",
)
PHASE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "release": ("リリース", "判定", "受け入れ", "go/no-go", "go no-go", "本番"),
    "operations": ("テスト", "運用", "監視", "アラート", "性能", "再実行", "初動"),
    "validation": ("精度", "評価", "kpi", "閾値", "係数", "プロンプト", "レビュー"),
    "implementation": ("実装", "基盤", "業務", "処理", "バッチ", "連携"),
    "foundation": ("仕様", "設計", "要件", "api", "データ", "認証"),
}
PHASE_LABELS: dict[str, tuple[str, int]] = {
    "foundation": ("初動レビュー: 仕様とデータ前提を揃える", 2),
    "implementation": ("実装レビュー: 実行基盤と主要機能を成立させる", 4),
    "validation": ("品質レビュー: 精度検証サイクルを確立する", 3),
    "operations": ("運用準備レビュー: 総合検証と運用手順を固める", 2),
    "release": ("意思決定レビュー: 受け入れ判定と本番可否を決める", 1),
    "generic": ("レビュー: 未分類タスクを前進させる", 2),
}


@dataclass
class DAGNode:
    node_id: str
    title: str
    description: str
    acceptance_criteria: list[str]
    depends_on: list[str]


@dataclass
class RootPlan:
    root_id: str
    title: str
    sprint_days: int
    sprint_goal: str
    business_value: str
    review_steps: list[str]
    done_criteria: list[str]
    background: str
    scope: list[str]
    non_scope: list[str]
    constraints: list[str]
    risks: list[str]
    nodes: list[DAGNode]


@dataclass
class CrossRootDependency:
    source_root_id: str
    target_root_id: str
    reason: str


@dataclass
class CreatedRoot:
    plan: RootPlan
    issue_number: int
    issue_url: str


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


def parse_non_empty_list(raw: Any, *, field_name: str) -> list[str]:
    if not isinstance(raw, list):
        raise OrchestratorError(f"plan validation failed: {field_name} must be a list")
    out = [as_str(x, "").strip() for x in raw if as_str(x, "").strip()]
    if not out:
        raise OrchestratorError(f"plan validation failed: {field_name} must not be empty")
    return out


def toposort(nodes: dict[str, list[str]], *, context: str) -> list[str]:
    visiting: set[str] = set()
    visited: set[str] = set()
    order: list[str] = []

    def dfs(node_id: str) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            raise OrchestratorError(f"plan validation failed: cycle detected ({context})")
        visiting.add(node_id)
        for dep in nodes.get(node_id, []):
            dfs(dep)
        visiting.remove(node_id)
        visited.add(node_id)
        order.append(node_id)

    for node_id in nodes:
        dfs(node_id)
    return order


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create root issues and DAG task issues from complex requirements.",
    )
    parser.add_argument(
        "--config",
        default=".codex/orchestrator/config.toml",
        help="path to orchestrator config TOML",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--root-issue", type=int, help="existing root issue number")
    source_group.add_argument("--title", default="", help="title for new root issues")
    parser.add_argument(
        "--body-file",
        default="",
        help="requirements markdown file (required when --title is used)",
    )
    parser.add_argument(
        "--plan-file",
        default=None,
        help="optional prebuilt plan JSON file (single-root nodes[] or multi-root roots[])",
    )
    parser.add_argument(
        "--extra-root-label",
        action="append",
        default=[],
        help="extra labels to add to root issues (repeatable)",
    )
    parser.add_argument("--run-id", default=None, help="optional run id")
    parser.add_argument(
        "--max-sprint-days",
        type=int,
        default=7,
        help="max allowed sprint_days in each root issue (default: 7)",
    )
    parser.add_argument(
        "--max-roots",
        type=int,
        default=6,
        help="upper bound of auto-generated root issues (default: 6)",
    )
    parser.add_argument(
        "--review-purpose",
        default=DEFAULT_REVIEW_PURPOSE,
        help="review objective that root-splitting should optimize",
    )
    parser.add_argument(
        "--planner-profile",
        default="",
        help="optional Codex profile override for planning step",
    )
    parser.add_argument(
        "--planner-model",
        default="",
        help="optional Codex model override for planning step",
    )
    parser.add_argument(
        "--planner-timeout-sec",
        type=int,
        default=180,
        help="timeout for auto root planner call (default: 180)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print actions only")

    args = parser.parse_args(argv)
    if args.max_sprint_days <= 0:
        parser.error("--max-sprint-days must be >= 1")
    if args.max_roots <= 0:
        parser.error("--max-roots must be >= 1")
    if args.planner_timeout_sec <= 0:
        parser.error("--planner-timeout-sec must be >= 1")

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
        if not JAPANESE_CHAR_PATTERN.search(args.title):
            parser.error("--title must include Japanese characters")
    return args


class Taskizer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo_root = REPO_ROOT
        self.run_id = args.run_id or f"taskize-{time.strftime('%Y%m%d-%H%M%S')}"
        self.run_dir = self.repo_root / ".agent/taskize-runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

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
        self.github_project_url = as_str(github.get("project_url"), "").strip()

        self.root_label = as_str(github.get("root_label"), "agent:root")
        self.dag_label = as_str(github.get("dag_label"), "agent:dag")
        self.scope_label = as_str(github.get("scope_label"), "")
        self.max_sprint_days = args.max_sprint_days
        self.max_roots = args.max_roots

        codex_manager = section(self.cfg, "codex.manager")
        self.approval = as_str(codex_manager.get("ask_for_approval"), "never")
        self.sandbox = as_str(codex_manager.get("sandbox"), "danger-full-access")
        self.profile = args.planner_profile or as_str(codex_manager.get("profile"), "")
        self.model = args.planner_model or as_str(codex_manager.get("model"), "")
        self.planner_timeout_sec = args.planner_timeout_sec

        self.pipeline_script = (
            self.repo_root / ".codex/skills/orchestrator-manager/scripts/github_dag_pipeline.py"
        )
        if not self.pipeline_script.exists():
            raise OrchestratorError(f"pipeline script not found: {self.pipeline_script}")

        self.pipeline_config_path = self.resolve_pipeline_config_path()

        self.plan_raw_file = self.run_dir / "root-plan.raw.txt"
        self.plan_json_file = self.run_dir / "root-plan.json"
        self.summary_file = self.run_dir / "summary.json"
        self.root_issue_template = self.load_asset_text("root-issue.template.md")

    def resolve_path(self, raw: str) -> Path:
        path = Path(raw)
        if path.is_absolute():
            return path
        cwd_path = (Path.cwd() / path).resolve()
        if cwd_path.exists():
            return cwd_path
        return (self.repo_root / path).resolve()

    def load_asset_text(self, name: str) -> str:
        path = ASSETS_DIR / name
        if not path.exists():
            raise OrchestratorError(f"asset template not found: {path}")
        return path.read_text(encoding="utf-8", errors="replace")

    def gh_ok(self, args: list[str]) -> bool:
        cp = run_cmd(["gh", *args], check=False, capture_output=True)
        return cp.returncode == 0

    def token_scopes(self) -> set[str]:
        cp = run_cmd(["gh", "api", "-i", "user"], check=False, capture_output=True)
        if cp.returncode != 0:
            return set()
        header = (cp.stdout or "").split("\n\n", 1)[0]
        for raw in header.splitlines():
            line = raw.strip()
            if not line.lower().startswith("x-oauth-scopes:"):
                continue
            _, _, value = line.partition(":")
            return {item.strip() for item in value.split(",") if item.strip()}
        return set()

    def remove_github_project_url(self, config_text: str) -> str:
        out: list[str] = []
        in_github = False
        for raw in config_text.splitlines():
            stripped = raw.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                in_github = stripped == "[github]"
            if in_github and stripped.startswith("project_url"):
                continue
            out.append(raw)
        return "\n".join(out).rstrip() + "\n"

    def resolve_pipeline_config_path(self) -> Path:
        if not self.github_project_url:
            return self.config_path
        scopes = self.token_scopes()
        if "read:project" in scopes:
            return self.config_path
        patched = self.remove_github_project_url(
            self.config_path.read_text(encoding="utf-8", errors="replace")
        )
        out = self.run_dir / "config.no-project.toml"
        out.write_text(patched, encoding="utf-8")
        print("warning=project_sync_disabled_missing_scope(read:project)")
        return out

    def extract_first_json_object(self, text: str) -> str:
        depth = 0
        start = None
        in_string = False
        escaped = False
        for i, ch in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if start is None:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                if start is not None and depth == 0:
                    return text[start : i + 1]
        return ""

    def ensure_root_labels(self) -> None:
        for label in dedupe_labels([self.root_label, self.dag_label, self.scope_label]):
            self.gh_ok(["label", "create", "-R", self.repo, label, "--color", "C5DEF5", "--force"])

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
                "body,url,title",
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

    def create_root_issue(self, title: str, body_file: Path) -> tuple[int, str]:
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
            title.strip(),
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

    def run_dag_creation(self, root_issue: int, *, plan_file: Path | None, run_id: str | None = None) -> int:
        cmd = [
            sys.executable,
            str(self.pipeline_script),
            "--config",
            str(self.pipeline_config_path),
            "--root-issue",
            str(root_issue),
            "--create-only",
        ]
        if run_id:
            cmd.extend(["--run-id", run_id])
        if plan_file:
            cmd.extend(["--plan-file", str(plan_file)])

        if self.args.dry_run:
            print("dry-run:create-dag:", " ".join(cmd))
            return 0

        cp = run_cmd(cmd, check=False, capture_output=True)
        if cp.returncode != 0:
            detail = ((cp.stdout or "") + "\n" + (cp.stderr or "")).strip()
            raise OrchestratorError(f"github_dag_pipeline.py failed (rc={cp.returncode})\n{detail}")
        return 0

    def build_planner_prompt(self, title: str, body: str) -> str:
        return "\n".join(
            [
                "あなたは GitHub Issue 分割プランナーです。",
                "目的は、要件をレビュー可能な単位の root issue 群へ分割し、各 root 配下の DAG ノードを定義することです。",
                "",
                "[Review Objective]",
                self.args.review_purpose.strip(),
                "",
                "[Input Title]",
                title.strip(),
                "",
                "[Input Body]",
                body.strip(),
                "",
                "[Rules]",
                "1. root issue は 1 以上、最大 max_roots 件にする。",
                f"2. 各 root の sprint_days は 1..{self.max_sprint_days} の整数にする。",
                "3. root はビジネスレビューで認識齟齬や方向性を確認しやすい単位に分割する。",
                "4. シンプルな単一タスク入力なら root は 1 件にする。",
                "5. 混在した大量入力なら root を複数件に分割する。",
                "6. 各 node は 1 PR で完結できる粒度にする。",
                "7. root title / node title は日本語で書く。",
                "8. acceptance_criteria は各項目を定量評価可能にする（数値・閾値・件数を必須）。",
                "9. ACの表現はタスクごとに変える。固定テンプレート化しない。",
                "10. 『確認する』『適切』など曖昧語のみのACは禁止。",
                "11. acceptance_criteria は checklist 化される前提なので、Markdown 記号は含めない。",
                "12. depends_on / cross_root_dependencies に循環を作らない。",
                "13. 出力は JSON のみ。Markdown や説明文を含めない。",
                "",
                "[JSON Schema]",
                "{",
                '  "roots": [',
                "    {",
                '      "id": "root-1",',
                '      "title": "Root title",',
                '      "sprint_days": 3,',
                '      "sprint_goal": "この root の達成目標",',
                '      "business_value": "この root で提供する価値",',
                '      "review_steps": ["レビュー手順1", "レビュー手順2"],',
                '      "done_criteria": ["完了条件1", "完了条件2"],',
                '      "background": "背景と目的",',
                '      "scope": ["やること1", "やること2"],',
                '      "non_scope": ["やらないこと1"],',
                '      "constraints": ["制約1"],',
                '      "risks": ["リスク1"],',
                '      "nodes": [',
                "        {",
                '          "id": "task-1",',
                '          "title": "日本語のノードタイトル",',
                '          "description": "実装指示",',
                '          "acceptance_criteria": [',
                '            "Aエンドポイントの p95 レスポンス時間が 1.0 秒以下である",',
                '            "POST /v1/xxx の成功時ステータスが 200 で、失敗率が 1% 未満である"',
                "          ],",
                '          "depends_on": []',
                "        }",
                "      ]",
                "    }",
                "  ],",
                '  "cross_root_dependencies": [',
                "    {",
                '      "from": "root-2",',
                '      "to": "root-1",',
                '      "reason": "依存理由"',
                "    }",
                "  ]",
                "}",
            ]
        )

    def normalize_task_text(self, text: str) -> str:
        value = text.strip()
        value = re.sub(r"\s+", " ", value)
        return value[:200]

    def has_japanese(self, text: str) -> bool:
        return bool(JAPANESE_CHAR_PATTERN.search(text))

    def ensure_japanese_title(self, title: str, *, context: str) -> None:
        if self.has_japanese(title):
            return
        raise OrchestratorError(f"plan validation failed: title must be Japanese ({context})")

    def single_root_title(self, title: str) -> str:
        cleaned = title.strip()
        if self.has_japanese(cleaned):
            return cleaned
        return "レビュー: 要件を実装可能な単位へ分割する"

    def ensure_quantitative_check(self, text: str) -> str:
        value = text.strip()
        if QUANTITATIVE_PATTERN.search(value):
            return value
        raise OrchestratorError(
            "plan validation failed: acceptance check must be quantitative "
            f"(missing threshold/count: {value})"
        )

    def build_acceptance_line(self, *, check: str, evidence: str, verify: str) -> str:
        quantitative = self.ensure_quantitative_check(check)
        evidence_path = evidence.strip()
        if evidence_path and not EVIDENCE_PATH_PATTERN.search(evidence_path):
            raise OrchestratorError(
                f"plan validation failed: acceptance evidence must contain path-like token ({evidence_path})"
            )
        verify_cmd = verify.strip()
        if verify_cmd and evidence_path:
            return f"{quantitative}（確認: {verify_cmd} / 記録: {evidence_path}）"
        if verify_cmd:
            return f"{quantitative}（確認: {verify_cmd}）"
        if evidence_path:
            return f"{quantitative}（記録: {evidence_path}）"
        return quantitative

    def normalize_acceptance_criteria(
        self,
        raw_items: Any,
        *,
        root_id: str,
        node_id: str,
    ) -> list[str]:
        if not isinstance(raw_items, list) or not raw_items:
            raise OrchestratorError(
                f"plan validation failed: acceptance_criteria must be non-empty list ({root_id}/{node_id})"
            )
        out: list[str] = []
        for idx, raw in enumerate(raw_items, start=1):
            if isinstance(raw, dict):
                check = as_str(raw.get("check"), "").strip()
                evidence = as_str(raw.get("evidence"), "").strip()
                verify = as_str(raw.get("verify"), "").strip()
                if not check:
                    raise OrchestratorError(
                        f"plan validation failed: acceptance check is required ({root_id}/{node_id})"
                    )
                out.append(self.build_acceptance_line(check=check, evidence=evidence, verify=verify))
                continue

            text = as_str(raw, "").strip()
            if not text:
                continue
            self.ensure_quantitative_check(text)
            out.append(text)
        if not out:
            raise OrchestratorError(
                f"plan validation failed: acceptance_criteria became empty ({root_id}/{node_id})"
            )
        return out

    def parse_task_items(self, body: str) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        current_heading = "general"
        for raw in body.splitlines():
            heading_match = HEADING_PATTERN.match(raw)
            if heading_match:
                current_heading = heading_match.group(1).strip()
                continue
            task_match = TASK_LINE_PATTERN.match(raw)
            if not task_match:
                continue
            task_text = self.normalize_task_text(task_match.group("task"))
            if task_text:
                items.append((current_heading, task_text))
        if items:
            return items

        fallback_lines = [
            self.normalize_task_text(line)
            for line in body.splitlines()
            if self.normalize_task_text(line)
        ]
        return [("general", line) for line in fallback_lines[:8]]

    def classify_phase(self, heading: str, task: str) -> str:
        source = f"{heading} {task}".lower()
        for phase in PHASE_ORDER:
            if phase == "generic":
                continue
            for keyword in PHASE_KEYWORDS.get(phase, ()):
                if keyword.lower() in source:
                    return phase
        return "generic"

    def heuristic_plan_payload(self, title: str, body: str) -> dict[str, Any]:
        task_items = self.parse_task_items(body)
        if not task_items:
            task_items = [("general", title.strip())]

        grouped: dict[str, list[str]] = {phase: [] for phase in PHASE_ORDER}
        for heading, task in task_items:
            phase = self.classify_phase(heading, task)
            grouped.setdefault(phase, []).append(task)

        non_empty_phases = [phase for phase in PHASE_ORDER if grouped.get(phase)]
        total_tasks = sum(len(grouped[phase]) for phase in non_empty_phases)
        if total_tasks <= 3:
            merged = [task for phase in non_empty_phases for task in grouped[phase]]
            grouped = {phase: [] for phase in PHASE_ORDER}
            grouped["generic"] = merged
            non_empty_phases = ["generic"]

        non_empty_phases = non_empty_phases[: self.max_roots]
        roots: list[dict[str, Any]] = []
        for index, phase in enumerate(non_empty_phases, start=1):
            tasks = grouped.get(phase, [])
            if not tasks:
                continue
            phase_title, phase_sprint = PHASE_LABELS.get(phase, PHASE_LABELS["generic"])
            sprint_days = min(self.max_sprint_days, max(1, phase_sprint))
            root_id = f"root-{index:02d}-{phase}"
            node_entries: list[dict[str, Any]] = []
            prev_node = ""
            for node_idx, task in enumerate(tasks, start=1):
                node_id = f"{phase}-task-{node_idx:02d}"
                deps = [prev_node] if prev_node else []
                prev_node = node_id
                node_entries.append(
                    {
                        "id": node_id,
                        "title": f"タスク: {task[:72]}",
                        "description": task,
                        "acceptance_criteria": [
                            f"「{task}」を満たす実装/設定ケース数が 1 件以上である",
                            f"「{task}」に関する未解決指摘件数が 0 件である",
                        ],
                        "depends_on": deps,
                    }
                )
            roots.append(
                {
                    "id": root_id,
                    "title": phase_title,
                    "sprint_days": sprint_days,
                    "sprint_goal": f"{phase_title} を完了し、次のレビュー判断に必要な情報を揃える。",
                    "business_value": "レビュー時点で認識齟齬と方向性リスクを可視化できる。",
                    "review_steps": [
                        "タスク成果物がレビュー観点を満たすか確認する",
                        "次フェーズ着手に必要な前提が揃っているか確認する",
                    ],
                    "done_criteria": [
                        "配下タスクがすべて Issue 化され、受け入れ条件が明記されている",
                        "レビューで次アクションを判断できる状態になっている",
                    ],
                    "background": f"{title.strip()} の要件をレビュー単位へ分割した。",
                    "scope": tasks[: min(8, len(tasks))],
                    "non_scope": ["実行ディスパッチと実装作業自体は本スクリプトの対象外"],
                    "constraints": [
                        f"sprint_days は 1..{self.max_sprint_days} に収める",
                        "受け入れ条件は検証可能な文にする",
                    ],
                    "risks": ["入力要件の曖昧さにより追加レビューが必要になる可能性"],
                    "nodes": node_entries,
                }
            )

        cross_root_dependencies: list[dict[str, str]] = []
        for idx in range(1, len(roots)):
            current = roots[idx]["id"]
            prev = roots[idx - 1]["id"]
            cross_root_dependencies.append(
                {
                    "from": current,
                    "to": prev,
                    "reason": "レビュー判断の順序を保つため",
                }
            )
        return {
            "roots": roots,
            "cross_root_dependencies": cross_root_dependencies,
        }

    def read_plan_payload(self, title: str, body: str) -> dict[str, Any]:
        if self.args.plan_file:
            plan_path = self.resolve_path(self.args.plan_file)
            if not plan_path.exists():
                raise OrchestratorError(f"plan file not found: {plan_path}")
            text = plan_path.read_text(encoding="utf-8")
            self.plan_json_file.write_text(text, encoding="utf-8")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise OrchestratorError(f"invalid JSON in plan file: {plan_path}") from exc
            if not isinstance(payload, dict):
                raise OrchestratorError("plan file payload must be an object")
            return payload

        planner_prompt = self.build_planner_prompt(title, body)
        cmd = ["codex", "-a", self.approval, "-s", self.sandbox, "exec"]
        if self.profile:
            cmd.extend(["--profile", self.profile])
        if self.model:
            cmd.extend(["--model", self.model])
        cmd.extend(["-o", str(self.plan_raw_file), "-"])

        use_fallback = False
        fallback_reason = ""
        try:
            cp = subprocess.run(
                cmd,
                input=planner_prompt,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.planner_timeout_sec,
            )
        except subprocess.TimeoutExpired:
            use_fallback = True
            fallback_reason = "timeout"
            cp = None
        if cp is not None:
            (self.run_dir / "planner.events.jsonl").write_text(cp.stdout or "", encoding="utf-8")
            (self.run_dir / "planner.stderr.log").write_text(cp.stderr or "", encoding="utf-8")
            if cp.returncode != 0:
                use_fallback = True
                fallback_reason = f"exit={cp.returncode}"

        if use_fallback:
            payload = self.heuristic_plan_payload(title, body)
            self.plan_json_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"warning=planner_fallback_used({fallback_reason})")
            return payload

        raw = self.plan_raw_file.read_text(encoding="utf-8", errors="replace")
        plan_text = raw.strip()
        try:
            payload = json.loads(plan_text)
        except json.JSONDecodeError:
            extracted = self.extract_first_json_object(raw)
            if not extracted:
                payload = self.heuristic_plan_payload(title, body)
                self.plan_json_file.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print("warning=planner_fallback_used(invalid_json)")
                return payload
            self.plan_json_file.write_text(extracted, encoding="utf-8")
            try:
                payload = json.loads(extracted)
            except json.JSONDecodeError as exc:
                payload = self.heuristic_plan_payload(title, body)
                self.plan_json_file.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print("warning=planner_fallback_used(json_decode_error)")
                return payload
        else:
            self.plan_json_file.write_text(plan_text, encoding="utf-8")

        if not isinstance(payload, dict):
            payload = self.heuristic_plan_payload(title, body)
            self.plan_json_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print("warning=planner_fallback_used(non_object_payload)")
            return payload
        return payload

    def parse_nodes(self, raw_nodes: Any, *, root_id: str) -> list[DAGNode]:
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise OrchestratorError(f"plan validation failed: nodes must be a non-empty list ({root_id})")
        nodes: list[DAGNode] = []
        seen: set[str] = set()
        for raw in raw_nodes:
            if not isinstance(raw, dict):
                raise OrchestratorError(f"plan validation failed: node must be object ({root_id})")
            node_id = as_str(raw.get("id"), "").strip()
            title = as_str(raw.get("title"), "").strip()
            description = as_str(raw.get("description"), "").strip()
            if not node_id or not title or not description:
                raise OrchestratorError(
                    f"plan validation failed: node id/title/description are required ({root_id})"
                )
            self.ensure_japanese_title(title, context=f"{root_id}/{node_id}")
            if node_id in seen:
                raise OrchestratorError(f"plan validation failed: duplicate node id ({root_id}/{node_id})")
            seen.add(node_id)
            depends_raw = raw.get("depends_on", [])
            if not isinstance(depends_raw, list):
                raise OrchestratorError(f"plan validation failed: depends_on must be list ({root_id}/{node_id})")
            depends_on = [as_str(x, "").strip() for x in depends_raw if as_str(x, "").strip()]
            acceptance = self.normalize_acceptance_criteria(
                raw.get("acceptance_criteria", []),
                root_id=root_id,
                node_id=node_id,
            )
            nodes.append(
                DAGNode(
                    node_id=node_id,
                    title=title,
                    description=description,
                    acceptance_criteria=acceptance,
                    depends_on=depends_on,
                )
            )

        id_set = {node.node_id for node in nodes}
        graph: dict[str, list[str]] = {}
        for node in nodes:
            graph[node.node_id] = list(node.depends_on)
            for dep in node.depends_on:
                if dep not in id_set:
                    raise OrchestratorError(
                        "plan validation failed: depends_on references unknown node "
                        f"({root_id}/{node.node_id} -> {dep})"
                    )
        _ = toposort(graph, context=f"node-dependency:{root_id}")
        return nodes

    def fallback_single_root_plan(self, title: str, body: str, payload: dict[str, Any]) -> list[RootPlan]:
        sprint_days = 5
        match = SPRINT_DAYS_PATTERN.search(body)
        if match:
            sprint_days = int(match.group(1))
        sprint_days = max(1, min(self.max_sprint_days, sprint_days))

        raw_nodes = payload.get("nodes")
        nodes = self.parse_nodes(raw_nodes, root_id="root-1")
        return [
            RootPlan(
                root_id="root-1",
                title=self.single_root_title(title),
                sprint_days=sprint_days,
                sprint_goal=f"{title.strip()} を実装可能な単位へ分解して着手する。",
                business_value="実装優先順位が明確になり、レビューで認識齟齬を早期に発見できる。",
                review_steps=[
                    "タスク分割と依存関係が妥当か確認する",
                    "受け入れ条件が検証可能か確認する",
                ],
                done_criteria=[
                    "DAG 子 issue が作成済みである",
                    "各子 issue に検証可能な受け入れ条件がある",
                ],
                background=body.strip()[:1200] or f"{title.strip()} の要件を実行可能タスクに変換する。",
                scope=["入力要件を root + DAG issue へ変換する"],
                non_scope=["実装ディスパッチは本スクリプト対象外"],
                constraints=[f"sprint_days は 1..{self.max_sprint_days}"],
                risks=["入力要件が曖昧な場合、追加のレビューで補正が必要になる"],
                nodes=nodes,
            )
        ]

    def parse_root_plans(self, title: str, body: str, payload: dict[str, Any]) -> list[RootPlan]:
        roots_raw = payload.get("roots")
        if roots_raw is None:
            return self.fallback_single_root_plan(title, body, payload)
        if not isinstance(roots_raw, list) or not roots_raw:
            raise OrchestratorError("plan validation failed: roots must be a non-empty list")
        if len(roots_raw) > self.max_roots:
            raise OrchestratorError(
                f"plan validation failed: roots count exceeds max_roots ({len(roots_raw)} > {self.max_roots})"
            )

        roots: list[RootPlan] = []
        seen_root_ids: set[str] = set()
        for idx, raw_root in enumerate(roots_raw, start=1):
            if not isinstance(raw_root, dict):
                raise OrchestratorError(f"plan validation failed: root must be object (index={idx})")
            root_id = as_str(raw_root.get("id"), "").strip()
            root_title = as_str(raw_root.get("title"), "").strip()
            sprint_days = as_int(raw_root.get("sprint_days"), 0)
            if not root_id:
                raise OrchestratorError(f"plan validation failed: root.id is required (index={idx})")
            if root_id in seen_root_ids:
                raise OrchestratorError(f"plan validation failed: duplicate root id ({root_id})")
            if not root_title:
                raise OrchestratorError(f"plan validation failed: root.title is required ({root_id})")
            self.ensure_japanese_title(root_title, context=f"{root_id}")
            if sprint_days < 1 or sprint_days > self.max_sprint_days:
                raise OrchestratorError(
                    f"plan validation failed: sprint_days out of range ({root_id}: {sprint_days})"
                )
            seen_root_ids.add(root_id)

            nodes = self.parse_nodes(raw_root.get("nodes"), root_id=root_id)
            plan = RootPlan(
                root_id=root_id,
                title=root_title,
                sprint_days=sprint_days,
                sprint_goal=as_str(raw_root.get("sprint_goal"), "").strip(),
                business_value=as_str(raw_root.get("business_value"), "").strip(),
                review_steps=parse_non_empty_list(raw_root.get("review_steps", []), field_name=f"{root_id}.review_steps"),
                done_criteria=parse_non_empty_list(raw_root.get("done_criteria", []), field_name=f"{root_id}.done_criteria"),
                background=as_str(raw_root.get("background"), "").strip(),
                scope=parse_non_empty_list(raw_root.get("scope", []), field_name=f"{root_id}.scope"),
                non_scope=parse_non_empty_list(raw_root.get("non_scope", []), field_name=f"{root_id}.non_scope"),
                constraints=parse_non_empty_list(raw_root.get("constraints", []), field_name=f"{root_id}.constraints"),
                risks=parse_non_empty_list(raw_root.get("risks", []), field_name=f"{root_id}.risks"),
                nodes=nodes,
            )
            if not plan.sprint_goal:
                raise OrchestratorError(f"plan validation failed: sprint_goal is required ({root_id})")
            if not plan.business_value:
                raise OrchestratorError(f"plan validation failed: business_value is required ({root_id})")
            if not plan.background:
                raise OrchestratorError(f"plan validation failed: background is required ({root_id})")
            roots.append(plan)
        return roots

    def parse_cross_root_dependencies(
        self,
        payload: dict[str, Any],
        *,
        root_ids: set[str],
    ) -> list[CrossRootDependency]:
        raw_deps = payload.get("cross_root_dependencies", [])
        if raw_deps is None:
            return []
        if not isinstance(raw_deps, list):
            raise OrchestratorError("plan validation failed: cross_root_dependencies must be a list")
        deps: list[CrossRootDependency] = []
        graph: dict[str, list[str]] = {root_id: [] for root_id in root_ids}
        for raw in raw_deps:
            if not isinstance(raw, dict):
                raise OrchestratorError("plan validation failed: cross_root_dependency must be object")
            source = as_str(raw.get("from"), "").strip()
            target = as_str(raw.get("to"), "").strip()
            reason = as_str(raw.get("reason"), "").strip()
            if not source or not target:
                raise OrchestratorError("plan validation failed: cross_root_dependency from/to are required")
            if source not in root_ids or target not in root_ids:
                raise OrchestratorError(
                    f"plan validation failed: cross_root_dependency references unknown root ({source} -> {target})"
                )
            graph[source].append(target)
            deps.append(CrossRootDependency(source_root_id=source, target_root_id=target, reason=reason))
        _ = toposort(graph, context="cross-root-dependency")
        return deps

    def bullet_lines(self, items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items)

    def numbered_lines(self, items: list[str]) -> str:
        return "\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1))

    def relation_section(
        self,
        *,
        heading: str,
        deps: list[CrossRootDependency],
        field_name: str,
    ) -> str:
        if not deps:
            return ""
        lines = [f"## {heading}"]
        for dep in deps:
            ref = as_str(getattr(dep, field_name), "")
            reason = f" ({dep.reason})" if dep.reason else ""
            lines.append(f"- {ref}{reason}")
        return "\n\n" + "\n".join(lines)

    def render_root_body(
        self,
        plan: RootPlan,
        *,
        deps_in: list[CrossRootDependency],
        deps_out: list[CrossRootDependency],
    ) -> str:
        replacements = {
            "{{SPRINT_DAYS}}": str(plan.sprint_days),
            "{{SPRINT_GOAL_LINES}}": self.bullet_lines([plan.sprint_goal]),
            "{{BUSINESS_VALUE_LINES}}": self.bullet_lines([plan.business_value]),
            "{{REVIEW_STEPS_LINES}}": self.numbered_lines(plan.review_steps),
            "{{DONE_CRITERIA_LINES}}": self.bullet_lines(plan.done_criteria),
            "{{BACKGROUND}}": plan.background,
            "{{SCOPE_LINES}}": self.bullet_lines(plan.scope),
            "{{NON_SCOPE_LINES}}": self.bullet_lines(plan.non_scope),
            "{{CONSTRAINTS_LINES}}": self.bullet_lines(plan.constraints),
            "{{RISKS_LINES}}": self.bullet_lines(plan.risks),
            "{{DEPENDS_ON_SECTION}}": self.relation_section(
                heading="依存する Root Issue",
                deps=deps_in,
                field_name="target_root_id",
            ),
            "{{BLOCKS_SECTION}}": self.relation_section(
                heading="ブロックする Root Issue",
                deps=deps_out,
                field_name="source_root_id",
            ),
        }

        body = self.root_issue_template
        for key, value in replacements.items():
            body = body.replace(key, value)
        if "{{" in body:
            raise OrchestratorError("unresolved placeholder in root issue template")
        return body.rstrip() + "\n"

    def compose_root_title(self, root: RootPlan, *, root_count: int) -> str:
        base_title = self.args.title.strip()
        if root_count <= 1:
            return root.title
        return f"{base_title} / {root.title}"

    def write_node_plan_file(self, root: RootPlan) -> Path:
        out = self.run_dir / f"dag-plan-{safe_key(root.root_id)}.json"
        payload = {
            "nodes": [
                {
                    "id": node.node_id,
                    "title": node.title,
                    "description": node.description,
                    "acceptance_criteria": node.acceptance_criteria,
                    "depends_on": node.depends_on,
                }
                for node in root.nodes
            ]
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out

    def process_existing_root(self) -> int:
        payload = self.root_issue_payload(self.args.root_issue)
        body = as_str(payload.get("body"), "")
        url = as_str(payload.get("url"), "")
        validate_root_issue_body(
            body,
            max_sprint_days=self.max_sprint_days,
            source=f"existing root issue #{self.args.root_issue}",
        )
        plan_file: Path | None = None
        if self.args.plan_file:
            plan_file = self.resolve_path(self.args.plan_file)
            if not plan_file.exists():
                raise OrchestratorError(f"plan file not found: {plan_file}")
        self.run_dag_creation(self.args.root_issue, plan_file=plan_file, run_id=self.run_id)
        print(f"repo={self.repo}")
        print(f"root_issue={self.args.root_issue}")
        if url:
            print(f"root_url={url}")
        print("status=dag_issues_created")
        return 0

    def plan_multi_root(self) -> tuple[list[RootPlan], list[CrossRootDependency]]:
        body_file = self.resolve_path(self.args.body_file)
        if not body_file.exists():
            raise OrchestratorError(f"body file not found: {body_file}")
        body_text = body_file.read_text(encoding="utf-8", errors="replace")
        payload = self.read_plan_payload(self.args.title, body_text)
        roots = self.parse_root_plans(self.args.title, body_text, payload)
        if not roots:
            raise OrchestratorError("plan validation failed: at least one root is required")
        root_ids = {root.root_id for root in roots}
        deps = self.parse_cross_root_dependencies(payload, root_ids=root_ids)
        return roots, deps

    def root_dependencies(
        self,
        deps: list[CrossRootDependency],
    ) -> tuple[dict[str, list[CrossRootDependency]], dict[str, list[CrossRootDependency]], list[str]]:
        incoming: dict[str, list[CrossRootDependency]] = {}
        outgoing: dict[str, list[CrossRootDependency]] = {}
        graph: dict[str, list[str]] = {}
        for dep in deps:
            incoming.setdefault(dep.source_root_id, []).append(dep)
            outgoing.setdefault(dep.target_root_id, []).append(dep)
            graph.setdefault(dep.source_root_id, []).append(dep.target_root_id)
            graph.setdefault(dep.target_root_id, [])
        order = toposort(graph, context="cross-root-dependency-order")
        return incoming, outgoing, order

    def create_multi_root_issues(self, roots: list[RootPlan], deps: list[CrossRootDependency]) -> list[CreatedRoot]:
        self.ensure_root_labels()

        incoming, outgoing, dep_order = self.root_dependencies(deps)
        by_id = {root.root_id: root for root in roots}
        if dep_order:
            ordered_roots = [by_id[root_id] for root_id in dep_order if root_id in by_id]
            missing = [root for root in roots if root.root_id not in {x.root_id for x in ordered_roots}]
            ordered_roots.extend(missing)
        else:
            ordered_roots = roots

        created: list[CreatedRoot] = []
        root_count = len(roots)
        for index, root in enumerate(ordered_roots, start=1):
            body_text = self.render_root_body(
                root,
                deps_in=incoming.get(root.root_id, []),
                deps_out=outgoing.get(root.root_id, []),
            )
            validate_root_issue_body(
                body_text,
                max_sprint_days=self.max_sprint_days,
                source=f"planned root {root.root_id}",
            )
            body_file = self.run_dir / f"root-body-{safe_key(root.root_id)}.md"
            body_file.write_text(body_text, encoding="utf-8")
            issue_title = self.compose_root_title(root, root_count=root_count)
            issue_no, issue_url = self.create_root_issue(issue_title, body_file)
            created.append(CreatedRoot(plan=root, issue_number=issue_no, issue_url=issue_url))

            if issue_no > 0:
                node_plan_file = self.write_node_plan_file(root)
                root_run_id = f"{self.run_id}-{index:02d}-{safe_key(root.root_id)}"
                self.run_dag_creation(issue_no, plan_file=node_plan_file, run_id=root_run_id)

        if created and not self.args.dry_run:
            root_number_by_id = {item.plan.root_id: item.issue_number for item in created}
            for dep in deps:
                src_issue = root_number_by_id.get(dep.source_root_id, 0)
                dst_issue = root_number_by_id.get(dep.target_root_id, 0)
                if src_issue <= 0 or dst_issue <= 0:
                    continue
                reason = f"\n- reason: {dep.reason}" if dep.reason else ""
                comment = (
                    "Cross-root dependency registered\n"
                    f"- depends_on_root: #{dst_issue} ({dep.target_root_id}){reason}\n"
                )
                self.gh_ok(["issue", "comment", "-R", self.repo, str(src_issue), "--body", comment])

        summary_payload = {
            "repo": self.repo,
            "run_id": self.run_id,
            "root_count": len(created),
            "roots": [
                {
                    "root_id": item.plan.root_id,
                    "title": item.plan.title,
                    "issue_number": item.issue_number,
                    "issue_url": item.issue_url,
                    "sprint_days": item.plan.sprint_days,
                    "node_count": len(item.plan.nodes),
                }
                for item in created
            ],
        }
        self.summary_file.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return created

    def run(self) -> int:
        if self.args.root_issue is not None:
            return self.process_existing_root()

        roots, deps = self.plan_multi_root()
        created = self.create_multi_root_issues(roots, deps)
        print(f"repo={self.repo}")
        print(f"root_issue_count={len(created)}")
        for item in created:
            if item.issue_number > 0:
                print(f"root_issue={item.issue_number} root_id={item.plan.root_id} root_url={item.issue_url}")
            else:
                print(f"root_issue=dry-run root_id={item.plan.root_id}")
        print("status=dag_issues_created")
        return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    runner = Taskizer(args)
    return runner.run()


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except OrchestratorError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
