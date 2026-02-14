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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
COMMON_DIR = REPO_ROOT / ".codex/orchestrator"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from orchestrator_common import (  # noqa: E402
    OrchestratorError,
    as_bool,
    as_int,
    as_str,
    ensure_command,
    extract_first_url,
    load_toml,
    log,
    run_cmd,
    run_cmd_text,
    safe_key,
    section,
)


@dataclass
class DAGNode:
    node_id: str
    title: str
    description: str
    acceptance_criteria: list[str]
    depends_on: list[str]


class DAGPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo_root = REPO_ROOT
        self.config_path = Path(args.config)
        if not self.config_path.is_absolute():
            self.config_path = self.repo_root / self.config_path
        if not self.config_path.exists():
            raise OrchestratorError(f"config not found: {self.config_path}")

        ensure_command("gh")
        ensure_command("codex")

        self.cfg = load_toml(self.config_path)
        core = section(self.cfg, "core")
        if as_str(core.get("mode"), "github") != "github":
            raise OrchestratorError("github_dag_pipeline.py requires [core].mode = github")

        self.root_issue = args.root_issue
        self.run_id = args.run_id or f"dag-{time.strftime('%Y%m%d-%H%M%S')}"
        self.plan_file = Path(args.plan_file) if args.plan_file else None
        self.max_rounds = args.max_rounds
        self.dispatch_limit = args.dispatch_limit

        github = section(self.cfg, "github")
        self.github_repo = as_str(github.get("repo"), "")
        if not self.github_repo:
            raise OrchestratorError("[github].repo is required")

        self.github_queue_label = as_str(github.get("queue_label"), "agent:queued")
        self.github_running_label = as_str(github.get("running_label"), "agent:running")
        self.github_review_label = as_str(github.get("review_label"), "agent:review")
        self.github_done_label = as_str(github.get("done_label"), "agent:done")
        self.github_failed_label = as_str(github.get("failed_label"), "agent:failed")
        self.github_needs_human_label = as_str(github.get("needs_human_label"), "agent:needs-human")
        self.github_scope_label = as_str(github.get("scope_label"), "")
        self.github_dag_label = as_str(github.get("dag_label"), "agent:dag")
        self.github_dag_blocked_label = as_str(github.get("dag_blocked_label"), "agent:blocked")
        self.github_root_label = as_str(github.get("root_label"), "agent:root")

        codex_manager = section(self.cfg, "codex.manager")
        self.approval = as_str(codex_manager.get("ask_for_approval"), "never")
        self.sandbox = as_str(codex_manager.get("sandbox"), "danger-full-access")
        self.profile = as_str(codex_manager.get("profile"), "")
        self.model = as_str(codex_manager.get("model"), "")
        self.output_language = as_str(core.get("output_language"), "ja")
        self.output_language_instruction = as_str(core.get("output_language_instruction"), "")
        if not self.output_language_instruction:
            if self.output_language == "ja":
                self.output_language_instruction = "説明文は日本語で書いてください。"
            else:
                self.output_language_instruction = f"Write all narrative text in {self.output_language}."

        self.run_dir = self.repo_root / ".agent/dag-runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_log_file = self.run_dir / "pipeline.log"
        self.plan_raw_file = self.run_dir / "plan.raw.txt"
        self.plan_json_file = self.run_dir / "plan.json"
        self.node_map_file = self.run_dir / "node_issues.tsv"

        self.manager_script = SCRIPT_DIR / "manager_dispatch.py"
        if not self.manager_script.exists():
            raise OrchestratorError(f"manager python script not found: {self.manager_script}")

    def plog(self, message: str) -> None:
        log(message)
        with self.pipeline_log_file.open("a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    def gh_text(self, args: list[str], *, check: bool = True, default: str = "") -> str:
        return run_cmd_text(["gh", *args], check=check, default=default)

    def gh_ok(self, args: list[str]) -> bool:
        cp = run_cmd(["gh", *args], check=False, capture_output=True)
        return cp.returncode == 0

    def create_labels_if_needed(self) -> None:
        labels = [
            self.github_queue_label,
            self.github_running_label,
            self.github_review_label,
            self.github_done_label,
            self.github_failed_label,
            self.github_needs_human_label,
            self.github_dag_label,
            self.github_dag_blocked_label,
            self.github_root_label,
        ]
        if self.github_scope_label:
            labels.append(self.github_scope_label)
        for label in labels:
            if not label:
                continue
            self.gh_ok(["label", "create", "-R", self.github_repo, label, "--color", "C5DEF5", "--force"])

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

    def load_plan_nodes(self) -> list[DAGNode]:
        if self.plan_file:
            if not self.plan_file.exists():
                raise OrchestratorError(f"plan file not found: {self.plan_file}")
            self.plan_json_file.write_text(self.plan_file.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            root_title = self.gh_text(
                ["issue", "view", "-R", self.github_repo, str(self.root_issue), "--json", "title", "--jq", ".title"]
            )
            root_body = self.gh_text(
                ["issue", "view", "-R", self.github_repo, str(self.root_issue), "--json", "body", "--jq", ".body"]
            )
            root_url = self.gh_text(
                ["issue", "view", "-R", self.github_repo, str(self.root_issue), "--json", "url", "--jq", ".url"]
            )
            planner_prompt = "\n".join(
                [
                    "あなたは GitHub Issue を DAG に分解するプランナーです。",
                    "以下の Issue を、依存関係を持つ実装タスクへ分割してください。",
                    "",
                    f"Root Issue: #{self.root_issue}",
                    f"Title: {root_title}",
                    "Body:",
                    root_body,
                    f"URL: {root_url}",
                    "",
                    "要件:",
                    "1. ノード数は 1 以上 8 以下。",
                    "2. 各ノードは独立した実装単位にする。",
                    "3. depends_on は node id の配列で表す。",
                    "4. 依存循環を作らない。",
                    "5. 出力は JSON のみ。Markdown や説明文を含めない。",
                    "6. JSON schema:",
                    "{",
                    '  "nodes": [',
                    "    {",
                    '      "id": "task-1",',
                    '      "title": "短いタイトル",',
                    '      "description": "実装指示",',
                    '      "acceptance_criteria": ["完了条件1", "完了条件2"],',
                    '      "depends_on": ["task-0"]',
                    "    }",
                    "  ]",
                    "}",
                    "",
                    "[Output language requirement]",
                    self.output_language_instruction,
                ]
            )

            cmd = ["codex", "-a", self.approval, "-s", self.sandbox, "exec"]
            if self.profile:
                cmd.extend(["--profile", self.profile])
            if self.model:
                cmd.extend(["--model", self.model])
            cmd.extend(["-o", str(self.plan_raw_file), "-"])
            cp = subprocess.run(
                cmd,
                input=planner_prompt,
                text=True,
                capture_output=True,
                check=False,
            )
            (self.run_dir / "plan.events.jsonl").write_text(cp.stdout or "", encoding="utf-8")
            (self.run_dir / "plan.stderr.log").write_text(cp.stderr or "", encoding="utf-8")
            if cp.returncode != 0:
                raise OrchestratorError(f"planner codex execution failed (rc={cp.returncode})")

            raw = self.plan_raw_file.read_text(encoding="utf-8", errors="replace")
            plan_text = raw.strip()
            try:
                json.loads(plan_text)
            except json.JSONDecodeError:
                extracted = self.extract_first_json_object(raw)
                if not extracted:
                    raise OrchestratorError(f"failed to parse DAG JSON from planner output: {self.plan_raw_file}")
                plan_text = extracted
            self.plan_json_file.write_text(plan_text, encoding="utf-8")

        payload = json.loads(self.plan_json_file.read_text(encoding="utf-8"))
        nodes_raw = payload.get("nodes")
        if not isinstance(nodes_raw, list) or not nodes_raw:
            raise OrchestratorError("plan validation failed: nodes array is required")

        nodes: list[DAGNode] = []
        seen: set[str] = set()
        for raw in nodes_raw:
            if not isinstance(raw, dict):
                raise OrchestratorError("plan validation failed: node must be an object")
            node_id = as_str(raw.get("id"), "").strip()
            title = as_str(raw.get("title"), "").strip()
            description = as_str(raw.get("description"), "").strip()
            if not node_id or not title or not description:
                raise OrchestratorError("plan validation failed: id/title/description are required")
            if node_id in seen:
                raise OrchestratorError(f"plan validation failed: duplicate node id ({node_id})")
            seen.add(node_id)
            depends_raw = raw.get("depends_on", [])
            if not isinstance(depends_raw, list):
                raise OrchestratorError(f"plan validation failed: depends_on must be list ({node_id})")
            depends_on = [as_str(x, "").strip() for x in depends_raw if as_str(x, "").strip()]
            acceptance_raw = raw.get("acceptance_criteria", [])
            if isinstance(acceptance_raw, list):
                acceptance = [as_str(x, "").strip() for x in acceptance_raw if as_str(x, "").strip()]
            else:
                acceptance = []
            nodes.append(
                DAGNode(
                    node_id=node_id,
                    title=title,
                    description=description,
                    acceptance_criteria=acceptance,
                    depends_on=depends_on,
                )
            )

        ids = {node.node_id for node in nodes}
        for node in nodes:
            for dep in node.depends_on:
                if dep not in ids:
                    raise OrchestratorError(
                        f"plan validation failed: depends_on references unknown node id ({node.node_id} -> {dep})"
                    )
        self.ensure_no_cycles(nodes)
        return nodes

    def ensure_no_cycles(self, nodes: list[DAGNode]) -> None:
        graph = {node.node_id: list(node.depends_on) for node in nodes}
        visited: set[str] = set()
        visiting: set[str] = set()

        def dfs(node_id: str) -> None:
            if node_id in visited:
                return
            if node_id in visiting:
                raise OrchestratorError("plan validation failed: cycle detected in depends_on")
            visiting.add(node_id)
            for dep in graph.get(node_id, []):
                dfs(dep)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in graph:
            dfs(node_id)

    def create_node_issues(self, nodes: list[DAGNode], root_url: str) -> dict[str, str]:
        mapping: dict[str, str] = {}
        with self.node_map_file.open("w", encoding="utf-8") as f:
            f.write("node_id\tissue_number\tdepends_on_csv\n")
            for node in nodes:
                depends_csv = ",".join(node.depends_on)
                issue_body_file = self.run_dir / f"issue-body-{safe_key(node.node_id)}.md"
                lines = [
                    "<!-- orchestrator:dag-node:v1 -->",
                    f"dag_run_id: {self.run_id}",
                    f"root_issue: {self.root_issue}",
                    f"dag_node_id: {node.node_id}",
                    f"depends_on: {depends_csv}",
                    "",
                    "## Root Issue",
                    f"- #{self.root_issue}",
                    f"- {root_url}",
                    "",
                    "## Task Description",
                    node.description,
                    "",
                ]
                if node.acceptance_criteria:
                    lines.append("## Acceptance Criteria")
                    for item in node.acceptance_criteria:
                        lines.append(f"- {item}")
                    lines.append("")
                issue_body_file.write_text("\n".join(lines), encoding="utf-8")

                cmd = [
                    "gh",
                    "issue",
                    "create",
                    "-R",
                    self.github_repo,
                    "--title",
                    f"[DAG:{self.run_id}][{node.node_id}] {node.title}",
                    "--body-file",
                    str(issue_body_file),
                    "--label",
                    self.github_dag_label,
                    "--label",
                    self.github_root_label,
                ]
                if self.github_scope_label:
                    cmd.extend(["--label", self.github_scope_label])
                if depends_csv:
                    cmd.extend(["--label", self.github_dag_blocked_label])
                else:
                    cmd.extend(["--label", self.github_queue_label])

                cp = run_cmd(cmd, check=False, capture_output=True)
                if cp.returncode != 0:
                    raise OrchestratorError(f"gh issue create failed for node={node.node_id}: {cp.stderr.strip()}")
                created_text = (cp.stdout or "") + (cp.stderr or "")
                issue_url = extract_first_url(created_text)
                m = re.search(r"/issues/(\d+)", issue_url)
                if not m:
                    raise OrchestratorError(
                        f"failed to parse issue number for node {node.node_id}: {created_text.strip()}"
                    )
                issue_number = m.group(1)
                mapping[node.node_id] = issue_number
                f.write(f"{node.node_id}\t{issue_number}\t{depends_csv}\n")
                self.plog(f"created node issue node={node.node_id} issue={issue_number} deps={depends_csv}")
        return mapping

    def node_rows(self) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        if not self.node_map_file.exists():
            return rows
        for line in self.node_map_file.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            node_id, issue, depends_csv = parts[0], parts[1], parts[2]
            if node_id == "node_id":
                continue
            rows.append((node_id, issue, depends_csv))
        return rows

    def issue_has_label(self, issue: str, label: str) -> bool:
        if not label:
            return False
        out = self.gh_text(
            ["issue", "view", "-R", self.github_repo, issue, "--json", "labels", "--jq", ".labels[].name"],
            check=False,
            default="",
        )
        return label in {x.strip() for x in out.splitlines() if x.strip()}

    def issue_is_done(self, issue: str) -> bool:
        state = self.gh_text(
            ["issue", "view", "-R", self.github_repo, issue, "--json", "state", "--jq", ".state"],
            check=False,
            default="",
        )
        if state == "CLOSED":
            return True
        return self.issue_has_label(issue, self.github_done_label)

    def issue_is_failed(self, issue: str) -> bool:
        return self.issue_has_label(issue, self.github_failed_label) or self.issue_has_label(
            issue, self.github_needs_human_label
        )

    def all_dependencies_done(self, node_to_issue: dict[str, str], depends_csv: str) -> bool:
        if not depends_csv:
            return True
        for dep in [x for x in depends_csv.split(",") if x]:
            dep_issue = node_to_issue.get(dep, "")
            if not dep_issue:
                return False
            if not self.issue_is_done(dep_issue):
                return False
        return True

    def refresh_dependency_gates(self) -> None:
        rows = self.node_rows()
        node_to_issue = {node: issue for node, issue, _ in rows}
        for node, issue, depends_csv in rows:
            if self.issue_is_done(issue):
                continue
            if self.all_dependencies_done(node_to_issue, depends_csv):
                self.gh_ok(["issue", "edit", "-R", self.github_repo, issue, "--add-label", self.github_queue_label])
                self.gh_ok(
                    ["issue", "edit", "-R", self.github_repo, issue, "--remove-label", self.github_dag_blocked_label]
                )
                self.gh_ok(["issue", "edit", "-R", self.github_repo, issue, "--remove-label", self.github_failed_label])
                self.gh_ok(
                    ["issue", "edit", "-R", self.github_repo, issue, "--remove-label", self.github_needs_human_label]
                )
            else:
                self.gh_ok(["issue", "edit", "-R", self.github_repo, issue, "--add-label", self.github_dag_blocked_label])
                self.gh_ok(["issue", "edit", "-R", self.github_repo, issue, "--remove-label", self.github_queue_label])

    def count_done_nodes(self) -> int:
        total = 0
        for _, issue, _ in self.node_rows():
            if self.issue_is_done(issue):
                total += 1
        return total

    def any_failed_node(self) -> str:
        for _, issue, _ in self.node_rows():
            if self.issue_is_failed(issue):
                return issue
        return ""

    def run(self) -> int:
        self.plog(f"run_id={self.run_id} root_issue={self.root_issue} repo={self.github_repo}")
        self.create_labels_if_needed()

        root_title = self.gh_text(
            ["issue", "view", "-R", self.github_repo, str(self.root_issue), "--json", "title", "--jq", ".title"]
        )
        root_url = self.gh_text(
            ["issue", "view", "-R", self.github_repo, str(self.root_issue), "--json", "url", "--jq", ".url"]
        )
        _ = root_title  # kept for future extensions

        nodes = self.load_plan_nodes()
        mapping = self.create_node_issues(nodes, root_url)
        node_total = len(mapping)
        if node_total <= 0:
            raise OrchestratorError("no DAG nodes created")

        root_comment_file = self.run_dir / "root-summary-comment.md"
        lines = [
            "<!-- orchestrator:dag-run:v1 -->",
            f"dag_run_id: {self.run_id}",
            f"node_total: {node_total}",
            "",
            "DAG Issue の作成が完了しました。",
            "",
        ]
        for node_id, issue, depends_csv in self.node_rows():
            lines.append(f"- node `{node_id}`: #{issue} (depends_on: {depends_csv or 'none'})")
        root_comment_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.gh_ok(["issue", "comment", "-R", self.github_repo, str(self.root_issue), "--body-file", str(root_comment_file)])

        prev_done = -1
        stagnant_rounds = 0
        for round_no in range(1, self.max_rounds + 1):
            self.refresh_dependency_gates()

            failed_issue = self.any_failed_node()
            if failed_issue:
                self.plog(f"pipeline failed: node issue #{failed_issue} has failed/needs-human label")
                return 2

            done_count = self.count_done_nodes()
            self.plog(f"round={round_no} done={done_count}/{node_total}")
            if done_count >= node_total:
                self.plog("pipeline completed: all DAG nodes done")
                return 0

            round_run_id = f"{self.run_id}-r{round_no}"
            cp = run_cmd(
                [
                    sys.executable,
                    str(self.manager_script),
                    "--config",
                    str(self.config_path),
                    "--mode",
                    "github",
                    "--run-id",
                    round_run_id,
                    "--limit",
                    str(self.dispatch_limit),
                ],
                check=False,
                capture_output=True,
            )
            if cp.returncode != 0:
                (self.run_dir / f"round-{round_no}.stderr.log").write_text(
                    (cp.stdout or "") + (cp.stderr or ""),
                    encoding="utf-8",
                )
                self.plog(f"manager dispatch failed at round={round_no} rc={cp.returncode}")
                return cp.returncode

            done_after = self.count_done_nodes()
            if done_after == prev_done:
                stagnant_rounds += 1
            else:
                stagnant_rounds = 0
            prev_done = done_after

            if stagnant_rounds >= 5:
                self.plog(f"pipeline stalled: no progress for 5 rounds (done={done_after}/{node_total})")
                return 3

        self.plog(f"pipeline stopped: max rounds exceeded ({self.max_rounds})")
        return 4


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DAG issues and execute them through orchestrator manager.")
    parser.add_argument("--root-issue", type=int, required=True, help="root issue number")
    parser.add_argument(
        "--config",
        default=".codex/orchestrator/config.toml",
        help="path to config TOML (default: .codex/orchestrator/config.toml)",
    )
    parser.add_argument("--run-id", default=None, help="fixed run id (default: dag-<timestamp>)")
    parser.add_argument("--plan-file", default=None, help="path to prebuilt plan JSON")
    parser.add_argument("--max-rounds", type=int, default=30, help="maximum manager rounds")
    parser.add_argument("--dispatch-limit", type=int, default=50, help="limit passed to manager dispatch")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    return DAGPipeline(args).run()


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except OrchestratorError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
