#!/usr/bin/env python3
"""
Create GitHub parent/child issues from a structured plan.

Features:
- Duplicate title check (stop before create)
- Japanese issue body rendering
- Parent/child creation
- Sub-issue linking
- Intra-parent dependency linking
- Title-only logging
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

URL_ISSUE_RE = re.compile(r"https://github\.com/[^/\s]+/[^/\s]+/issues/(\d+)")
NORMALIZE_RE = re.compile(r"[^\wぁ-んァ-ヶ一-龥]+", re.UNICODE)
STRICT_SCENARIO_KEYWORDS = (
    "migration",
    "migrate",
    "db",
    "database",
    "外部api",
    "外部 api",
    "認可",
    "権限",
    "oauth",
    "auth",
)
SPIKE_KEYWORDS = ("spike", "調査", "検証", "poc", "research", "investigation")
GENERIC_SCENARIO_NAMES = {
    "成果物が定義どおり作成される",
    "前提不足時の扱いが明確である",
    "完了判定を満たす",
}
GENERIC_TODO_PHRASES = (
    "実装または仕様化を実施する",
    "受け入れ条件を満たす証跡を添えてprがマージされる",
)

CANONICAL_LABEL_ALIASES: dict[str, list[str]] = {
    "bug": ["bug", "type:bug", "kind/bug"],
    "feature": ["feature", "enhancement", "type:feature"],
    "chore": ["chore", "task", "maintenance", "type:chore"],
    "docs": ["docs", "documentation", "type:docs"],
    "api": ["api", "backend", "server"],
    "ui": ["ui", "frontend", "client"],
    "db": ["db", "database", "migration"],
    "test": ["test", "qa", "testing"],
    "security": ["security"],
    "performance": ["performance", "perf"],
    "infra": ["infra", "devops", "ops"],
    "spike": ["spike", "research", "investigation"],
    "release": ["release", "deploy"],
}


@dataclass(frozen=True)
class CreatedIssue:
    id: int
    number: int
    url: str
    title: str
    body_file: str


class CommandError(RuntimeError):
    pass


def run_command(cmd: list[str]) -> str:
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "unknown error"
        raise CommandError(f"Command failed: {' '.join(cmd)}\n{detail}")
    return result.stdout.strip()


def run_json(cmd: list[str]) -> Any:
    out = run_command(cmd)
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise CommandError(f"Expected JSON output from command: {' '.join(cmd)}") from exc


def normalize_title(title: str) -> str:
    text = NORMALIZE_RE.sub("", title).lower()
    return text


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9ぁ-んァ-ヶ一-龥]+", "-", text.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "issue"


def ensure_repo(repo_override: str | None, plan_repo: str | None) -> str:
    if repo_override:
        return repo_override
    if plan_repo:
        return plan_repo
    try:
        return run_command(
            ["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"]
        )
    except CommandError as exc:
        raise CommandError(
            "Failed to detect repository. Run in a git repo with GitHub remote "
            "or pass --repo OWNER/REPO."
        ) from exc


def read_plan(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Plan file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Plan root must be a JSON object.")
    if "parents" not in data or not isinstance(data["parents"], list) or not data["parents"]:
        raise ValueError("Plan must contain non-empty 'parents' array.")
    return data


def ensure_non_empty_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string.")
    value = " ".join(value.split()).strip()
    if not value:
        raise ValueError(f"{field} must not be empty.")
    return value


def ensure_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty array.")
    out: list[str] = []
    for idx, item in enumerate(value):
        out.append(ensure_non_empty_str(item, f"{field}[{idx}]"))
    return out


def ensure_scenarios(value: Any, field: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty array.")
    out: list[dict[str, str]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{idx}] must be an object.")
        scenario = ensure_non_empty_str(item.get("scenario", ""), f"{field}[{idx}].scenario")
        given = ensure_non_empty_str(item.get("given", ""), f"{field}[{idx}].given")
        when = ensure_non_empty_str(item.get("when", ""), f"{field}[{idx}].when")
        then = ensure_non_empty_str(item.get("then", ""), f"{field}[{idx}].then")
        out.append({"scenario": scenario, "given": given, "when": when, "then": then})
    return out


def normalize_issue(raw: dict[str, Any], context: str, default_ref: str) -> dict[str, Any]:
    raw_todos = raw.get("todos", raw.get("tasks"))
    issue: dict[str, Any] = {
        "ref": ensure_non_empty_str(raw.get("id", default_ref), f"{context}.id"),
        "title": ensure_non_empty_str(raw.get("title"), f"{context}.title"),
        "background": ensure_non_empty_str(raw.get("background"), f"{context}.background"),
        "purpose": ensure_non_empty_str(raw.get("purpose"), f"{context}.purpose"),
        "scope": ensure_string_list(raw.get("scope"), f"{context}.scope"),
        "acceptance_criteria": ensure_scenarios(
            raw.get("acceptance_criteria"), f"{context}.acceptance_criteria"
        ),
        "todos": ensure_string_list(raw_todos, f"{context}.todos"),
        "labels": [],
        "depends_on": [],
        "questions": [],
        "out_of_scope": [],
    }
    raw_labels = raw.get("labels", [])
    if raw_labels:
        issue["labels"] = ensure_string_list(raw_labels, f"{context}.labels")
    raw_depends_on = raw.get("depends_on", [])
    if raw_depends_on:
        issue["depends_on"] = ensure_string_list(raw_depends_on, f"{context}.depends_on")
    raw_questions = raw.get("questions", [])
    if raw_questions:
        issue["questions"] = ensure_string_list(raw_questions, f"{context}.questions")
    raw_out = raw.get("out_of_scope", [])
    if raw_out:
        issue["out_of_scope"] = ensure_string_list(raw_out, f"{context}.out_of_scope")
    return issue


def is_spike_issue(issue: dict[str, Any]) -> bool:
    text = " ".join(
        [
            issue["title"],
            issue["background"],
            issue["purpose"],
            " ".join(issue["scope"]),
            " ".join(issue["todos"]),
            " ".join(issue["labels"]),
        ]
    ).lower()
    return any(keyword in text for keyword in SPIKE_KEYWORDS)


def needs_strict_scenarios(issue: dict[str, Any]) -> bool:
    text = " ".join(
        [
            issue["title"],
            issue["background"],
            issue["purpose"],
            " ".join(issue["scope"]),
            " ".join(issue["todos"]),
            " ".join(issue["labels"]),
        ]
    ).lower()
    return any(keyword in text for keyword in STRICT_SCENARIO_KEYWORDS)


def validate_issue_specific_content(issue: dict[str, Any], context: str) -> None:
    for idx, scenario in enumerate(issue["acceptance_criteria"]):
        if scenario["scenario"] in GENERIC_SCENARIO_NAMES:
            raise ValueError(
                f"{context}.acceptance_criteria[{idx}].scenario uses generic template text: "
                f"'{scenario['scenario']}'."
            )

    for idx, todo in enumerate(issue["todos"]):
        todo_lower = todo.lower()
        for phrase in GENERIC_TODO_PHRASES:
            if phrase in todo_lower:
                raise ValueError(
                    f"{context}.todos[{idx}] uses generic template phrase: '{todo}'."
                )


def validate_non_duplicate_content(parents: list[dict[str, Any]]) -> None:
    criteria_map: dict[str, list[str]] = {}
    todo_map: dict[str, list[str]] = {}

    for p_idx, parent in enumerate(parents):
        p_ctx = f"parents[{p_idx}]({parent['title']})"
        criteria_key = json.dumps(parent["acceptance_criteria"], ensure_ascii=False, sort_keys=True)
        todo_key = json.dumps(parent["todos"], ensure_ascii=False)
        criteria_map.setdefault(criteria_key, []).append(p_ctx)
        todo_map.setdefault(todo_key, []).append(p_ctx)

        for c_idx, child in enumerate(parent["children"]):
            c_ctx = f"{p_ctx}.children[{c_idx}]({child['title']})"
            criteria_key = json.dumps(child["acceptance_criteria"], ensure_ascii=False, sort_keys=True)
            todo_key = json.dumps(child["todos"], ensure_ascii=False)
            criteria_map.setdefault(criteria_key, []).append(c_ctx)
            todo_map.setdefault(todo_key, []).append(c_ctx)

    duplicated_criteria = [refs for refs in criteria_map.values() if len(refs) > 1]
    duplicated_todos = [refs for refs in todo_map.values() if len(refs) > 1]
    if duplicated_criteria:
        refs = " | ".join(", ".join(group) for group in duplicated_criteria)
        raise ValueError(
            "Identical acceptance_criteria found across multiple issues. "
            f"Rewrite them per issue. Duplicates: {refs}"
        )
    if duplicated_todos:
        refs = " | ".join(", ".join(group) for group in duplicated_todos)
        raise ValueError(
            "Identical todos found across multiple issues. "
            f"Rewrite them per issue. Duplicates: {refs}"
        )


def validate_scenario_minimum(issue: dict[str, Any], context: str) -> None:
    actual = len(issue["acceptance_criteria"])
    if is_spike_issue(issue):
        minimum = 1
    elif needs_strict_scenarios(issue):
        minimum = 3
    else:
        minimum = 2
    if actual < minimum:
        raise ValueError(
            f"{context}.acceptance_criteria must contain at least {minimum} scenarios, got {actual}."
        )


def normalize_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
    parents: list[dict[str, Any]] = []
    raw_parents = raw_plan["parents"]
    for p_idx, raw_parent in enumerate(raw_parents):
        if not isinstance(raw_parent, dict):
            raise ValueError(f"parents[{p_idx}] must be an object.")
        parent = normalize_issue(raw_parent, f"parents[{p_idx}]", default_ref=f"parent-{p_idx + 1}")

        raw_children = raw_parent.get("children", [])
        if not isinstance(raw_children, list):
            raise ValueError(f"parents[{p_idx}].children must be an array.")
        if len(raw_children) > 10:
            raise ValueError(f"parents[{p_idx}] has {len(raw_children)} children; maximum is 10.")

        children: list[dict[str, Any]] = []
        ref_set: set[str] = set()
        for c_idx, raw_child in enumerate(raw_children):
            if not isinstance(raw_child, dict):
                raise ValueError(f"parents[{p_idx}].children[{c_idx}] must be an object.")
            default_ref = f"child-{p_idx + 1}-{c_idx + 1}"
            child = normalize_issue(
                raw_child,
                f"parents[{p_idx}].children[{c_idx}]",
                default_ref=default_ref,
            )
            ref = child["ref"]
            if ref in ref_set:
                raise ValueError(
                    f"parents[{p_idx}] has duplicate child id '{ref}'. Child ids must be unique."
                )
            ref_set.add(ref)
            children.append(child)

        child_keys = {child["ref"] for child in children}
        child_keys.update(normalize_title(child["title"]) for child in children)
        for child in children:
            for dep in child["depends_on"]:
                dep_key = dep if dep in child_keys else normalize_title(dep)
                if dep_key not in child_keys:
                    raise ValueError(
                        f"Unknown dependency '{dep}' in child '{child['title']}'. "
                        "Use child id or exact child title."
                    )
                if dep == child["ref"] or normalize_title(dep) == normalize_title(child["title"]):
                    raise ValueError(f"Child '{child['title']}' cannot depend on itself.")

        validate_scenario_minimum(parent, f"parents[{p_idx}]")
        for c_idx, child in enumerate(children):
            validate_scenario_minimum(child, f"parents[{p_idx}].children[{c_idx}]")

        parent["children"] = children
        parents.append(parent)

    return {"repo": raw_plan.get("repo"), "parents": parents}


def render_issue_markdown(issue: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("## 背景")
    lines.append(issue["background"])
    lines.append("")
    lines.append("## 目的")
    lines.append(issue["purpose"])
    lines.append("")
    lines.append("## スコープ")
    for item in issue["scope"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 受け入れ条件 (Gherkin)")
    for idx, scenario in enumerate(issue["acceptance_criteria"], start=1):
        lines.append(f"### Scenario {idx}: {scenario['scenario']}")
        lines.append(f"Given {scenario['given']}")
        lines.append(f"When {scenario['when']}")
        lines.append(f"Then {scenario['then']}")
        lines.append("")
    lines.append("## 実装タスク")
    for task in issue["tasks"]:
        lines.append(f"- [ ] {task}")
    if issue["out_of_scope"]:
        lines.append("")
        lines.append("## Out of scope")
        for item in issue["out_of_scope"]:
            lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def extract_issue_number_from_url(output: str) -> int:
    match = URL_ISSUE_RE.search(output)
    if not match:
        raise CommandError(f"Could not parse issue URL from output: {output}")
    return int(match.group(1))


def fetch_issue_details(repo: str, number: int) -> CreatedIssue:
    issue = run_json(["gh", "api", f"repos/{repo}/issues/{number}"])
    return CreatedIssue(
        id=int(issue["id"]),
        number=int(issue["number"]),
        url=issue["html_url"],
        title=issue["title"],
        body_file="",
    )


def fetch_existing_labels(repo: str) -> dict[str, str]:
    out = run_command(
        ["gh", "api", f"repos/{repo}/labels", "--paginate", "--jq", ".[].name"]
    ).splitlines()
    return {label.lower(): label for label in out if label.strip()}


def infer_canonical_labels(issue: dict[str, Any], is_parent: bool) -> list[str]:
    text = " ".join(
        [
            issue["title"],
            issue["background"],
            issue["purpose"],
            " ".join(issue["scope"]),
            " ".join(issue["tasks"]),
        ]
    ).lower()
    selected: list[str] = []

    def add(label: str) -> None:
        if label not in selected:
            selected.append(label)

    if any(k in text for k in ("bug", "不具合", "障害", "回帰")):
        add("bug")
    if any(k in text for k in ("doc", "ドキュメント", "仕様書", "readme")):
        add("docs")
    if any(k in text for k in ("security", "脆弱性", "認可", "認証", "権限")):
        add("security")
    if any(k in text for k in ("performance", "perf", "遅い", "高速化", "性能")):
        add("performance")
    if any(k in text for k in ("infra", "devops", "ci", "cd", "運用")):
        add("infra")
    if any(k in text for k in ("spike", "調査", "検証", "poc", "research")):
        add("spike")
    if any(k in text for k in ("release", "deploy", "rollout", "rollback", "リリース")):
        add("release")
    if any(k in text for k in ("db", "database", "migration", "schema", "テーブル")):
        add("db")
    if any(k in text for k in ("api", "backend", "endpoint")):
        add("api")
    if any(k in text for k in ("ui", "frontend", "画面", "表示")):
        add("ui")
    if any(k in text for k in ("test", "テスト", "qa")):
        add("test")

    if not selected:
        add("feature" if is_parent else "chore")
    return selected


def resolve_labels(
    issue: dict[str, Any],
    existing_labels: dict[str, str],
    is_parent: bool,
) -> list[str]:
    canonical = infer_canonical_labels(issue, is_parent=is_parent)
    for raw in issue["labels"]:
        raw_lower = raw.lower()
        if raw_lower in existing_labels and raw_lower not in canonical:
            canonical.append(raw_lower)

    resolved: list[str] = []
    for key in canonical:
        if key in existing_labels:
            resolved.append(existing_labels[key])
            continue
        aliases = CANONICAL_LABEL_ALIASES.get(key, [key])
        for alias in aliases:
            alias_lower = alias.lower()
            if alias_lower in existing_labels:
                resolved.append(existing_labels[alias_lower])
                break

    # Deduplicate while preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for label in resolved:
        if label not in seen:
            deduped.append(label)
            seen.add(label)
    return deduped


def fetch_existing_issues(repo: str, limit: int = 1000) -> list[dict[str, Any]]:
    return run_json(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "all",
            "--limit",
            str(limit),
            "--json",
            "number,title,url",
        ]
    )


def find_duplicate_candidates(
    planned_titles: list[str],
    existing_issues: list[dict[str, Any]],
    threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    duplicates: dict[str, list[dict[str, Any]]] = {}

    for title in planned_titles:
        norm_title = normalize_title(title)
        scored: list[tuple[float, dict[str, Any]]] = []
        for issue in existing_issues:
            existing_title = issue.get("title", "")
            norm_existing = normalize_title(existing_title)
            if not norm_existing:
                continue
            ratio = difflib.SequenceMatcher(None, norm_title, norm_existing).ratio()
            exact = norm_title == norm_existing
            contains = norm_title and norm_title in norm_existing
            if exact or contains or ratio >= threshold:
                scored.append((ratio, issue))

        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            duplicates[title] = [
                {
                    "number": item["number"],
                    "title": item["title"],
                    "url": item["url"],
                    "similarity": round(score, 3),
                }
                for score, item in scored[:3]
            ]
    return duplicates


def write_body_file(issue: dict[str, Any], temp_dir: str | None) -> str:
    prefix = f"gh-req-issue-{slugify(issue['title'])[:24]}-"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=prefix,
        suffix=".md",
        delete=False,
        dir=temp_dir,
    ) as tmp:
        tmp.write(render_issue_markdown(issue))
        return tmp.name


def create_issue(
    repo: str,
    issue: dict[str, Any],
    labels: list[str],
    temp_dir: str | None,
) -> CreatedIssue:
    body_file = write_body_file(issue, temp_dir=temp_dir)
    cmd = [
        "gh",
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        issue["title"],
        "--body-file",
        body_file,
    ]
    for label in labels:
        cmd.extend(["--label", label])

    output = run_command(cmd)
    number = extract_issue_number_from_url(output)
    created = fetch_issue_details(repo, number)
    return CreatedIssue(
        id=created.id,
        number=created.number,
        url=created.url,
        title=created.title,
        body_file=body_file,
    )


def add_sub_issue(repo: str, parent_number: int, child_issue_id: int) -> None:
    run_command(
        [
            "gh",
            "api",
            "-X",
            "POST",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "X-GitHub-Api-Version: 2022-11-28",
            f"repos/{repo}/issues/{parent_number}/sub_issues",
            "-F",
            f"sub_issue_id={child_issue_id}",
        ]
    )


def add_dependency(repo: str, blocked_issue_number: int, blocking_issue_id: int) -> None:
    run_command(
        [
            "gh",
            "api",
            "-X",
            "POST",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "X-GitHub-Api-Version: 2022-11-28",
            f"repos/{repo}/issues/{blocked_issue_number}/dependencies/blocked_by",
            "-F",
            f"issue_id={blocking_issue_id}",
        ]
    )


def append_title_log(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{title}\n")


def collect_planned_titles(plan: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    for parent in plan["parents"]:
        titles.append(parent["title"])
        for child in parent["children"]:
            titles.append(child["title"])
    return titles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create issue graph from requirements plan JSON.")
    parser.add_argument("--plan", required=True, help="Path to plan JSON file")
    parser.add_argument("--repo", help="Override repository in OWNER/REPO format")
    parser.add_argument(
        "--title-log",
        default="/tmp/gh-requirements-to-issues-titles.log",
        help="Path to title-only log file",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for duplicate detection (default: 0.9)",
    )
    parser.add_argument(
        "--temp-dir",
        help="Directory for temporary issue body files (default: system tmp dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate, duplicate-check, and print plan without creating issues",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    raw_plan = read_plan(Path(args.plan))
    normalized_plan = normalize_plan(raw_plan)
    repo = ensure_repo(args.repo, normalized_plan.get("repo"))

    existing_issues = fetch_existing_issues(repo)
    planned_titles = collect_planned_titles(normalized_plan)
    duplicates = find_duplicate_candidates(
        planned_titles, existing_issues, threshold=args.duplicate_threshold
    )
    if duplicates:
        print(
            json.dumps(
                {
                    "status": "duplicate_detected",
                    "repo": repo,
                    "duplicates": duplicates,
                    "message": "Stop creation and ask user before retrying.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    existing_labels = fetch_existing_labels(repo)

    if args.dry_run:
        preview: list[dict[str, Any]] = []
        for parent in normalized_plan["parents"]:
            parent_labels = resolve_labels(parent, existing_labels, is_parent=True)
            preview_parent = {
                "title": parent["title"],
                "labels": parent_labels,
                "children": [],
            }
            for child in parent["children"]:
                child_labels = resolve_labels(child, existing_labels, is_parent=False)
                preview_parent["children"].append(
                    {
                        "id": child["ref"],
                        "title": child["title"],
                        "labels": child_labels,
                        "depends_on": child["depends_on"],
                    }
                )
            preview.append(preview_parent)

        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "repo": repo,
                    "parents": preview,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    title_log = Path(args.title_log)
    summary_parents: list[dict[str, Any]] = []

    for parent in normalized_plan["parents"]:
        parent_labels = resolve_labels(parent, existing_labels, is_parent=True)
        parent_created = create_issue(repo, parent, parent_labels, temp_dir=args.temp_dir)
        append_title_log(title_log, parent_created.title)

        created_children: list[tuple[dict[str, Any], CreatedIssue]] = []
        ref_map: dict[str, CreatedIssue] = {}
        title_map: dict[str, CreatedIssue] = {}
        for child in parent["children"]:
            child_labels = resolve_labels(child, existing_labels, is_parent=False)
            child_created = create_issue(repo, child, child_labels, temp_dir=args.temp_dir)
            append_title_log(title_log, child_created.title)
            created_children.append((child, child_created))
            ref_map[child["ref"]] = child_created
            title_map[normalize_title(child["title"])] = child_created

        for _, child_created in created_children:
            add_sub_issue(repo, parent_created.number, child_created.id)

        seen_dependency_edges: set[tuple[int, int]] = set()
        for child, child_created in created_children:
            for raw_dep in child["depends_on"]:
                dep_issue = ref_map.get(raw_dep) or title_map.get(normalize_title(raw_dep))
                if dep_issue is None:
                    raise ValueError(
                        f"Dependency target not found: '{raw_dep}' for child '{child['title']}'."
                    )
                edge = (child_created.number, dep_issue.id)
                if edge in seen_dependency_edges:
                    continue
                seen_dependency_edges.add(edge)
                add_dependency(repo, child_created.number, dep_issue.id)

        parent_summary = {
            "title": parent_created.title,
            "number": parent_created.number,
            "url": parent_created.url,
            "body_file": parent_created.body_file,
            "children": [
                {
                    "id": child["ref"],
                    "title": child_created.title,
                    "number": child_created.number,
                    "url": child_created.url,
                    "body_file": child_created.body_file,
                    "depends_on": child["depends_on"],
                }
                for child, child_created in created_children
            ],
        }
        summary_parents.append(parent_summary)

    print(
        json.dumps(
            {
                "status": "created",
                "repo": repo,
                "title_log": str(title_log),
                "parents": summary_parents,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, CommandError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
