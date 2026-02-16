---
name: gh-requirements-to-issues
description: Convert mixed requirements into Japanese GitHub parent/child issues and execute creation with gh CLI. Use when turning text, markdown files, URLs, or existing issue notes into sprint-goal parent issues, sub-issues, and intra-parent issue dependencies with duplicate checks.
---

# GH Requirements To Issues

## Overview

Convert requirements into executable issue graphs:
- Parent issue = sprint goal
- Child issues = one-day-max implementation units
- Child links = sub-issues + dependency edges

Always create issues in Japanese and execute `gh issue create` unless duplicate checks block creation.

## Workflow

1. Collect input requirements from text, markdown files, URLs, or existing issues.
2. Resolve target repo from current directory.
3. If repo cannot be resolved, ask the user before proceeding.
4. Ask clarifying questions when requirements are ambiguous.
5. Skip clarifying questions only when the user explicitly requests non-interactive execution.
6. Decompose requirements into parent/child issue graph with rules in `Task Decomposition Rules`.
7. Build a plan file that follows `references/plan-schema.md`.
8. Run `python3 scripts/create_issue_graph.py --plan <plan-file>`.
9. If duplicate candidates are reported, stop and ask the user how to proceed.

## Task Decomposition Rules

Apply these rules before creating any issue:

- Parent issue = sprint-goal function unit.
- Child issue size = as small as possible, never exceed one day of work.
- Child issue completion = PR merged.
- Child issue owner = single assignee principle.
- Allow FE/BE/DB in one child issue only when responsibility remains small.
- Split DB migration into a dedicated child issue.
- Include tests inside implementation child issues and follow TDD.
- Include docs updates inside implementation child issues when needed.
- Allow investigation/design-only issues.
- Split uncertain work into spike child issue first.
- Split API changes and UI changes into separate child issues.
- Create dependency links as completely as possible across children in the same parent.
- Keep each parent to at most 10 children.
- Split non-functional work (performance, observability, security) into dedicated child issues.
- Split release work into dedicated child issue; release unit follows parent issue.

## Issue Writing Rules

Apply these rules to every issue body:

- Japanese title and Japanese body.
- Required sections:
  - `## 背景`
  - `## 目的`
  - `## スコープ`
  - `## 受け入れ条件 (Gherkin)`
  - `## 実装タスク`
- Optional section:
  - `## Out of scope`
- Make acceptance criteria concrete and measurable.
- Write acceptance criteria in Gherkin style (`Given` / `When` / `Then`).
- Ensure checklist task lines with `- [ ]`.

Scenario minimums:
- Spike/investigation child issue: at least 1 scenario.
- Normal implementation child issue: at least 2 scenarios (normal + edge/failure).
- Child issue that includes DB migration, external API integration, or authorization/permission: at least 3 scenarios.

## Sub-Issues And Dependencies

- Use sub-issues for parent-to-children decomposition.
- Consider parent completion equivalent to completion of all its child issues.
- Use dependencies only among child issues under the same parent when execution order matters.
- Do not auto-wire GitHub Project fields in this skill. Issue-to-project association is handled by repository-side automation.

## Duplicate Handling

- Always run duplicate checks before creating issues.
- If duplicates are detected, stop creation and show candidates to the user.
- Resume only after user instruction.

## Labels, Assignee, Milestone, Project

- Apply labels automatically via `scripts/create_issue_graph.py` and `references/label-mapping.md`.
- Leave assignee empty unless explicitly provided.
- Do not require due dates.
- Do not require milestone/project assignment in this skill.

## Execution Commands

Build and execute:
```bash
python3 scripts/create_issue_graph.py --plan /tmp/issue-plan.json
```

Dry run:
```bash
python3 scripts/create_issue_graph.py --plan /tmp/issue-plan.json --dry-run
```

Custom repo:
```bash
python3 scripts/create_issue_graph.py --plan /tmp/issue-plan.json --repo OWNER/REPO
```

The script:
- Creates temporary body files under `/tmp`.
- Logs created issue titles into `/tmp/gh-requirements-to-issues-titles.log`.
- Creates parent and child issues.
- Adds sub-issues.
- Adds child dependency edges (`blocked_by`).
- Stops before creation when duplicate candidates exist.

## References

- Plan format: `references/plan-schema.md`
- Label behavior: `references/label-mapping.md`
