---
name: speckit-tasks
description: Generate dependency-ordered Spec Kit tasks.md for the active feature and stop for explicit user review before implementation.
compatibility: Requires spec-kit project structure with .specify/ directory
metadata:
  author: local
  source: adapted from .pi/prompts/speckit.tasks.md
---

# Speckit Tasks

Generate the implementation task breakdown for the current feature.

## Purpose

This skill owns the task-planning stage and the review gate before coding.

## Command

```text
/speckit.tasks
```

## Expected Output

- `specs/<feature>/tasks.md`

## Local Rules

- Respect `.specify/extensions.yml` hooks, especially `before_tasks` and `after_tasks`
- Use `.specify/templates/tasks-template.md`
- Read `plan.md` and `spec.md` as required inputs
- Read `research.md`, `data-model.md`, `contracts/`, and `quickstart.md` when present
- Organize tasks by user story when applicable
- Include exact file paths in each task
- Keep tasks executable by another agent without extra context

## Required Task Format

Every task must follow:

```text
- [ ] T001 [P?] [US?] Description with file path
```

## Mandatory Review Gate

After generating `tasks.md`, STOP.

Report back with:
- feature directory path
- `spec.md` path
- `plan.md` path
- `tasks.md` path
- total task count
- open risks or questions

Then wait for the user to choose one:
1. approve and continue
2. revise with agent
3. revise manually

Do not proceed to `/speckit.implement` without explicit user approval.

## Next Step

If approved:
- optional problem framing:

```text
/speckit.assess.define slug=<slug>
```

- implementation:

```text
/speckit.implement
```
