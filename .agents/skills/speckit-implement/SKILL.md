---
name: speckit-implement
description: Execute Spec Kit tasks only after user approval, marking progress in tasks.md and keeping spec artifacts synchronized with implementation changes.
compatibility: Requires spec-kit project structure with .specify/ directory
metadata:
  author: local
  source: adapted from .pi/prompts/speckit.implement.md
---

# Speckit Implement

Implement the approved feature from `tasks.md`.

## Purpose

This skill owns the coding stage of the workflow and enforces the living-spec rule.

## Preconditions

Run this only after:
- `tasks.md` exists
- the user explicitly approved the spec/plan/tasks package
- any required checklists are complete, or the user explicitly chose to proceed despite incomplete items

## Command

```text
/speckit.implement
```

## Required Inputs

Read from the active feature directory:
- `tasks.md` (required)
- `plan.md` (required)
- `spec.md` (required by workflow intent)
- `data-model.md` when present
- `contracts/` when present
- `research.md` when present
- `quickstart.md` when present
- `.specify/memory/constitution.md` when present

## Local Rules

- Respect `.specify/extensions.yml` hooks, especially `before_implement` and `after_implement`
- Follow task order and dependency markers
- Mark finished tasks as `[X]` in `tasks.md`
- Halt on blocking failures
- Report progress clearly

## Living Spec Requirement

If implementation changes reality, update the matching spec artifacts before continuing:
- scope or requirement change -> `spec.md`
- technical design change -> `plan.md`, `research.md`, `data-model.md`, `contracts/`, or `quickstart.md` as needed
- task order or completion state change -> `tasks.md`
- validation expectation change -> checklist files and verification notes

Never let code drift away from the approved spec package.

## Required Behavior

1. Check checklist status first
2. Load the full implementation context
3. Execute tasks phase by phase
4. Update `tasks.md` as tasks complete
5. Update spec artifacts when implementation reveals changes
6. Validate completion against the original spec and plan
7. Report completed work, failed work, and remaining work

## Approval Reminder

If approval is missing or ambiguous, stop and ask before implementing.
