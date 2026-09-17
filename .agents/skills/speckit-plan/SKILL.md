---
name: speckit-plan
description: Generate the local Spec Kit implementation plan and design artifacts from an approved feature spec.
compatibility: Requires spec-kit project structure with .specify/ directory
metadata:
  author: local
  source: adapted from .pi/prompts/speckit.plan.md
---

# Speckit Plan

Generate the implementation plan for the current feature.

## Purpose

This skill owns the planning stage of the local workflow:

```text
specify -> research -> plan -> tasks
```

## Preconditions

Run this only after:
- `spec.md` exists
- the spec is good enough to plan from
- any important evidence or constraints have been captured through research when needed

## Command

```text
/speckit.plan
```

## Expected Outputs

Inside the active feature directory:
- `plan.md`
- `research.md`
- `data-model.md`
- `quickstart.md`
- `contracts/` when applicable

## Local Rules

- Respect `.specify/extensions.yml` hooks, especially `before_plan` and `after_plan`
- Run the setup script described by Spec Kit to locate the active feature files
- Read `.specify/memory/constitution.md`
- Use `.specify/templates/plan-template.md`
- Resolve technical unknowns before implementation begins
- Update `AGENTS.md` plan reference if the workflow requires it

## Required Behavior

1. Load the current feature spec and plan template
2. Fill the technical context with concrete repository-relevant details
3. Produce Phase 0 research decisions where unknowns exist
4. Produce Phase 1 design artifacts
5. Re-check constitution constraints after design
6. Report generated files and any unresolved blockers

## Next Step

After plan generation, create executable tasks:

```text
/speckit.tasks
```
