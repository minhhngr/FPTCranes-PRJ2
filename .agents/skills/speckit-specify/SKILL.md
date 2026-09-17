---
name: speckit-specify
description: Create the initial Spec Kit feature specification in this repository, including the feature directory, spec.md, and requirements checklist.
compatibility: Requires spec-kit project structure with .specify/ directory
metadata:
  author: local
  source: adapted from .pi/prompts/speckit.specify.md
---

# Speckit Specify

Create or update the initial feature specification for this repository.

## Purpose

This skill owns the first stage of the local Spec Kit workflow:

```text
specify -> research -> plan -> tasks -> user review -> define? -> implement
```

## Use When

Use when the user has described a feature or change and wants a formal spec before implementation.

## Command

```text
/speckit.specify <feature description>
```

## Expected Outputs

- `specs/<feature>/spec.md`
- `specs/<feature>/checklists/requirements.md`
- `.specify/feature.json`

## Local Rules

- Respect `.specify/extensions.yml` hooks, especially `before_specify` and `after_specify`
- Use `.specify/templates/spec-template.md` as the required structure
- Keep the spec focused on user needs and measurable outcomes, not implementation details
- Record assumptions explicitly
- Use at most 3 `[NEEDS CLARIFICATION: ...]` markers
- Validate the generated spec against the requirements checklist

## Required Behavior

1. Generate the feature directory under `specs/`
2. Write `spec.md` from the template
3. Write `checklists/requirements.md`
4. Persist the resolved feature directory to `.specify/feature.json`
5. Report:
   - `SPECIFY_FEATURE_DIRECTORY`
   - `SPEC_FILE`
   - checklist status
   - readiness for the next step

## Next Step

After this skill completes, continue with research if evidence gathering is needed:

```text
/speckit.assess.research slug=<slug>
```

Then continue to:

```text
/speckit.plan
```
