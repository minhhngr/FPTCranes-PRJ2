# Implementation Plan: English/Vietnamese internationalization

**Branch**: `009-add-i18n-support` | **Date**: 2026-09-21 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/009-add-i18n-support/spec.md`

## Summary

Add a small centralized i18n layer for English and Vietnamese. Initialize validated active-language session state and a shared sidebar selector in the Streamlit entrypoint; migrate every static visible string to translation keys; and localize known generated-output labels/values through display-only mapping adapters. Preserve raw evidence, calculations, downloads, artifact schema, and offline model boundaries.

## Technical Context

**Language/Version**: Python 3.13; Ruff target py312  
**Primary Dependencies**: Existing Streamlit, pandas, Plotly, pytest; no additional i18n dependency proposed  
**Storage**: UTF-8 JSON resources in `config/language/`; existing read-only `outputs/`, run workspaces, and artifacts  
**Testing**: pytest, Streamlit `AppTest`, focused pure translation/output-localization tests  
**Target Platform**: Linux-hosted Streamlit application  
**Project Type**: Python Streamlit multipage-style dashboard with offline ML pipeline  
**Performance Goals**: Resource loading is negligible and cached/centralized; no new output copies outside render-time display copies  
**Constraints**: No model fitting or artifact mutation in UI; all supported visible strings use keys; unknown dynamic output values pass through unchanged  
**Scale/Scope**: `streamlit.py`, `src/components/`, all `src/pages/` renderers/helpers, EN/VI resources, and affected AppTests/docs

## Constitution Check

| Gate | Design disposition |
|---|---|
| Data integrity | N/A to fitting/partitions. Localization is presentation-only and cannot change data frames used in analysis or inference. |
| Reproducibility and ML evidence | Existing outputs/artifacts are reused because no producer, schema, data, config, lock, or training contract changes. Raw evidence remains unmodified; only display copies are localized. |
| Verification | Add failing-first resource parity, resolver, passthrough, and output-display tests; then AppTests for EN/VI selection, navigation persistence, all pages, and role-specific views. |
| Streamlit boundary | Selector and display adapters live in UI/shared importable helpers; they do not call `run_pipeline`, fitting, tuning, or artifact writes. |
| Input validation | Restrict/normalize the session/widget language to registered codes and default invalid values to EN. Existing security-sensitive input validation remains intact. |
| Scientific and security claims | Translate wording only. Preserve metric values, units, raw downloads, partition/evidence meaning, and warnings. Do not localize secrets, paths, IDs, or free-form input. |
| Documentation | Document resource structure, output mapping boundary, and add-language process; update Streamlit content map as needed. |
| Graphify/Karpathy review | Queried the existing graph and inspected the entrypoint, shared components, pages, readers, and AppTests. One small i18n module plus JSON resources is simpler than a dependency or page-local conditions. |

## Project Structure

```text
config/
└── language/
    ├── en.json                         # EN static keys and output mappings
    └── vi.json                         # VI static keys and output mappings
src/
├── ai_job_market/
│   └── i18n.py                         # resource validation/loading, session validation, static/output resolvers
├── components/
│   ├── language.py                     # cached catalog, session language, shared selector
│   ├── auth.py                         # keyed login/status text
│   └── data_source.py                  # keyed workspace/upload/status text
└── pages/
    ├── common.py                       # display-only frame/JSON localization helpers
    ├── model_evidence.py               # keyed narrative/chart labels and output display adapters
    ├── model_training_presentation.py  # keyed audit guides/tables
    ├── training_validation_presentation.py # keyed compatible-evidence displays
    └── page01...page08                 # keyed static text and specialized dynamic displays
streamlit.py                             # selector initialization, localized navigation/sidebar
specs/009-add-i18n-support/
├── spec.md
├── research.md
├── data-model.md
├── contracts/i18n-resource-and-output-localization.md
├── quickstart.md
└── tasks.md
tests/
├── test_i18n.py                         # resource/resolver/output mapping unit coverage
└── test_model_ui_pages.py                # entrypoint EN/VI AppTest coverage
```

**Structure Decision**: Add one importable translation boundary in the existing package and reuse existing shared page helpers/readers. Do not change page routing, output producers, output schemas, Streamlit component architecture, or the offline pipeline.

## Implementation outline

1. Inventory all user-visible static strings and controlled output labels/values by renderer, including login and all role-available pages. Assign stable resource keys before modifying code.
2. Define/validate EN and VI resource parity and an explicit allowlist of raw output headers/values that are safe to localize.
3. Implement the smallest resource/resolver/session API with safe EN fallback and named interpolation; unit-test it before migration.
4. Initialize language state and render a shared sidebar selector in `streamlit.py` before login/navigation. Migrate authentication, workspace controls, and navigation.
5. Migrate pages/shared helpers incrementally, preserving raw computational frames and applying localized display copies only where tables/charts/metadata are rendered.
6. Update tests that currently assert English strings so unrelated behavior is tested in an explicit language or against resource-derived values. Add EN/VI coverage for all pages.
7. Document resources/mapping behavior, run quality gates, inspect the actual UI, and refresh Graphify after code changes.

## Complexity Tracking

No constitutional exception proposed. JSON resources and one translation module are sufficient for two initial languages; a third-party localization framework or artifact migration would add complexity without solving a demonstrated need.

## Implementation progress (2026-09-21)

First increment implemented; the feature is not complete. See `verification.md` and
`docs/I18N.md` for coverage and remaining gates. Static resource keys are flat dot-delimited
entries. Output mappings require a raw field context; arbitrary metadata is never recursively
translated. Pure display-copy and shallow metadata adapters live on `Translator` in
`src/ai_job_market/i18n.py`; `src/pages/common.py` consumes them. Streamlit-specific state
and caching live in `src/components/language.py` to keep the core independently testable.
Navigation stores stable page IDs and workspace selection stores raw paths, not translated
labels. The two existing AppTest navigation helpers were updated accordingly without
changing their assertions or inherited feature-008 edits.

## Repository-state note

The feature branch was created while inherited worktree changes from feature 008 are present. Do not commit, overwrite, or include those pre-existing changes in this feature. `AGENTS.md` is among those inherited modifications, so its current-plan reference is not edited during planning; update it only with maintainer direction when the competing feature work is reconciled.
