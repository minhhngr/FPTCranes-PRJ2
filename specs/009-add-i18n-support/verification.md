# Verification — first implementation increment

**Date**: 2026-09-21  
**Status**: Partial implementation; full feature acceptance NOT met.

## Implemented

- Pure catalog validation, EN/VI lookup, safe interpolation/fallback, automatic locale
  discovery, and field-scoped output-value/header translation in `src/ai_job_market/i18n.py`.
- Display-only DataFrame copies and shallow metadata projections preserve raw inputs,
  unknown values, numeric columns, index, and downloadable evidence.
- Shared selector available before login; language state is session-local and validates
  against discovered resources. No language-specific state is kept in shared caches.
- Key-driven entrypoint, authentication, navigation, workspace/upload controls, common
  CSV download labels, and interpretation-card headings.
- Stable page IDs and workspace path identities prevent locale changes resetting them.
- Central initial resources under `config/language/` and maintenance guide `docs/I18N.md`.

## Test evidence

1. RED: `tests/test_i18n.py` failed collection before `ai_job_market.i18n` existed.
2. RED: all five original `tests/test_i18n_ui.py` tests failed before selector integration.
3. RED: metadata and common display tests failed before implementing their adapters.
4. GREEN: focused core/UI/common/model-page suite: **44 passed**.
5. Full `.venv/bin/pytest -q`: **228 passed, 1 skipped, 2 failed**.
6. Ruff passed on the changed implementation and new/updated release tests.
7. `git diff --check` passed.
8. SHA-256 comparison across **757 existing files** in `outputs/` and `artifacts/` found
   **zero changes** after tests. No pipeline/training run was started for this feature.

### Existing full-suite failures (not modified or hidden)

- `tests/test_release_contract.py::test_validation_report_passes`: existing
  `outputs/validation_report.json` already has `overall: FAIL`. It includes stale
  interpretation-card checks, missing archive evidence, and an old runtime import warning.
  Its bytes were unchanged during verification.
- `tests/test_release_contract.py::test_legacy_main_archive_manifest_packaged`: the
  inherited worktree already deleted `docs/legacy_reference/FPTCranes-PRJ2-main.rar`.
  The deletion was recorded before implementation. No archive was restored or recreated.

The new static resource extraction initially broke a source-text assertion expecting
English inside `data_source.py`; the assertion now checks the translation-key call and
central EN resource instead. Two pre-existing AppTest helpers now select stable page IDs
rather than translated labels. Their original behavior assertions remain unchanged.

### Tooling notes

The root file `streamlit.py` shadows the installed library in plain `python -c` invocations.
Use `.venv/bin/python -P` or the existing pytest/Streamlit console entrypoints. Installed
Streamlit is 1.64.0; local `streamlit docs st.selectbox` confirmed selector parameters.

`graphify update .` succeeded: 2,036 nodes, 3,719 edges, 137 communities. It warned that
six JSON sources produced no AST nodes and community labels need refreshing. This was a
code graph refresh only, not a semantic refresh of the new documents/resources.

No Chrome DevTools/browser tool was available. No screenshot, browser layout, or manual
smoke test is claimed. AppTest exercises the real entrypoint without a running server.

## Inherited worktree boundary

Before implementation, branch `009-add-i18n-support` contained extensive uncommitted
feature-008 changes and deleted archives. A diff/status backup was retained at
`/tmp/i18n-009-baseline/` for this session. No commit was made; configured automatic
commit hooks are disabled. Training producers, inherited presentation changes, and
`AGENTS.md` were left untouched, apart from the two minimal test navigation adjustments
noted above. New code does not overwrite the inherited work.

## Remaining work / release gates

- Classify `text-inventory.csv` candidates and complete the EN/VI resources for all pages.
  The inventory includes technical identifiers and formatting fragments, not just UI text.
- Migrate all page bodies, charts, dynamic narratives, specialized audit/training views,
  validation messages from generated outputs, and the complete controlled output vocabulary.
- Add failing-first coverage for each migration, all-page EN/VI render assertions, output
  fixtures, and a guard against hardcoded translatable text.
- Resolve inherited release checks separately with maintainer guidance; do not conceal them
  by editing generated validation evidence or recreating deleted archives.
- Complete browser/manual smoke testing, full regression verification, and final review.

**Do not describe the application as fully localized.** VI currently translates the shared
shell and selected common displays, while unmigrated page content remains in English.
