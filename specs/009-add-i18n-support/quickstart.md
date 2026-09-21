# Quickstart: validate English/Vietnamese internationalization

## Prerequisites

- Work from branch `009-add-i18n-support`.
- Preserve inherited uncommitted feature-008 work; do not stage it with this feature.
- Use the project Python environment with Streamlit and pytest installed.
- Existing `outputs/` are sufficient. Do **not** run training for this feature.

## Automated validation

1. Run the i18n unit tests. They must verify resource loading, EN/VI key parity, interpolation, invalid-language fallback, known column/value localization, and unknown-value passthrough.
2. Run the entrypoint/AppTest suite. It must authenticate both roles, select EN and VI, navigate page views, and assert resource-derived labels/messages without exceptions.
3. Run the relevant full pytest suite and Ruff.
4. Run `git diff --check` and confirm no output/artifact producer files or generated evidence were changed.
5. Run `graphify update .` after code changes and record a clean/update result.

## Manual Streamlit smoke test

1. Start the existing app entrypoint with `streamlit run streamlit.py` using the project environment.
2. Sign in with the documented local demo account.
3. In the sidebar, select **English** and confirm login/status, data source, navigation, a page title, a chart/table label, and an evidence status use English.
4. Select **Tiếng Việt** and confirm those same elements rerender in Vietnamese without a server restart.
5. Navigate among pages 1–8 as admin and page 6 as standard user; verify the selection persists.
6. Inspect at least one normal output table and one compatible training/audit view. Confirm known headers/statuses localize, while an unknown/free-form value and raw download remain unchanged.
7. Confirm no UI interaction triggered processing, training, or output/artifact mutation.

## Add a future language

1. Copy the complete EN resource into `config/language/<new-code>.json`.
2. Translate all static keys and register the language metadata; retain output mappings unless a controlled value needs a language-specific rendered phrase.
3. Run resource-parity validation and add the new language to supported-language tests.
4. Do not edit individual page modules to add a language-specific branch.
