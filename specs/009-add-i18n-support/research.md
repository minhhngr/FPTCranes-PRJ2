# Research: English/Vietnamese internationalization

## Repository evidence

| Question | Finding | Decision |
|---|---|---|
| Where does shared UI render? | `streamlit.py` authenticates, builds role-specific navigation, and renders the sidebar before dispatching a page. | Initialize validated language state and render the single selector in `streamlit.py` before shared/page UI. |
| How does navigation work? | The app uses a sidebar `st.radio` with module dispatch, not `st.navigation`; page modules expose `render(st, root, role)`. | Preserve current navigation; localize its labels through the shared translator without a navigation rewrite. |
| How does session state work? | Auth, workspace, and prediction controls use `st.session_state`; Streamlit reruns from the entrypoint. | Store one validated active-language code in session state and give the selector a stable key. |
| Where are static strings concentrated? | A scan found 333 Streamlit display/widget calls: Page 03 (88), Page 05 (57), Page 06 (39), Page 04 (38), plus shared pages/components/entrypoint. | Include every Streamlit entrypoint, component, and page in the text inventory; do not limit the change to the sidebar. |
| How is output data loaded? | `pages/common.py` loads CSV/JSON under `outputs/`; `model_evidence.py`, `model_training_presentation.py`, and `training_validation_presentation.py` load compatible evidence/audits; pages 01–08 render those structures. | Apply a presentation-only DataFrame/JSON label/value adapter at the shared output-reader/render boundary and use explicit mappings for specialized evidence readers. |
| Can existing evidence be reused? | The feature does not change training code, source data, output schema, dependency lock, or output producer. | Reuse existing outputs. Do not train; preserve raw artifacts and localize only values displayed in the frontend. |
| Is a Light/Dark control already present? | No source hit identifies an existing theme control. | Put language in the existing sidebar settings/control region; do not add unrelated theme infrastructure. |
| How are UI tests run? | Existing tests use `streamlit.testing.v1.AppTest` from `streamlit.py` and inject `auth_role`. | Extend that pattern for EN/VI renders, session persistence, and role-specific pages. |

## External/technical decisions

1. **Resource format**: use committed UTF-8 JSON resource files in `config/language/`, one per language, because the project already uses JSON configuration and no extra i18n dependency is needed for two languages.
2. **Key convention**: use dot-delimited keys such as `common.save`, `navigation.model_comparison`, `status.completed`, and `output.columns.mae_mean`. Page code imports a shared translator rather than carrying EN/VI conditionals.
3. **Translation API**: provide a small importable translation module that loads resources, validates supported codes/key parity, resolves a key with named interpolation, and falls back to the complete default EN resource. It must expose a separate safe resolver for known output fields/values.
4. **Dynamic-output boundary**: never mutate output files or generic data values. Translate only values registered in a resource mapping, including column headers, controlled enum/status/category values, and known JSON metadata labels. Return unregistered values unchanged.
5. **DataFrames and charts**: create localized display copies for table/chart labels at render time. Keep raw frames and raw column names for calculations, filtering, artifact lookup, and download payloads so logic and provenance remain stable.
6. **Scope control**: use native Streamlit selector/session state; no custom component, CSS, third-party i18n library, output-schema migration, or pipeline retraining is justified.

## Risks and mitigations

- **High string volume / missed literal**: build an inventory and add a test/review guard across `streamlit.py`, `src/components`, and `src/pages`; test every role-available page in both languages.
- **Raw names used as programmatic DataFrame keys**: localize only copies at display boundaries, never the source frame used for computation.
- **Free-form evidence accidentally translated**: mappings are allowlists; the resolver leaves unmapped content untouched.
- **English fallback masks incomplete VI resources**: enforce EN/VI required-key parity in tests/validation, while retaining fallback as a runtime safety net.
- **Existing tests assert English literals**: update them to assert translation keys/resource-derived expected strings or run explicitly in English where the behavior itself is unrelated to language.
