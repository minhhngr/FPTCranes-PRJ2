# Feature Specification: English/Vietnamese internationalization

**Feature Branch**: `009-add-i18n-support`  
**Created**: 2026-09-21  
**Status**: Approved; implementation in progress (shared UI/core only, full page migration pending)  
**Input**: User description: "Introduce centralized English and Vietnamese internationalization across the Streamlit application, including runtime language selection and output-value translation."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Switch the application language (Priority: P1)

As an authenticated application user, I can select English or Vietnamese from a shared settings location so that the currently displayed application immediately uses my selected language and retains it while I navigate within the active Streamlit session.

**Why this priority**: Selecting a language and preserving it across the application is the minimum viable i18n experience.

**Independent Test**: In a Streamlit `AppTest`, select each supported language, navigate between at least two available pages, and verify the selected language and representative shared UI text remain correct without restarting the app.

**Acceptance Scenarios**:

1. **Given** an authenticated user starts a new session, **When** the application loads, **Then** English is selected by default and the selector is available in the shared sidebar/settings area.
2. **Given** the user selects Vietnamese, **When** Streamlit reruns, **Then** the visible shared UI and current page use Vietnamese without a server restart.
3. **Given** Vietnamese is selected, **When** the user changes pages, **Then** Vietnamese remains selected and the newly opened page uses Vietnamese.
4. **Given** an invalid or unsupported language value reaches session state, **When** the application renders, **Then** it safely falls back to English rather than failing.

---

### User Story 2 - Read every application page in the selected language (Priority: P1)

As a user, I can read the static UI on every page available to my role in the selected language, including authentication, navigation, data-source controls, labels, descriptions, buttons, warnings, errors, tooltips, table headers, chart titles, and status indicators.

**Why this priority**: A selector is incomplete if page-level text remains hardcoded in another language.

**Independent Test**: Render every admin page and the standard-user prediction page in English and Vietnamese; verify there are no known untranslated UI literals and that representative controls, messages, charts, and tables use the selected resource.

**Acceptance Scenarios**:

1. **Given** English is selected, **When** any supported page renders, **Then** its supported static UI text is English.
2. **Given** Vietnamese is selected, **When** any supported page renders, **Then** its supported static UI text is Vietnamese.
3. **Given** a developer adds or changes a supported UI phrase, **When** the component renders, **Then** it obtains the phrase by translation key rather than embedding English or Vietnamese text in the component.

---

### User Story 3 - Localize known evidence/output values safely (Priority: P2)

As a user viewing generated evidence, I see known output field names, status values, categories, labels, and other application-controlled values in the selected language, while unfamiliar or user-provided content is shown unchanged.

**Why this priority**: The app is outputs-first; generated tables and metadata are a substantial part of what users read.

**Independent Test**: Load representative CSV/JSON evidence containing known columns and values plus an unknown value; verify the known values translate in both languages and the unknown value remains unchanged.

**Acceptance Scenarios**:

1. **Given** Vietnamese is selected and a rendered output table contains a known backend field or status, **When** it is displayed, **Then** the frontend shows its Vietnamese mapping.
2. **Given** English is selected and a rendered output table contains a known backend field or status, **When** it is displayed, **Then** the frontend shows its English mapping.
3. **Given** an output contains an unknown, free-form, or user-generated value, **When** it is displayed, **Then** the value is preserved exactly rather than translated, removed, or guessed.

---

### User Story 4 - Maintain translations centrally and extend them safely (Priority: P3)

As a maintainer, I can add or update language resources in one centralized location without editing individual Streamlit components, and resource validation identifies missing keys or malformed resources before a partial UI is released.

**Why this priority**: Central ownership prevents language drift and makes additional languages low-risk.

**Independent Test**: Add a test-only language resource or compare complete EN/VI key sets; verify resource loading, key parity, fallback behavior, and interpolation work without changing a page module.

**Acceptance Scenarios**:

1. **Given** a new supported language resource with the required keys is registered, **When** it is selected, **Then** shared translation logic can resolve its UI and output mappings without page-specific branching.
2. **Given** a required translation key is absent or a resource has an invalid structure, **When** resources are loaded or validated, **Then** tests fail with a precise missing-key/resource error.

### Edge Cases

- A missing, malformed, or unreadable language resource fails clearly in development/validation; runtime uses the documented default-language fallback only where a complete default resource is available.
- Unsupported or tampered session-state language values are treated as invalid input and safely reset to English.
- Missing output files and existing artifact/schema errors retain actionable UI behavior; localization must not hide or reinterpret their evidence meaning.
- Dynamic values are translated only when they match a controlled mapping. Names, uploaded values, IDs, numeric values, dates, file paths, credentials, free-form evidence, and unrecognized values remain unchanged.
- Interpolated text safely renders missing optional values without exposing formatting exceptions or raw placeholders to users.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide EN and VI as the initial supported languages, with English as the safe default.
- **FR-002**: The system MUST provide one shared language selector in the sidebar/settings area near other shared controls; it MUST be available before page content and persist throughout the active Streamlit session.
- **FR-003**: Changing the language selector MUST rerender all applicable shared and page content in the selected language without restarting the application.
- **FR-004**: All supported user-facing static text in `streamlit.py`, `src/components/`, and `src/pages/` MUST resolve via translation keys from centralized resources rather than hardcoded English or Vietnamese phrases.
- **FR-005**: Centralized language resources under `config/language/` MUST be the single source of truth for static UI phrases, supported output fields/values, and language metadata.
- **FR-006**: The translation layer MUST support named interpolation and a documented fallback for unknown keys; it MUST never show an exception to an end user due to a missing translation.
- **FR-007**: Before rendering application-generated CSV/JSON/artifact data from `outputs/` or compatible run workspaces, the frontend MUST map known application-controlled headers and values through the same translation layer.
- **FR-008**: The output translation mechanism MUST preserve unknown, free-form, user-provided, identifiers, numbers, and other unmapped values unchanged.
- **FR-009**: Adding a supported language MUST require only a resource addition/registration and resource validation, not page-specific language conditionals.
- **FR-010**: The system MUST validate every user-controlled language selection against the supported language enum and use English with a clear internal validation path for invalid values.
- **FR-011**: The system MUST document the language-resource contract, dynamic-output mapping boundary, and maintainer procedure for adding languages or terms.

### Constitutional Requirements *(mandatory)*

- **Data boundary**: N/A for model fitting and partitions. This is a presentation-only localization change; it must not alter source data, target values, feature policy, temporal partitions, preprocessing, training, inference, or scientific calculations.
- **Artifact contract**: Existing generated outputs/artifacts remain source-language evidence and are consumed read-only. The frontend presentation contract changes because known displayed headers/values are localized; no producer schema changes or retraining are required. Existing training outputs may be reused, provided localization preserves raw data and maps only known values at render time.
- **Streamlit boundary**: The selector and rendering adapters remain presentation logic. No language action may fit/retrain a model, mutate an output/artifact, or invoke the offline pipeline.
- **User input validation**: Validate the language selector/session-state value against `EN`/`VI`. Existing upload, workspace, credential, path, run-ID, and prediction validation behavior remains unchanged and must retain translated user-facing failures where applicable.
- **Scientific interpretation**: Localization may translate labels but must not change metric values, units, provenance, partition labels' scientific meaning, warning severity, or causal/production caveats.
- **Documentation impact**: Update the Streamlit content/documentation map and add resource/mapping maintenance guidance. Document changed shared helpers and output-localization behavior.
- **Verification evidence**: Tests must be written first for resource parity/loading, interpolation/fallback, output-value preservation/mapping, selector session persistence, and representative EN/VI renders through `AppTest`. Run the affected app test suite, full relevant pytest suite, Ruff, and Streamlit entrypoint checks.
- **Graphify/Karpathy review**: A Graphify query found the entrypoint, `components/auth.py`, `components/data_source.py`, eight page modules, common render helpers, and output readers as the affected presentation path. Keep the design to one small shared translation boundary and centralized resources; success is complete EN/VI rendering without page-local language branching.

### Key Entities *(include if feature involves data)*

- **Language resource**: A centralized EN or VI resource containing display metadata, static translation keys, and known output field/value mappings.
- **Translation key**: A stable, language-neutral identifier used by UI code to request a localized phrase, optionally with named interpolation values.
- **Output localization mapping**: The controlled mapping from raw application-generated headers/values to translation keys; it is applied only at presentation time.
- **Language session state**: The validated active-language code held for one Streamlit session and shared by all pages.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An authenticated `AppTest` can select EN or VI and retain the selected language after navigating between at least two pages in the same session.
- **SC-002**: Automated resource validation reports identical required static-key sets for EN and VI and rejects malformed/unsupported language resources.
- **SC-003**: Automated render coverage exercises every role-available page in EN and VI and verifies representative titles, navigation, controls, messages, chart/table labels, and status text are localized.
- **SC-004**: Automated output-localization tests show that all fixture-known headers/values map to the active language while fixture-unknown values remain byte-for-byte unchanged.
- **SC-005**: No supported application-visible string remains directly embedded in Streamlit entrypoint, component, or page code, except approved non-display identifiers/keys and raw evidence values intentionally preserved by the output-localization boundary.

## Assumptions

- English is the default language and canonical fallback for this initial release.
- “All user-facing text” covers the authenticated Streamlit experience, login screen, role-specific pages, shared controls, and visible generated evidence; it excludes source-code identifiers, raw downloadable file contents, user-provided data, and immutable provenance hashes.
- The current application has a shared sidebar but no implemented Light/Dark control; the selector will occupy the shared sidebar/settings area rather than introducing a separate toolbar.
- Translation resources will be versioned text data (for example JSON) under `config/language/`; no third-party i18n package is assumed necessary unless the approved implementation finds the existing stack cannot meet this contract.
- Existing generated artifacts remain valid because this feature changes presentation only; no model or offline-pipeline rerun is needed.
