# Data model: English/Vietnamese internationalization

## Language resource

A version-controlled UTF-8 JSON document stored at `config/language/<code>.json`.

| Field | Type | Rules |
|---|---|---|
| `code` | string | Supported normalized code: `EN` or `VI`. |
| `name` | string | Localized language name used by the selector. |
| `translations` | object | Flat dot-delimited static UI keys. EN is the fallback baseline; VI has identical required keys. |
| `output_mappings.columns` | object | Raw output column/field name → translation key. |
| `output_mappings.values` | object | Raw field → exact controlled raw value → translation key. Field scoping is mandatory to protect user/free-form values. |

## Translation key

| Attribute | Rule |
|---|---|
| Format | Lowercase dot-delimited stable identifier, e.g. `common.download_csv`, `navigation.salary_prediction`, `status.completed`. |
| Ownership | Defined in language resources, not page modules. |
| Values | May contain named placeholders, e.g. `{run_id}`. |
| Fallback | Resolve in active language, then EN; return a non-user-breaking fallback for absent unknown keys while validation reports the defect. |

## Language session state

| Attribute | Value |
|---|---|
| Key | `app_language`, owned by `src/components/language.py`; pure `Translator` receives language explicitly. |
| Lifecycle | Initialized before authentication/shared navigation, updated by the selector, retained for the active Streamlit session. |
| Validation | Normalize and allow only registered language codes. Invalid/missing values resolve to `EN`. |

## Output localization mapping

| Input | Output | Invariants |
|---|---|---|
| Known DataFrame column | Localized display column | Raw DataFrame is unchanged. |
| Known controlled cell value/status/category | Localized display value | Only exact registered mapping matches are translated. |
| Known JSON metadata label/value | Localized rendered text | Artifact JSON on disk is unchanged. |
| Unknown/free-form/user/generated value | Original value | No guessing, translation API call, normalization, or data loss. |
| Numeric/date/ID/path/hash | Original value | Formatting logic may remain separate but is not language translation unless explicitly keyed. |

## Relationships

```text
Language selector -> validated language session state -> translation service
Language resource --------------------------------------^ 
translation service -> static UI resolver -> Streamlit entrypoint/components/pages
translation service -> output mapping resolver -> display-only table/chart/metadata copies
raw outputs/artifacts ---------------------------------> output mapping resolver (read-only)
```
