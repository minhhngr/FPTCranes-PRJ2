# Contract: i18n resource and output-localization boundary

## Resource location and minimum shape

Language resources reside exclusively in `config/language/`.

```json
{
  "code": "EN",
  "name": "English",
  "translations": {
    "common.unavailable": "Text unavailable",
    "common.download_csv": "Download CSV",
    "navigation.salary_prediction": "Salary prediction",
    "status.completed": "Completed",
    "status.failed": "Failed",
    "output.columns.mae_mean": "Mean MAE"
  },
  "output_mappings": {
    "columns": {"MAE_mean": "output.columns.mae_mean"},
    "values": {
      "status": {
        "completed": "status.completed",
        "failed": "status.failed"
      }
    }
  }
}
```

The exact complete key set is established by the EN resource. Every other supported resource must contain the same required key set and valid string values.

## Resolver behavior

| Operation | Input | Required behavior |
|---|---|---|
| Static translation | language code, translation key, optional named values | Resolve active resource, then EN fallback. Named placeholders are formatted safely. |
| Language validation | candidate session/widget value | Normalize and accept only registered language codes; use `EN` for invalid/missing input. |
| Column localization | raw DataFrame/display column name | Translate only when registered under `output_mappings.columns`; otherwise return the original name. |
| Controlled value localization | raw field name and raw value | Translate only exact field-scoped registered mapping; without a field return the original value unchanged. |
| Frame localization | raw DataFrame | Return a display copy with mapped headers/cells; never mutate the input frame or change its row count, values outside mappings, dtypes used by logic, or download payload. |

## Compatibility and non-goals

- Existing output and artifact file schemas, filenames, hashes, provenance, and downloadable raw content are unchanged.
- Localization is a frontend consumer contract only; it must not run model training, preprocessing, inference, or artifact generation.
- The resolver does not translate user inputs, arbitrary narrative text, IDs, paths, numeric values, dates, names, or unknown backend values.
- A future language is added by supplying a valid resource with its code/name; discovery is automatic and page modules need no language-specific branches.
- Runtime implementation uses `Translator.text`, `column`, `value`, `frame`, and shallow `record`. Locale JSON uses flat dot-delimited keys. Mapping/header collisions preserve raw labels. Missing keys or interpolation values log the key and render `common.unavailable`.

## Failure behavior

- Resource/key parity defects fail automated validation with the language code and missing/invalid key path.
- A malformed active resource does not crash an end-user page when a valid EN resource is available; the runtime uses EN fallback and developers receive a diagnosable error/test failure.
- Unknown output values are successful pass-through values, not errors.
