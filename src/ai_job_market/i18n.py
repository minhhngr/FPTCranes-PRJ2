"""Pure, field-scoped presentation translations; never modifies source evidence.

Resources are trusted, versioned JSON. Strict loading is a release gate; tolerant
loading uses English for a defective secondary locale and logs the defect. No
active language is stored globally: each Translator belongs to one render/session.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

import pandas as pd

RESOURCE_DIR = Path(__file__).resolve().parents[2] / "config" / "language"
LOGGER = logging.getLogger(__name__)


class CatalogError(ValueError):
    """A resource is missing, malformed, or inconsistent with the English contract."""


def _placeholders(text: str) -> set[str]:
    fields = set()
    for _, field, _, conversion in Formatter().parse(text):
        if field is not None:
            if not field.isidentifier() or conversion not in (None, "s", "r"):
                raise ValueError("Only named placeholders are supported")
            fields.add(field)
    return fields


def _read_resource(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("Expected an object")
        code = obj.get("code")
        if not isinstance(code, str) or not re.fullmatch(r"[A-Z]{2,8}", code):
            raise ValueError("Invalid language code")
        if path.stem != code.lower():
            raise ValueError("Filename must match the lowercase language code")
        if not isinstance(obj.get("name"), str) or not obj["name"].strip():
            raise ValueError("Missing language name")
        translations = obj.get("translations")
        if not isinstance(translations, dict) or not translations:
            raise ValueError("Missing translations object")
        for key, text in translations.items():
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid translation: {key}")
            try:
                _placeholders(text)
            except ValueError as exc:
                raise ValueError(f"Invalid placeholder in {key}: {exc}") from exc
        mappings = obj.get("output_mappings")
        if not isinstance(mappings, dict) or set(mappings) != {"columns", "values"}:
            raise ValueError("Expected output_mappings.columns and output_mappings.values")
        groups = [mappings["columns"]]
        if not isinstance(mappings["values"], dict):
            raise ValueError("Expected field-scoped output values")
        groups.extend(mappings["values"].values())
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError("Expected mapping object")
            for raw, key in group.items():
                if not isinstance(raw, str) or not isinstance(key, str) or key not in translations:
                    raise ValueError(f"Invalid output mapping: {raw} -> {key}")
        return obj
    except (OSError, ValueError, TypeError) as exc:
        raise CatalogError(f"{path.name}: {exc}") from exc


@dataclass(frozen=True)
class Catalog:
    """Validated resources; callers must treat these dictionaries as read-only."""

    resources: dict[str, dict]

    @property
    def languages(self) -> dict[str, str]:
        return {code: data["name"] for code, data in self.resources.items()}

    def normalize(self, language: Any) -> str:
        code = language.strip().upper() if isinstance(language, str) else "EN"
        return code if code in self.resources else "EN"


def load_catalog(directory: Path = RESOURCE_DIR, *, strict: bool = True) -> Catalog:
    """Discover locales without registration code; validate parity and placeholders.

    English must always be valid. With strict=False a defective secondary locale
    is excluded, making requests for that locale fall back to English.
    """
    directory = Path(directory)
    default = _read_resource(directory / "en.json")
    if "common.unavailable" not in default["translations"]:
        raise CatalogError("en.json: missing common.unavailable")
    resources = {"EN": default}
    for path in sorted(directory.glob("*.json")):
        if path.name == "en.json":
            continue
        try:
            obj = _read_resource(path)
            baseline, values = default["translations"], obj["translations"]
            missing, extra = baseline.keys() - values.keys(), values.keys() - baseline.keys()
            if missing or extra:
                raise CatalogError(
                    f"{obj['code']}: missing {sorted(missing)}, extra {sorted(extra)}"
                )
            for key in baseline:
                if _placeholders(baseline[key]) != _placeholders(values[key]):
                    raise CatalogError(f"{obj['code']}: placeholder mismatch for {key}")
            if obj["output_mappings"] != default["output_mappings"]:
                raise CatalogError(f"{obj['code']}: output mappings differ from EN")
            resources[obj["code"]] = obj
        except CatalogError:
            if strict:
                raise
            LOGGER.exception("Ignoring invalid language resource: %s", path.name)
    return Catalog(resources)


class Translator:
    """Resolve keys and explicit output fields using one immutable language choice."""

    def __init__(self, catalog: Catalog, language: Any = "EN"):
        self.catalog = catalog
        self.language = catalog.normalize(language)

    def text(self, key: str, **values: Any) -> str:
        """Render named interpolation; missing keys/arguments log and show a safe label."""
        default = self.catalog.resources["EN"]["translations"]
        selected = self.catalog.resources[self.language]["translations"]
        template = selected.get(key, default.get(key))
        if template is not None:
            try:
                return template.format(**values)
            except (KeyError, ValueError, TypeError, IndexError):
                LOGGER.warning("Invalid translation arguments for %s", key)
        else:
            LOGGER.warning("Missing translation key: %s", key)
        return selected.get("common.unavailable", default["common.unavailable"])

    def column(self, field: Any) -> Any:
        """Localize a registered header, preserving all other labels exactly."""
        if not isinstance(field, str):
            return field
        key = self.catalog.resources["EN"]["output_mappings"]["columns"].get(field)
        return self.text(key) if key else field

    def value(self, value: Any, *, field: str | None = None) -> Any:
        """Exact, field-scoped allowlist lookup, never free-text substitution."""
        if not isinstance(value, str) or not isinstance(field, str):
            return value
        mapping = self.catalog.resources["EN"]["output_mappings"]["values"].get(field, {})
        key = mapping.get(value)
        return self.text(key) if key else value

    def record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Shallow metadata display projection; unknown/nested content stays raw."""
        labels = [self.column(field) for field in record]
        if len(set(labels)) != len(labels):
            labels = list(record)
        return {
            label: self.value(value, field=field)
            for label, (field, value) in zip(labels, record.items())
        }

    def frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a display copy; keep raw frames for computation and downloads.

        A translated header collision retains all original headers. Only mapped
        string columns are transformed; unrelated numeric/date dtypes stay intact.
        """
        result = frame.copy(deep=True)
        mappings = self.catalog.resources["EN"]["output_mappings"]["values"]
        for index, field in enumerate(frame.columns):
            if isinstance(field, str) and field in mappings:
                result.isetitem(
                    index, frame.iloc[:, index].map(lambda value: self.value(value, field=field))
                )
        headers = [self.column(field) for field in frame.columns]
        if len(set(headers)) == len(headers):
            result.columns = headers
        else:
            LOGGER.warning("Localized headers collide; preserving raw headers")
        return result
