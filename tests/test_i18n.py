from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from ai_job_market.i18n import CatalogError, Translator, load_catalog

RESOURCES = Path(__file__).resolve().parents[1] / "config/language"


@pytest.fixture
def resources(tmp_path):
    shutil.copytree(RESOURCES, tmp_path / "language")
    return tmp_path / "language"


def change_resource(root, name, change):
    path = root / name
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_catalog_language_parity_and_interpolation():
    catalog = load_catalog(RESOURCES)
    assert set(catalog.languages) == {"EN", "VI"}
    en, vi = Translator(catalog, "EN"), Translator(catalog, "VI")
    assert en.text("auth.username") == "Username"
    assert vi.text("auth.username") == "Tên đăng nhập"
    assert vi.text("app.active_evidence", name="run_123") == "Bằng chứng đang dùng: `run_123`"


@pytest.mark.parametrize("value", [None, [], {}, 1, "../../secret", "", "unsupported"])
def test_invalid_language_is_safe(value):
    assert Translator(load_catalog(RESOURCES), value).language == "EN"


def test_language_normalization():
    assert Translator(load_catalog(RESOURCES), " vi ").language == "VI"


def test_missing_key_and_missing_interpolation_do_not_leak_placeholders(caplog):
    vi = Translator(load_catalog(RESOURCES), "VI")
    assert vi.text("does.not.exist") == vi.text("common.unavailable")
    assert vi.text("app.active_evidence") == vi.text("common.unavailable")
    assert "does.not.exist" in caplog.text
    assert "app.active_evidence" in caplog.text


def test_missing_translation_fails_validation_but_runtime_falls_back(resources):
    change_resource(resources, "vi.json", lambda d: d["translations"].pop("auth.username"))
    with pytest.raises(CatalogError, match="VI.*auth.username"):
        load_catalog(resources)
    catalog = load_catalog(resources, strict=False)
    assert Translator(catalog, "VI").text("auth.username") == "Username"


def test_malformed_secondary_resource_uses_english(resources):
    (resources / "vi.json").write_text("{", encoding="utf-8")
    with pytest.raises(CatalogError, match="vi.json"):
        load_catalog(resources)
    assert Translator(load_catalog(resources, strict=False), "VI").language == "EN"


def test_malformed_default_resource_is_actionable(resources):
    (resources / "en.json").write_text("{", encoding="utf-8")
    with pytest.raises(CatalogError, match="en.json"):
        load_catalog(resources, strict=False)


def test_placeholder_parity_is_validated(resources):
    change_resource(
        resources,
        "vi.json",
        lambda d: d["translations"].update({"app.active_evidence": "Bằng chứng: {wrong}"}),
    )
    with pytest.raises(CatalogError, match="app.active_evidence"):
        load_catalog(resources)


def test_only_known_field_scoped_values_are_translated():
    vi = Translator(load_catalog(RESOURCES), "VI")
    assert vi.value("completed", field="status") == "Hoàn tất"
    assert vi.value("completed", field="user_note") == "completed"
    assert vi.value("completed") == "completed"
    assert vi.value("unknown", field="status") == "unknown"
    assert vi.value(4, field="status") == 4
    assert vi.value(None, field="status") is None
    assert vi.value(pd.NA, field="status") is pd.NA
    assert vi.value({"status": "completed"}, field="status") == {"status": "completed"}


def test_display_frame_preserves_raw_data_types_index_and_downloads():
    vi = Translator(load_catalog(RESOURCES), "VI")
    raw = pd.DataFrame(
        {"status": ["completed", "alien"], "user_note": ["completed", "a,b"], "rows": [2, 3]},
        index=[7, 9],
    )
    before = raw.copy(deep=True)
    download = raw.to_csv(index=False)
    display = vi.frame(raw)
    assert list(display.columns) == ["Trạng thái", "user_note", "Số dòng"]
    assert display["Trạng thái"].tolist() == ["Hoàn tất", "alien"]
    assert display["user_note"].tolist() == before["user_note"].tolist()
    assert display["Số dòng"].dtype == before["rows"].dtype
    assert display.index.equals(before.index)
    pd.testing.assert_frame_equal(raw, before)
    assert raw.to_csv(index=False) == download


def test_colliding_headers_do_not_drop_or_hide_columns():
    vi = Translator(load_catalog(RESOURCES), "VI")
    raw = pd.DataFrame([[1, 2]], columns=["status", "Trạng thái"])
    display = vi.frame(raw)
    assert display.columns.is_unique
    assert display.shape == raw.shape
    assert list(display.columns) == list(raw.columns)


def test_metadata_localization_is_shallow_and_does_not_rewrite_unknown_content():
    vi = Translator(load_catalog(RESOURCES), "VI")
    raw = {
        "status": "completed",
        "source_file": "outputs/completed.csv",
        "details": {"status": "completed"},
    }
    assert vi.record(raw) == {
        "Trạng thái": "Hoàn tất",
        "Tệp nguồn": "outputs/completed.csv",
        "details": {"status": "completed"},
    }
    assert raw["status"] == "completed"


def test_new_resource_is_discovered_without_code_change(resources):
    data = json.loads((resources / "en.json").read_text())
    data.update(code="FR", name="Français")
    data["translations"]["auth.username"] = "Identifiant"
    (resources / "fr.json").write_text(json.dumps(data))
    catalog = load_catalog(resources)
    assert "FR" in catalog.languages
    assert Translator(catalog, "FR").text("auth.username") == "Identifiant"
